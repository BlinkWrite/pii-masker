import CryptoKit
import Foundation

/// Downloads, verifies, and installs the on-device model.
///
/// Fetches the `.tar.gz` a ``ModelPin`` names, checks its SHA-256, unpacks it, checks the SHA-256
/// of the unpacked weights, and flips a `current` symlink atomically — so a failed or tampered
/// download can never replace a working install.
///
/// There is no manifest and no version negotiation: the pin is compiled in, so the only question
/// this asks is "are the bytes the pin names already on disk?".
public actor ModelInstaller {

    public struct Outcome: Sendable {
        public let directory: URL
        /// False when the pinned version was already installed and complete: no download, no
        /// unpack, no swap — the bytes on disk are exactly the ones that were there before.
        public let installed: Bool
    }

    public enum Progress: Sendable {
        case checking
        case downloading(fraction: Double)
        case verifying
        case installing
        case upToDate(version: String)
        case installed(version: String)
    }

    public enum InstallError: Error, Sendable, CustomStringConvertible {
        case offline(String)
        /// The pin itself is unusable — an unsafe version string, an implausible size. A build
        /// defect rather than a runtime condition, and not fixable by retrying.
        case invalidPin(String)
        case checksumMismatch(expected: String, got: String)
        case downloadFailed(String)
        case unpackFailed(String)
        /// String is log-facing detail only — the user-visible copy never quotes a size.
        case outOfSpace(String)
        /// The payload was fine; this machine's model store could not be written. Distinct from
        /// `unpackFailed` because re-downloading cannot fix it — the retry would fail identically.
        case storeUnwritable(String)

        public var description: String {
            switch self {
            case .offline(let s): return "offline: \(s)"
            case .invalidPin(let s): return "invalid model pin: \(s)"
            case .checksumMismatch(let e, let g): return "checksum mismatch (expected \(e), got \(g))"
            case .downloadFailed(let s): return "download failed: \(s)"
            case .unpackFailed(let s): return "unpack failed: \(s)"
            case .outOfSpace(let s): return "out of space: \(s)"
            case .storeUnwritable(let s): return "model store unwritable: \(s)"
            }
        }
    }

    /// True only for URLError codes that mean "the device has no usable network" (Wi-Fi off,
    /// airplane mode, connection dropped) — NOT for a reachable network that can't reach the model
    /// host (host down, DNS miss, timeout, refused). That distinction is what lets a host app
    /// distinguish "no internet connection" from "we can't reach the download": the first is the
    /// user's network, the second is the publisher's.
    public static func isOfflineURLError(_ error: Error) -> Bool {
        guard let u = error as? URLError else { return false }
        switch u.code {
        case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed, .internationalRoamingOff:
            return true
        default:
            return false
        }
    }

    private let pin: ModelPin
    private let installRoot: URL
    private let session: URLSession

    /// - Parameters:
    ///   - pin: The model to install. Defaults to ``ModelPin/current``.
    ///   - installRoot: Where version directories and the `current` symlink are written. Required:
    ///     a library has no business guessing a location under the host's Application Support.
    public init(
        pin: ModelPin = .current,
        installRoot: URL,
        session: URLSession? = nil
    ) {
        self.pin = pin
        self.installRoot = installRoot
        self.session = session ?? ModelInstaller.defaultSession
    }

    /// A short-timeout, connectivity-failing session so an unreachable host fails in seconds
    /// instead of hanging on `.shared`'s 60s.
    private static let defaultSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 20
        // Whole-transfer budget. Without it the default is a WEEK, so a connection that stalls
        // without dropping leaves the install spinning with no error and no way to retry.
        cfg.timeoutIntervalForResource = 15 * 60
        cfg.waitsForConnectivity = false
        // Never cache model-install traffic: a cached response paired with a freshly
        // re-downloaded (too-big-to-cache) tarball yields a bogus checksum mismatch.
        cfg.urlCache = nil
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: cfg)
    }()

    /// Upper bound on a published archive, checked when the pin is read. Two orders of magnitude
    /// above the real model, so it only ever catches a malformed pin.
    public static let maxPlausibleArchiveBytes: Int64 = 100_000_000_000

    /// Everything ``PrivacyFilter`` opens — keep this in step with it. A dir holding only some of
    /// them must never count as installed: the loader fails, the fail-closed gate then holds every
    /// request, and nothing retries because both the installed-version check and the install itself
    /// thought they were done. Silent, unbadged, permanent.
    public static let requiredModelFiles = ["model.onnx", "tokenizer.json", "tokenizer_config.json"]

    public static func isCompleteModelDir(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return requiredModelFiles.allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    /// A full disk, however it reaches us. Cocoa reports `NSFileWriteOutOfSpaceError` and POSIX
    /// reports `ENOSPC`, either of which can sit on an underlying error rather than the top-level
    /// one — so walk the chain. `NSURLErrorCannotWriteToFile` is deliberately NOT treated as
    /// out-of-space on its own: URLSession raises it for an unwritable destination too, and telling
    /// someone with a read-only model store to free up space sends them after a fix that can never
    /// work.
    public static func isOutOfSpaceError(_ error: Error) -> Bool {
        var seen = 0
        var current: NSError? = error as NSError
        while let e = current, seen < 5 {
            if e.domain == NSCocoaErrorDomain && e.code == NSFileWriteOutOfSpaceError { return true }
            if e.domain == NSPOSIXErrorDomain && e.code == Int(ENOSPC) { return true }
            current = e.userInfo[NSUnderlyingErrorKey] as? NSError
            seen += 1
        }
        return false
    }

    /// A write refused for permissions rather than space. Same chain walk as
    /// ``isOutOfSpaceError(_:)``, because Cocoa and POSIX each report it at a different level. This
    /// is the dominant read-only-store failure — the move into the install root — and without it
    /// that store reports as a network problem, sending the user to retry a download the store can
    /// never accept.
    public static func isPermissionError(_ error: Error) -> Bool {
        var seen = 0
        var current: NSError? = error as NSError
        while let e = current, seen < 5 {
            if e.domain == NSCocoaErrorDomain
                && (e.code == NSFileWriteNoPermissionError || e.code == NSFileReadNoPermissionError)
            {
                return true
            }
            if e.domain == NSPOSIXErrorDomain && (e.code == Int(EACCES) || e.code == Int(EPERM)) { return true }
            current = e.userInfo[NSUnderlyingErrorKey] as? NSError
            seen += 1
        }
        return false
    }

    /// Free bytes on the install volume, by either of the two meanings Foundation offers.
    ///
    /// They answer different questions and this code needs both. `forImportantUsage` includes space
    /// macOS will purge on demand — the right (optimistic) number for deciding whether to *attempt*
    /// an install, since that space really is available. The plain key excludes it — the right
    /// (strict) number for explaining a write that has *already* failed, where purgeable space
    /// evidently didn't save us.
    ///
    /// Only the boot volume actually answers the `forImportantUsage` question. A disk image or any
    /// secondary volume reports **0** rather than declining to answer — indistinguishable from
    /// "full" — which refuses an install onto a store with hundreds of MB free and tells the user to
    /// clear space they already have. So a zero there means "this volume doesn't implement it", and
    /// the plain number is the only answer available.
    public static func availableBytes(at url: URL, purgeableCounts: Bool) -> Int64? {
        // Foundation caches resource values on the URL, and `installRoot` outlives every install —
        // so without this it answers with the capacity read at launch for the rest of the session.
        // A user who frees up space and hits retry would keep being told there isn't any.
        var probe = url
        probe.removeAllCachedResourceValues()
        let values = try? probe.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey,
        ])
        let plain = values?.volumeAvailableCapacity.map(Int64.init)
        guard purgeableCounts else { return plain }
        if let important = values?.volumeAvailableCapacityForImportantUsage, important > 0 {
            return important
        }
        return plain
    }

    private func sameVolume(_ a: URL, _ b: URL) -> Bool {
        let key: URLResourceKey = .volumeIdentifierKey
        var (x, y) = (a, b)
        x.removeAllCachedResourceValues()
        y.removeAllCachedResourceValues()
        guard let vx = try? x.resourceValues(forKeys: [key]).volumeIdentifier,
            let vy = try? y.resourceValues(forKeys: [key]).volumeIdentifier
        else { return true }   // unknown: assume shared, the stricter of the two budgets
        return vx.isEqual(vy)
    }

    /// The InstallError a transport failure should surface as. Collapsing every failure into
    /// `.downloadFailed` throws away the error code — so a full disk arrives at the UI as "we
    /// couldn't reach the server" and the user retries a download that could never fit.
    public static func installError(for error: Error) -> InstallError {
        if isOutOfSpaceError(error) { return .outOfSpace(error.localizedDescription) }
        if isPermissionError(error) { return .storeUnwritable(error.localizedDescription) }
        if isOfflineURLError(error) { return .offline(error.localizedDescription) }
        return .downloadFailed(error.localizedDescription)
    }

    /// The directory holding the ready-to-load model (the `current` symlink target), or nil if
    /// nothing is installed yet.
    public var currentModelDir: URL? {
        let link = installRoot.appendingPathComponent("current")
        guard FileManager.default.fileExists(atPath: link.path) else { return nil }
        return link.resolvingSymlinksInPath()
    }

    /// The installed version, read from the `current` symlink.
    public func installedVersion() -> String? {
        try? FileManager.default.destinationOfSymbolicLink(
            atPath: installRoot.appendingPathComponent("current").path)
    }

    /// Ensure the pinned model is installed; returns the model dir and whether this call is what
    /// put it there.
    ///
    /// Callers need `installed` because "succeeded" and "wrote new bytes" are different answers:
    /// the up-to-date branch succeeds having changed nothing, so anything keyed on the model having
    /// changed (a stale load failure, a cache, a probe) must not fire for it.
    ///
    /// - Parameters:
    ///   - force: Reinstall even when the installed version already matches — the recovery path for
    ///     a model whose files are all present but which the loader cannot actually use. Without it
    ///     the up-to-date branch reports success forever and the user has no way to replace it.
    ///   - skipping: Versions a rollback refused (``ModelRollback``). Only honoured while a usable
    ///     model is already installed: with nothing on disk there is no working model to protect,
    ///     and refusing the only pinned version would leave the host permanently without one.
    @discardableResult
    public func ensureLatest(
        force: Bool = false, skipping: Set<String> = [],
        onProgress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> Outcome {
        onProgress(.checking)
        // Before the up-to-date return below: a kill mid-download leaves a ~140 MB `.download-`
        // file, and for a host whose model is already current that path is never reached again — so
        // the leak would be permanent, and it is exactly what turns the next model release into
        // "not enough storage". Runs on every launch refresh; it is one directory listing.
        sweepPartials()

        guard Self.isSafePathComponent(pin.version) else {
            throw InstallError.invalidPin("unsafe version: \(pin.version)")
        }
        // `bytes` drives the disk budget and the figure quoted to the user. Reject an implausible
        // value at the boundary rather than carrying it into arithmetic and copy.
        guard pin.bytes > 0, pin.bytes < Self.maxPlausibleArchiveBytes else {
            throw InstallError.invalidPin("implausible archive size: \(pin.bytes)")
        }

        if skipping.contains(pin.version), let dir = currentModelDir,
            ModelInstaller.isCompleteModelDir(dir)
        {
            onProgress(.upToDate(version: installedVersion() ?? pin.version))
            return Outcome(directory: dir, installed: false)
        }

        if !force, installedVersion() == pin.version, let dir = currentModelDir,
            ModelInstaller.isCompleteModelDir(dir)
        {
            onProgress(.upToDate(version: pin.version))
            return Outcome(directory: dir, installed: false)
        }

        try FileManager.default.createDirectory(at: installRoot, withIntermediateDirectories: true)
        guard hasRoomFor(pin.bytes) else {
            throw InstallError.outOfSpace(
                "needs ~\(Self.requiredFreeBytes(forArchiveOf: pin.bytes) / 1_000_000) MB, archive \(pin.bytes / 1_000_000) MB")
        }
        // UUID-suffixed so two installs never write the same file (the versioned name alone would
        // let a second run clobber a first mid-download); `sweepPartials` clears any left behind.
        let archive = installRoot.appendingPathComponent(".download-\(pin.version)-\(UUID().uuidString).tar.gz")

        do {
            do {
                try await download(pin.sourceURL, to: archive, expectedBytes: pin.bytes) {
                    onProgress(.downloading(fraction: $0))
                }
            } catch {
                // A volume that filled mid-transfer surfaces as a bare `.cannotWriteToFile` with no
                // ENOSPC anywhere in the chain, which the static classifier cannot tell from a
                // server problem. Here both volumes are in scope, so ask them.
                let tmp = FileManager.default.temporaryDirectory
                if looksOutOfSpace(at: tmp, needing: pin.bytes)
                    || looksOutOfSpace(at: installRoot, needing: pin.bytes)
                {
                    throw InstallError.outOfSpace("volume filled during the download")
                }
                throw error
            }

            onProgress(.verifying)
            let got = try InstallSupport.sha256(ofFileAt: archive)
            guard got.caseInsensitiveCompare(pin.archiveSHA256) == .orderedSame else {
                throw InstallError.checksumMismatch(expected: pin.archiveSHA256, got: got)
            }

            onProgress(.installing)
            try install(archive: archive, archiveBytes: pin.bytes)
        } catch {
            try? FileManager.default.removeItem(at: archive)
            throw error
        }

        try? FileManager.default.removeItem(at: archive)
        onProgress(.installed(version: pin.version))
        return Outcome(
            directory: installRoot.appendingPathComponent(pin.version, isDirectory: true),
            installed: true)
    }

    // MARK: - steps

    /// See ``InstallSupport/isSafePathComponent(_:)``.
    public static func isSafePathComponent(_ v: String) -> Bool { InstallSupport.isSafePathComponent(v) }

    private func download(
        _ url: URL, to dest: URL, expectedBytes: Int64,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try? FileManager.default.removeItem(at: dest)
        if url.isFileURL {
            try FileManager.default.copyItem(at: url, to: dest)
            onProgress(1)
            return
        }
        let tmp = try await DownloadDelegate.run(
            url: url, session: session, expectedBytes: expectedBytes, onProgress: onProgress)
        do {
            try FileManager.default.moveItem(at: tmp, to: dest)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            throw ModelInstaller.installError(for: error)
        }
    }

    /// New disk the install needs at its peak: the downloaded archive (1×) plus what it unpacks to
    /// (~1.5× for this model), both present until the swap. An already-installed version is NOT
    /// counted — it occupies space we already hold, so requiring headroom for it would refuse
    /// installs that fit.
    public static func requiredFreeBytes(forArchiveOf bytes: Int64) -> Int64 {
        guard bytes > 0 else { return 0 }
        // `bytes` may come from a caller-built pin; multiply-then-halve would trap on a hostile value.
        let (product, overflowed) = bytes.multipliedReportingOverflow(by: 5)
        // Saturate to a large-but-sane value: Int64.max would be quoted as "9223372036860 MB".
        return overflowed ? Int64.max / 2 : product / 2
    }

    /// What the unpack alone needs, for quoting a shortfall discovered after the archive is already
    /// on disk.
    public static func unpackFreeBytes(forArchiveOf bytes: Int64) -> Int64 {
        guard bytes > 0 else { return 0 }
        let (product, overflowed) = bytes.multipliedReportingOverflow(by: 3)
        return overflowed ? Int64.max : product / 2
    }

    /// Refuse before spending the bandwidth when the volumes clearly can't hold the result. The
    /// archive lands in the TEMP volume and unpacks into the install volume — the same volume in
    /// the normal case, but not when the model store is redirected, and checking only one of them
    /// lets a full boot disk pass and then fail mid-download. Advisory only: an unreadable capacity
    /// is not treated as failure.
    private func hasRoomFor(_ bytes: Int64) -> Bool {
        guard bytes > 0 else { return true }
        let tmp = FileManager.default.temporaryDirectory
        if sameVolume(tmp, installRoot) {
            guard let available = Self.availableBytes(at: installRoot, purgeableCounts: true) else { return true }
            return available >= Self.requiredFreeBytes(forArchiveOf: bytes)
        }
        // temp holds the archive only while it transfers; installRoot then holds the archive AND
        // its unpacked contents at once, so it carries the full peak either way.
        let downloadOK = Self.availableBytes(at: tmp, purgeableCounts: true).map { $0 >= bytes } ?? true
        let unpackOK = Self.availableBytes(at: installRoot, purgeableCounts: true)
            .map { $0 >= Self.requiredFreeBytes(forArchiveOf: bytes) } ?? true
        return downloadOK && unpackOK
    }

    /// Whether a write that already failed did so for want of space. Uses the strict free-space
    /// number and the size still to be written: a volume with a few MB left is "full" for a 200 MB
    /// unpack, and telling that user their download was corrupted sends them retrying forever.
    private func looksOutOfSpace(at url: URL, needing bytes: Int64) -> Bool {
        guard let available = Self.availableBytes(at: url, purgeableCounts: false) else { return false }
        return available < bytes
    }

    private func install(archive: URL, archiveBytes: Int64) throws {
        let version = pin.version
        let fm = FileManager.default
        let staging = installRoot.appendingPathComponent(
            ".staging-\(version)-\(UUID().uuidString)", isDirectory: true)
        try? fm.removeItem(at: staging)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }

        // bsdtar handles .tar.gz.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        p.arguments = ["-xzf", archive.path, "-C", staging.path]
        let errPipe = Pipe()
        p.standardError = errPipe
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            let e = String(
                data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            // tar names a full disk only in its localised stderr, so ask the filesystem instead.
            // What is still needed at this point is the unpacked size — the archive is on disk.
            if looksOutOfSpace(at: installRoot, needing: Self.unpackFreeBytes(forArchiveOf: archiveBytes)) {
                throw InstallError.outOfSpace(
                    "unpack needs ~\(Self.unpackFreeBytes(forArchiveOf: archiveBytes) / 1_000_000) MB")
            }
            throw InstallError.unpackFailed(e)
        }
        guard ModelInstaller.isCompleteModelDir(staging) else {
            let present = (try? fm.contentsOfDirectory(atPath: staging.path))?.sorted().joined(separator: ", ") ?? "nothing"
            throw InstallError.unpackFailed(
                "archive is missing one of \(ModelInstaller.requiredModelFiles.joined(separator: ", ")) — got: \(present)")
        }

        // The weights, not just the archive. The archive hash is packer-dependent — bsdtar and GNU
        // tar produce different bytes from the same files — so it can only ever attest to one
        // publisher's tarball. The weights hash is the number the model card, the README and this
        // pin all quote, and it is what a reader can reproduce. Checked here, before the swap, so a
        // wrong-but-well-packed archive never becomes `current`.
        let unpackedWeights = staging.appendingPathComponent("model.onnx")
        let weightsHash = try InstallSupport.sha256(ofFileAt: unpackedWeights)
        guard weightsHash.caseInsensitiveCompare(pin.weightsSHA256) == .orderedSame else {
            throw InstallError.checksumMismatch(expected: pin.weightsSHA256, got: weightsHash)
        }

        let versionDir = installRoot.appendingPathComponent(version, isDirectory: true)
        if fm.fileExists(atPath: versionDir.path) {
            // replaceItemAt, not remove-then-move: the version being replaced can be the one
            // `current` points at (the repair path re-installs the same version), and a failed move
            // after a delete would leave the user with no model at all and a dangling symlink.
            let result = try fm.replaceItemAt(versionDir, withItemAt: staging)
            // It may hand back a different URL. The installed dir being named for its version is
            // the invariant `installedVersion()`, the up-to-date check and this function's return
            // all key on — following the new name instead would make every later launch
            // re-download.
            if let result, result.standardizedFileURL != versionDir.standardizedFileURL {
                // Never remove-then-move here: that reopens the window the line above exists to
                // close. Whatever sits at versionDir is the OLD content and stays until an atomic
                // swap replaces it, so a failure leaves the user with the model they already had.
                if fm.fileExists(atPath: versionDir.path) {
                    _ = try fm.replaceItemAt(versionDir, withItemAt: result)
                } else {
                    try fm.moveItem(at: result, to: versionDir)
                }
            }
        } else {
            try fm.moveItem(at: staging, to: versionDir)
        }
        guard ModelInstaller.isCompleteModelDir(versionDir) else {
            throw InstallError.storeUnwritable("installed dir is incomplete after the swap")
        }
        try pointCurrent(at: version)
    }

    // Superseded versions are deliberately NOT deleted. Each one costs ~200 MB, which is real, but
    // this code runs unattended at every launch and an install root can be shared: any rule for
    // "which directories are stale" is a rule for deleting a model something else is relying on, and
    // a wrong answer costs a working install and a 137 MB re-download. Model releases are rare, so
    // the disk cost is small and recoverable (delete the folder) while a bad delete is neither. It
    // is also what makes rollback cheap — the previous version is still there to flip back to.

    /// Remove partial downloads / staging dirs / half-swapped symlinks an interrupted run left
    /// behind (a mid-download crash leaks the ~140 MB `.download-…` file, which isn't OS-cleaned).
    /// The `current` symlink and installed version dirs never start with a dot, so they're
    /// untouched.
    private func sweepPartials() {
        let prefixes = [".download-", ".staging-", ".current-"]
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: installRoot.path) else { return }
        // Age-gated, because an install root can be shared: a second process installing at the same
        // moment has its own in-flight `.download-`/`.current-` entries here, and reaping those
        // breaks an install that is going fine. Anything genuinely abandoned is minutes old;
        // anything live is seconds.
        let cutoff = Date().addingTimeInterval(-15 * 60)
        for name in entries where prefixes.contains(where: name.hasPrefix) {
            let url = installRoot.appendingPathComponent(name)
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate
            guard let modified, modified < cutoff else { continue }
            try? fm.removeItem(at: url)
        }
    }

    private func pointCurrent(at version: String) throws {
        try Self.pointCurrent(at: version, in: installRoot)
    }

    /// Static because the rollback path (``ModelRollback``) runs it at launch, before there is an
    /// actor to hop from — and it needs nothing the actor protects: two filesystem calls, the
    /// second atomic.
    public static func pointCurrent(at version: String, in installRoot: URL) throws {
        guard isSafePathComponent(version) else {
            throw InstallError.storeUnwritable("unsafe version component")
        }
        let current = installRoot.appendingPathComponent("current")
        let tmpLink = installRoot.appendingPathComponent(".current-\(UUID().uuidString)")
        try FileManager.default.createSymbolicLink(atPath: tmpLink.path, withDestinationPath: version)
        // rename() replaces an existing symlink atomically on the same volume.
        guard rename(tmpLink.path, current.path) == 0 else {
            try? FileManager.default.removeItem(at: tmpLink)
            throw InstallError.storeUnwritable("could not update current symlink")
        }
    }

}

/// URLSessionDownloadTask wrapper: streams a large file to disk efficiently (the byte-by-byte
/// `URLSession.bytes` API is too slow for ~137 MB) while reporting progress. Kept out of the actor
/// so delegate callbacks don't hop actor isolation.
///
/// No `willPerformHTTPRedirection`, so URLSession's default follow applies — which is what a
/// Hugging Face `resolve/` URL needs, since it 302s to a CDN.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: (Double) -> Void
    private let expectedBytes: Int64
    private var continuation: CheckedContinuation<URL, Error>?
    private var movedURL: URL?

    private init(expectedBytes: Int64, onProgress: @escaping (Double) -> Void) {
        self.expectedBytes = expectedBytes
        self.onProgress = onProgress
    }

    static func run(
        url: URL, session: URLSession, expectedBytes: Int64,
        onProgress: @escaping (Double) -> Void
    ) async throws -> URL {
        let delegate = DownloadDelegate(expectedBytes: expectedBytes, onProgress: onProgress)
        // Inherit the caller's config (carries a URLProtocol mock in tests).
        let s = URLSession(configuration: session.configuration, delegate: delegate, delegateQueue: nil)
        return try await withCheckedThrowingContinuation { cont in
            delegate.continuation = cont
            s.downloadTask(with: url).resume()
        }
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : expectedBytes
        if total > 0 { onProgress(min(1, Double(totalBytesWritten) / Double(total))) }
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // A download task treats any response as success and writes the body to disk, so a 404/503
        // error page arrives here looking exactly like the archive. Left unchecked it reaches the
        // checksum and a server outage is reported to the user as a damaged download — "try again
        // for a fresh copy", forever. Resuming here leaves `continuation` nil, so
        // `didCompleteWithError` is a no-op.
        if let http = downloadTask.response as? HTTPURLResponse,
            !(200..<300).contains(http.statusCode)
        {
            continuation?.resume(
                throwing: ModelInstaller.InstallError.downloadFailed("HTTP \(http.statusCode)"))
            continuation = nil
            return
        }
        // The temp file is removed once this returns, so move it out synchronously.
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("pii-masker-model-\(UUID().uuidString).tar.gz")
        do {
            try FileManager.default.moveItem(at: location, to: dest)
            movedURL = dest
        } catch {
            continuation?.resume(throwing: ModelInstaller.installError(for: error))
            continuation = nil
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        defer { session.finishTasksAndInvalidate() }
        if let error {
            continuation?.resume(throwing: ModelInstaller.installError(for: error))
        } else if let movedURL {
            continuation?.resume(returning: movedURL)
        } else {
            continuation?.resume(throwing: ModelInstaller.InstallError.downloadFailed("no file"))
        }
        continuation = nil
    }
}

import Foundation
import Testing

@testable import PIIMasker

/// Hermetic tests for `ModelInstaller` — the download → verify → atomic-install pipeline. All
/// `file://` based (no network, no real weights), so they lock the contract that a good publish
/// installs, an idempotent re-run is a no-op, and a corrupted or tampered publish is rejected
/// without disturbing the working install.
@Suite("Model installer")
struct ModelInstallerTests {

    // MARK: - The happy path

    @Test func installsAndPointsCurrent() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "v1", marker: "onnx-v1", into: dist)
        let installer = ModelInstaller(pin: pin, installRoot: install)
        let dir = try await installer.ensureLatest().directory

        let text = (try? String(contentsOf: dir.appendingPathComponent("model.onnx"), encoding: .utf8)) ?? ""
        #expect(text == "onnx-v1", "\(text)")
        let version = await installer.installedVersion()
        #expect(version == "v1", "\(version ?? "nil")")
        #expect(FileManager.default.fileExists(
            atPath: install.appendingPathComponent("current/tokenizer.json").path))
        _ = trash
    }

    @Test func rerunIsANoOp() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "v1", marker: "onnx-v1", into: dist)
        let installer = ModelInstaller(pin: pin, installRoot: install)
        _ = try await installer.ensureLatest()

        let upToDate = Flag()
        let outcome = try await installer.ensureLatest { if case .upToDate = $0 { upToDate.value = true } }
        #expect(upToDate.value)
        #expect(!outcome.installed)
        _ = trash
    }

    @Test func newVersionSwapsCurrent() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        _ = try await ModelInstaller(
            pin: try TestPublish.publish(version: "v1", marker: "onnx-v1", into: dist),
            installRoot: install).ensureLatest()

        let inst2 = ModelInstaller(
            pin: try TestPublish.publish(version: "v2", marker: "onnx-v2", into: dist),
            installRoot: install)
        let dir = try await inst2.ensureLatest().directory
        let text = (try? String(contentsOf: dir.appendingPathComponent("model.onnx"), encoding: .utf8)) ?? ""
        #expect(text == "onnx-v2", "\(text)")
        let version = await inst2.installedVersion()
        #expect(version == "v2", "\(version ?? "nil")")
        _ = trash
    }

    // MARK: - Rejection

    /// A tampered archive is rejected and leaves the working install untouched (atomic).
    @Test func rejectsChecksumMismatchAtomically() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        _ = try await ModelInstaller(
            pin: try TestPublish.publish(version: "v1", marker: "onnx-v1", into: dist),
            installRoot: install).ensureLatest()

        let bad = try TestPublish.publish(
            version: "v2", marker: "onnx-v2", into: dist,
            forcedArchiveSHA: String(repeating: "0", count: 64))
        let instBad = ModelInstaller(pin: bad, installRoot: install)
        var rejected = false
        do { _ = try await instBad.ensureLatest() } catch {
            if case ModelInstaller.InstallError.checksumMismatch = error { rejected = true }
        }
        #expect(rejected)
        let version = await instBad.installedVersion()
        #expect(version == "v1", "\(version ?? "nil")")
        #expect(ModelInstaller.isCompleteModelDir(install.appendingPathComponent("current")))
        _ = trash
    }

    /// The check the archive hash cannot make. A tarball can hash correctly and still carry the
    /// wrong `model.onnx` — a different packer, a rebuilt archive, a substituted file inside an
    /// otherwise-honest publish. The weights hash is the number the model card and the README quote,
    /// so it is verified after the unpack and before the swap.
    @Test func rejectsWrongWeightsAfterUnpack() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(
            version: "w1", marker: "onnx-w1", into: dist,
            forcedWeightsSHA: String(repeating: "a", count: 64))
        var rejected = false
        do { _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest() } catch {
            if case ModelInstaller.InstallError.checksumMismatch = error { rejected = true }
        }
        #expect(rejected)
        #expect(!FileManager.default.fileExists(
            atPath: install.appendingPathComponent("current/model.onnx").path))
        _ = trash
    }

    /// A pin a caller built by hand can name an unsafe path component or an implausible size. Both
    /// are refused before any filesystem work.
    @Test func rejectsAnUnusablePin() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)
        let good = try TestPublish.publish(version: "v1", marker: "onnx-v1", into: dist)

        func expectInvalid(_ pin: ModelPin, _ what: String) async {
            var threw = false
            do { _ = try await ModelInstaller(pin: pin, installRoot: root.appendingPathComponent("i")).ensureLatest() }
            catch { if case ModelInstaller.InstallError.invalidPin = error { threw = true } }
            #expect(threw, "\(what)")
        }
        await expectInvalid(
            ModelPin(version: "../../etc", sourceURL: good.sourceURL,
                     archiveSHA256: good.archiveSHA256, weightsSHA256: good.weightsSHA256,
                     bytes: good.bytes, maxWidth: 12, maxSequenceLength: 768), "traversal version")
        await expectInvalid(
            ModelPin(version: "v9", sourceURL: good.sourceURL,
                     archiveSHA256: good.archiveSHA256, weightsSHA256: good.weightsSHA256,
                     bytes: 0, maxWidth: 12, maxSequenceLength: 768), "zero bytes")
        await expectInvalid(
            ModelPin(version: "v9", sourceURL: good.sourceURL,
                     archiveSHA256: good.archiveSHA256, weightsSHA256: good.weightsSHA256,
                     bytes: ModelInstaller.maxPlausibleArchiveBytes, maxWidth: 12, maxSequenceLength: 768), "implausible size")
        _ = trash
    }

    /// A version becomes an on-disk path component (the version dir, the `current` symlink target),
    /// so a traversal or slash must be rejected or a hostile pin could recursive-delete arbitrary
    /// directories.
    @Test func safePathComponentRules() {
        #expect(!ModelInstaller.isSafePathComponent("../../Users/x/Documents"))
        #expect(!ModelInstaller.isSafePathComponent(".."))
        #expect(!ModelInstaller.isSafePathComponent("."))
        #expect(!ModelInstaller.isSafePathComponent("a/b"))
        #expect(!ModelInstaller.isSafePathComponent(""))
        #expect(ModelInstaller.isSafePathComponent("c39036c"))
        #expect(ModelInstaller.isSafePathComponent("c39036c-dirty"))
        #expect(ModelInstaller.isSafePathComponent("2026.08.1"))
    }

    // MARK: - Completeness

    /// Every file the loader opens has to count as "installed". A dir missing any one of them loads
    /// nothing, and the fail-closed gate then holds every request — silently, with nothing retrying.
    /// So an incomplete archive must never be flipped into `current`.
    @Test(arguments: ["tokenizer.json", "tokenizer_config.json"])
    func incompleteArchiveIsRejected(missing: String) async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(
            version: "p1", marker: "onnx-p1", into: dist, omitting: missing)
        var rejected = false
        do { _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest() }
        catch { rejected = true }
        #expect(rejected)
        #expect(!FileManager.default.fileExists(
            atPath: install.appendingPathComponent("current/model.onnx").path))
        _ = trash
    }

    /// …and an install already on disk at the pinned version, but incomplete, must not short-circuit
    /// the up-to-date branch.
    @Test(arguments: ["tokenizer.json", "tokenizer_config.json"])
    func incompleteInstallIsRepaired(missing: String) async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "q1", marker: "onnx-q1", into: dist)
        _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()

        try? FileManager.default.removeItem(at: install.appendingPathComponent("current/\(missing)"))
        _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()
        #expect(FileManager.default.fileExists(
            atPath: install.appendingPathComponent("current/\(missing)").path))
        _ = trash
    }

    // MARK: - What must not be deleted

    /// Installing a new version must not delete anything. The install root is caller-supplied, so
    /// "which directories are stale" is unanswerable here. Old versions accumulate on purpose; that
    /// is recoverable, a wrong delete is not — and it is what makes a rollback a symlink flip.
    @Test func supersededVersionsAndSiblingsSurvive() async throws {
        let fm = FileManager.default
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try fm.createDirectory(at: dist, withIntermediateDirectories: true)
        try fm.createDirectory(at: install.appendingPathComponent("unrelated"), withIntermediateDirectories: true)
        try "keep me".data(using: .utf8)!.write(to: install.appendingPathComponent("unrelated/notes.txt"))

        _ = try await ModelInstaller(
            pin: try TestPublish.publish(version: "z1", marker: "onnx-z1", into: dist),
            installRoot: install).ensureLatest()
        _ = try await ModelInstaller(
            pin: try TestPublish.publish(version: "z2", marker: "onnx-z2", into: dist),
            installRoot: install).ensureLatest()

        #expect(ModelInstaller.isCompleteModelDir(install.appendingPathComponent("z1")))
        #expect(fm.fileExists(atPath: install.appendingPathComponent("unrelated/notes.txt").path))
        #expect(ModelInstaller.isCompleteModelDir(install.appendingPathComponent("current")))
        #expect((try? fm.destinationOfSymbolicLink(
            atPath: install.appendingPathComponent("current").path)) == "z2")
        _ = trash
    }

    // MARK: - Recovery

    /// A user must always be able to replace what is on disk. Without `force`, an install whose
    /// files are all present short-circuits as up-to-date — so a model that is complete but unusable
    /// (corrupt weights, a runtime that can't load it) could never be replaced, and the fail-closed
    /// gate would hold every request for good.
    @Test func forceReplacesAPresentButBrokenModel() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "f1", marker: "onnx-f1", into: dist)
        let installer = ModelInstaller(pin: pin, installRoot: install)
        _ = try await installer.ensureLatest()
        // Corrupt the installed copy in a way the file-presence checks cannot see.
        try "corrupted".data(using: .utf8)!.write(
            to: install.appendingPathComponent("current/model.onnx"))

        _ = try await installer.ensureLatest()
        let afterPlain = (try? String(
            contentsOf: install.appendingPathComponent("current/model.onnx"), encoding: .utf8)) ?? ""
        #expect(afterPlain == "corrupted", "\(afterPlain)")

        _ = try await installer.ensureLatest(force: true)
        let afterForce = (try? String(
            contentsOf: install.appendingPathComponent("current/model.onnx"), encoding: .utf8)) ?? ""
        #expect(afterForce == "onnx-f1", "\(afterForce)")
        _ = trash
    }

    /// `skipping` is honoured only while a usable model is installed: with nothing on disk there is
    /// no working model to protect, and refusing the only pinned version would leave the host
    /// permanently without one.
    @Test func skippingIsIgnoredWithNothingInstalled() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "s1", marker: "onnx-s1", into: dist)
        let outcome = try await ModelInstaller(pin: pin, installRoot: install)
            .ensureLatest(skipping: ["s1"])
        #expect(outcome.installed)
        #expect(ModelInstaller.isCompleteModelDir(install.appendingPathComponent("current")))

        // With a working install in place, the same skip is respected — no reinstall.
        let again = try await ModelInstaller(pin: pin, installRoot: install)
            .ensureLatest(force: true, skipping: ["s1"])
        #expect(!again.installed)
        _ = trash
    }

    // MARK: - Housekeeping

    /// Abandoned partials are swept even when the model is already current. A sweep that sat after
    /// the up-to-date early return meant a host that never installs again — the normal case — kept a
    /// killed download's ~140 MB for good, and that leak is what turns the next release into "not
    /// enough storage".
    @Test func sweepsPartialsOnAnUpToDateRun() async throws {
        let fm = FileManager.default
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try fm.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "f1", marker: "onnx-f1", into: dist)
        _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()

        let orphan = install.appendingPathComponent(".download-f1-ORPHAN.tar.gz")
        try Data(count: 1024).write(to: orphan)
        try fm.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-3600)], ofItemAtPath: orphan.path)

        let outcome = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()
        #expect(!outcome.installed)
        #expect(!fm.fileExists(atPath: orphan.path))
        _ = trash
    }

    /// …but not one that could still be another process's live download. The install root can be
    /// shared, and reaping a fresh `.download-` breaks an install that is going fine.
    @Test func sweepSparesAFreshPartial() async throws {
        let fm = FileManager.default
        let root = TestPublish.scratch(); let trash = Trash(root)
        let dist = root.appendingPathComponent("dist")
        let install = root.appendingPathComponent("install")
        try fm.createDirectory(at: dist, withIntermediateDirectories: true)

        let pin = try TestPublish.publish(version: "f1", marker: "onnx-f1", into: dist)
        _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()

        let live = install.appendingPathComponent(".download-f1-LIVE.tar.gz")
        try Data(count: 1024).write(to: live)
        _ = try await ModelInstaller(pin: pin, installRoot: install).ensureLatest()
        #expect(fm.fileExists(atPath: live.path))
        _ = trash
    }

    // MARK: - Budgets and classification

    /// `bytes` comes from a pin a caller can write, so the headroom maths must not trap on a hostile
    /// value — and the number the code DECIDES on has to match the number it QUOTES. Detecting a
    /// full disk with the archive size while quoting the unpack size is how a disk that fills during
    /// tar kept being reported as a corrupted download.
    @Test func headroomBudgets() {
        #expect(ModelInstaller.requiredFreeBytes(forArchiveOf: 100_000_000) == 250_000_000)
        #expect(ModelInstaller.unpackFreeBytes(forArchiveOf: 100_000_000) == 150_000_000)
        #expect(ModelInstaller.requiredFreeBytes(forArchiveOf: 0) == 0)
        #expect(ModelInstaller.unpackFreeBytes(forArchiveOf: 0) == 0)
        // Saturating rather than trapping.
        #expect(ModelInstaller.requiredFreeBytes(forArchiveOf: Int64.max) > 0)
        #expect(ModelInstaller.unpackFreeBytes(forArchiveOf: Int64.max) > 0)
    }

    /// The download path converts the transport error at the throw site, so a classifier that only
    /// understands raw NSErrors never sees a full disk: it would arrive as a `.downloadFailed`
    /// string and the user would be told the server is unreachable.
    @Test func transportErrorsAreClassifiedAtTheThrowSite() {
        let posix = NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))
        let full = URLError(.cannotWriteToFile, userInfo: [NSUnderlyingErrorKey: posix])
        if case .outOfSpace = ModelInstaller.installError(for: full) {} else {
            Issue.record("a full disk should survive conversion at the download site")
        }
        // A write refused for permissions is NOT a full disk. Telling that user to free up space
        // sends them after a fix that can never work.
        #expect(!ModelInstaller.isOutOfSpaceError(URLError(.cannotWriteToFile)))

        let denied = NSError(
            domain: NSCocoaErrorDomain, code: NSFileWriteNoPermissionError,
            userInfo: [NSUnderlyingErrorKey: NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES))])
        if case .storeUnwritable = ModelInstaller.installError(for: denied) {} else {
            Issue.record("a permission-denied write should classify as storeUnwritable")
        }
        if case .storeUnwritable = ModelInstaller.installError(
            for: NSError(domain: NSPOSIXErrorDomain, code: Int(EPERM))) {} else {
            Issue.record("a POSIX EPERM write should classify as storeUnwritable")
        }
        #expect(ModelInstaller.isOfflineURLError(URLError(.notConnectedToInternet)))
        #expect(!ModelInstaller.isOfflineURLError(URLError(.cannotFindHost)))
    }

    /// Capacity has to be read fresh every time. Foundation caches resource values on the URL, and
    /// the installer's `installRoot` outlives every install — so a stale reading kept telling a user
    /// who had just freed up space that there still wasn't any, for the rest of the session.
    @Test func capacityIsReadFresh() throws {
        let fm = FileManager.default
        let root = TestPublish.scratch(); let trash = Trash(root)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)

        let before = ModelInstaller.availableBytes(at: root, purgeableCounts: false) ?? 0
        let filler = root.appendingPathComponent("filler.bin")
        try Data(count: 120_000_000).write(to: filler)
        let afterFill = ModelInstaller.availableBytes(at: root, purgeableCounts: false) ?? 0
        #expect(afterFill < before, "\(before) → \(afterFill)")
        try fm.removeItem(at: filler)
        let afterFree = ModelInstaller.availableBytes(at: root, purgeableCounts: false) ?? 0
        #expect(afterFree > afterFill, "\(afterFill) → \(afterFree)")
        _ = trash
    }
}

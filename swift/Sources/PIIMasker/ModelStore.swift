import Foundation

/// Where the model lives, and what the loader last concluded about it.
///
/// Everything here is process-wide on purpose. The installer writes bytes; a long-lived masker
/// holds an ONNX session built from the bytes that were there when it loaded; a host's status
/// badge watches for changes. Those three never meet, and per-instance state would let them
/// disagree — most dangerously after a rollback, where a masker would keep masking with the very
/// version the probe just rejected. See ``generation``.
public enum ModelStore {

    // MARK: - Location

    /// Where ``PrivacyFilter/resolvedModelDirectory()`` looks for a model, in order. The first
    /// directory holding ALL of ``ModelInstaller/requiredModelFiles`` wins.
    ///
    /// Answering per file instead would let a directory missing one of them still report installed
    /// by resolving that file from a different candidate — and the loader, which reads them from a
    /// single folder, would then fail with nothing to retry it.
    public struct Location: Sendable {
        /// The install root ``ModelInstaller`` writes into. Its `current` symlink is searched
        /// first, resolved through the link so a caller holds a concrete version directory: a
        /// refresh flipping `current` afterwards cannot change what an in-flight load is reading.
        public let installRoot: URL?
        /// Searched after the install root — a copy bundled with the host app, a checkout in a
        /// development tree. Order is preserved.
        public let fallbacks: [URL]

        public init(installRoot: URL?, fallbacks: [URL] = []) {
            self.installRoot = installRoot
            self.fallbacks = fallbacks
        }

        /// Nothing configured. Every lookup answers "no model", so the fail-closed gate holds
        /// everything — the correct behaviour for a host that forgot to call ``configure(_:)``.
        public static let unconfigured = Location(installRoot: nil)
    }

    /// A benign race here costs one extra disk check, so this is `nonisolated(unsafe)` rather than
    /// a lock. Set it once at startup, before any masker is built.
    public nonisolated(unsafe) private(set) static var location: Location = .unconfigured

    /// Point the library at a model store. Call once, at startup. Re-configuring invalidates
    /// everything the loader concluded about the previous location.
    public static func configure(_ location: Location) {
        self.location = location
        invalidate()
    }

    /// Convenience for the common shape: one install root, optional fallbacks.
    public static func configure(installRoot: URL, fallbacks: [URL] = []) {
        configure(Location(installRoot: installRoot, fallbacks: fallbacks))
    }

    /// The one directory every model file is loaded from, or nil if no candidate is complete.
    public static func resolvedModelDirectory() -> URL? {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let root = location.installRoot {
            candidates.append(root.appendingPathComponent("current").resolvingSymlinksInPath())
        }
        candidates.append(contentsOf: location.fallbacks)
        return candidates.first { candidate in
            ModelInstaller.requiredModelFiles.allSatisfy {
                fm.fileExists(atPath: candidate.appendingPathComponent($0).path)
            }
        }
    }

    /// Whether a model is present. Cheap disk check — a caller's per-request gate uses it to
    /// avoid sending unmasked context before the model has been installed.
    ///
    /// Once the model is on disk it stays there for the process's life, so the positive result is
    /// cached: this runs many times a second while a user types and otherwise `stat()`s three files
    /// each call. Only false→true is cached (a monotonic transition), so a benign data race just
    /// repeats the disk check.
    public static func modelIsInstalled() -> Bool {
        if installedCache { return true }
        let installed = resolvedModelDirectory() != nil
        if installed { installedCache = true }
        return installed
    }

    private nonisolated(unsafe) static var installedCache = false

    // MARK: - State

    /// What the loader knows about the model on disk.
    ///
    /// Only the loader may report `.loaded` or `.unusable`, because only the loader has actually
    /// opened the files — everyone else can invalidate but never certify. That asymmetry is the
    /// point: letting the installer clear the failure flag directly means a refresh that found the
    /// model already up to date and wrote nothing still declares it healthy, putting an unloadable
    /// model back into silence.
    ///
    /// `.unknown` is "no load attempted since the bytes last changed", not "fine" — the absence of
    /// a model is a disk question the loader cannot answer until it tries.
    public enum State: Equatable, Sendable { case unknown, loaded, unusable }

    public nonisolated(unsafe) static var state: State = .unknown {
        didSet { if state != oldValue { onStateChange?() } }
    }

    /// Set by the host so a status badge is recomputed the moment the state moves. The load runs
    /// asynchronously off launch, so whether it lands before or after the host's last refresh is a
    /// race — and it is lost whenever the ONNX session builds successfully and only the tokenizer
    /// fails. Without this push the user sees a healthy-looking app with no badge and no retry,
    /// silently holding every request.
    public nonisolated(unsafe) static var onStateChange: (@Sendable () -> Void)?

    /// Bumped whenever the bytes on disk change. Maskers compare it against the generation they
    /// loaded at, so a session built from the old bytes is dropped rather than reused.
    internal nonisolated(unsafe) private(set) static var generation = 0

    /// The bytes on disk changed, so whatever the loader last concluded no longer describes them —
    /// and neither does any session already built from them.
    public static func invalidate() {
        state = .unknown
        installedCache = false
        generation &+= 1
    }

    /// Reported by the loader when the files were all present but could not be opened. Nothing
    /// else can detect this — ``modelIsInstalled()`` only sees the files — so without recording it
    /// the host looks healthy while masking nothing, with no affordance to recover.
    internal static func markUnusable() {
        state = .unusable
        installedCache = false
    }

    internal static func markLoaded() {
        state = .loaded
    }
}

import Foundation
import os

/// Probation for a freshly-installed model version, and the revert back to the one it replaced.
///
/// The payload is already on disk — superseded version dirs are deliberately never pruned — so the
/// record is just the version pair, and reverting is one atomic `rename()` of the `current`
/// symlink.
///
/// A bad model does not crash. Either the loader fails and the fail-closed gate holds every
/// request, so the host looks alive while doing nothing; or, worse, the model loads and detects
/// nothing and the gate stops closing at all. So the health signal is not "did we survive" but
/// "does it still find PII" — ``PrivacyFilter/probe()``. The one failure that *can* crash (ONNX
/// Runtime dying on a malformed graph) is caught by the launch counter instead, since nothing
/// in-process gets to report a verdict there.
///
/// **Tested, not battle-tested.** ``arm(new:previous:)`` only fires when a previous version exists,
/// which needs at least two published models. The arm → verify → revert chain is exercised by the
/// test suite; ``isPending``, ``resolveAtLaunch()``, ``clearBlocklist()`` and ``blockedVersions``
/// run on every launch but always find nothing armed.
public struct ModelRollback {
    /// Launches an unresolved probation gets before the version is reverted unprobed. Two, not one:
    /// an unrelated crash during launch would otherwise blocklist a model that was never at fault.
    public static let maxLaunchAttempts = 2

    /// The two `UserDefaults` keys this type owns. Passed in so a host can keep its own namespace —
    /// and so an existing host can keep reading records it wrote under its old key.
    public struct SettingsKeys: Sendable {
        public let probation: String
        public let failedVersions: String

        public init(probation: String, failedVersions: String) {
            self.probation = probation
            self.failedVersions = failedVersions
        }

        /// The library's own key names, used when a host supplies none of its own.
        ///
        /// The .NET target's `SettingsKeys.Default` holds these same two strings and has to keep
        /// agreeing with them: both targets can be pointed at one store, and a rollback record
        /// written by either has to be the record the other reads.
        ///
        /// They are persisted `UserDefaults` keys, so changing them once something relies on them
        /// abandons what was written under the old names — a model part-way through probation stops
        /// being watched, and versions already known to fail get offered again. Nothing relies on
        /// them yet, which is why they could still be renamed with the repository: the one shipping
        /// consumer passes its own pair, which is what this fallback exists for.
        public static let `default` = SettingsKeys(
            probation: "com.github.pii-masker.modelUpdate.probation",
            failedVersions: "com.github.pii-masker.modelUpdate.failedVersions")
    }

    public enum LaunchOutcome: Equatable {
        case none
        case proving
        /// Reverted at launch without probing — the previous launches died before reaching a verdict.
        case rolledBack(from: String)
    }

    private let installRoot: URL
    private let defaults: UserDefaults
    private let keys: SettingsKeys
    private let log: os.Logger

    public init(
        installRoot: URL,
        defaults: UserDefaults = .standard,
        keys: SettingsKeys = .default,
        logging: MaskerLogging = .silent
    ) {
        self.installRoot = installRoot
        self.defaults = defaults
        self.keys = keys
        self.log = logging.logger("model-install")
    }

    // MARK: - Record

    /// The version being proved, the one it replaced, and how many launches it has had without
    /// reaching a verdict. One record, so a torn write can't pair a version with the previous one's
    /// attempt count.
    public var probation: (new: String, previous: String)? {
        record.map { ($0.new, $0.previous) }
    }

    private var record: (new: String, previous: String, attempts: Int)? {
        guard let raw = defaults.dictionary(forKey: keys.probation),
              let new = raw["new"] as? String, let previous = raw["previous"] as? String
        else { return nil }
        return (new, previous, raw["attempts"] as? Int ?? 0)
    }

    private func write(new: String, previous: String, attempts: Int) {
        defaults.set(
            ["new": new, "previous": previous, "attempts": attempts],
            forKey: keys.probation)
    }

    public var isPending: Bool { probation != nil }

    public var blockedVersions: Set<String> {
        Set(defaults.stringArray(forKey: keys.failedVersions) ?? [])
    }

    /// Only armed when there is something to go back to — either the previous version's bytes are
    /// already on disk, or it is an entry in ``ModelPin/known`` that can be fetched again. A first
    /// install and a same-version repair have no previous version at all, so a failure there is a
    /// reinstall, not a rollback.
    public func arm(new: String, previous: String) {
        guard new != previous, canRevert(to: previous) else { return }
        write(new: new, previous: previous, attempts: 0)
    }

    private func canRevert(to version: String) -> Bool {
        if ModelInstaller.isCompleteModelDir(installRoot.appendingPathComponent(version)) { return true }
        return ModelPin.known.contains { $0.version == version }
    }

    public func confirm() {
        defaults.removeObject(forKey: keys.probation)
    }

    public func block(_ version: String) {
        var blocked = defaults.stringArray(forKey: keys.failedVersions) ?? []
        guard !blocked.contains(version) else { return }
        blocked.append(version)
        defaults.set(blocked, forKey: keys.failedVersions)
    }

    /// A user asking for a reinstall outranks our verdict — they may know the failure was
    /// environmental (a half-written disk, a killed process), and without this the only pinned
    /// version stays refused forever with no way to say "try again".
    public func clearBlocklist() {
        defaults.removeObject(forKey: keys.failedVersions)
    }

    // MARK: - Launch

    /// Cheap and synchronous, for the top of `main()`: no model load, no inference — just "has this
    /// probation already burned its launches?". The probe itself runs later, off the launch path.
    public func resolveAtLaunch() -> LaunchOutcome {
        guard let current = record else { return .none }
        guard current.attempts < Self.maxLaunchAttempts else {
            log.error("model \(current.new, privacy: .public) never reached a verdict in \(current.attempts) launches; reverting to \(current.previous, privacy: .public)")
            return revert() ? .rolledBack(from: current.new) : .none
        }
        write(new: current.new, previous: current.previous, attempts: current.attempts + 1)
        return .proving
    }

    // MARK: - Verdict

    public enum Verdict: Equatable {
        case idle          // nothing on probation
        case verified
        case reverted
        /// The probe could not run at all. Deliberately not a failure: the model may be fine and the
        /// machine merely out of memory or mid-swap. Left armed so the next launch tries again — and
        /// if it never gets a verdict, ``resolveAtLaunch()`` reverts it once the launches run out.
        case inconclusive
    }

    /// Load the model on probation and ask it to find known PII. Applies the verdict only if the
    /// record still names the version this started on — an install landing mid-probe moves the
    /// target, and applying a stale verdict would blocklist the wrong version.
    ///
    /// `probe` is injectable because the failure this whole mechanism exists for — a model that
    /// loads perfectly and quietly stops detecting PII — cannot be produced on demand: it needs a
    /// valid ONNX graph that under-detects. Every model a test can build fails to load instead,
    /// which is a different branch.
    @discardableResult
    public func verify(
        probe: (@Sendable (URL) async -> PrivacyFilter.ProbeResult)? = nil
    ) async -> Verdict {
        guard let started = probation else { return .idle }
        // Pinned to this store rather than resolved: a verdict formed against a different install
        // root would revert and blocklist a version it never actually loaded.
        let dir = installRoot.appendingPathComponent("current").resolvingSymlinksInPath()
        let result = await (probe ?? { await PrivacyFilter(modelDirectory: $0).probe() })(dir)
        guard probation?.new == started.new else { return .idle }
        if result.passes {
            log.notice("model \(started.new, privacy: .public) verified (found \(result.found.count)/\(PrivacyFilter.probeAnchors.count))")
            confirm()
            return .verified
        }
        // A probe that never ran is only a verdict about the model if the files were all there to
        // begin with: then they were opened and were not a model, which is this version,
        // definitively. A store that is missing files or was repointed mid-probe says nothing about
        // it, and blocklisting on that would pin the machine to an older version over a one-off
        // hiccup.
        guard result.ran || ModelInstaller.isCompleteModelDir(dir) else {
            log.error("model \(started.new, privacy: .public) could not be probed; leaving it on probation for another launch")
            return .inconclusive
        }
        log.error("model \(started.new, privacy: .public) failed its probe (missed: \(result.missed.joined(separator: ", "), privacy: .public)); reverting to \(started.previous, privacy: .public)")
        return revert() ? .reverted : .idle
    }

    // MARK: - Revert

    /// The pinned model a revert would have to download because its bytes are not on disk, or nil
    /// when the flip needs no network. Unreachable until a second version is published; see
    /// ``revertInstallingIfNeeded()``.
    public var pendingRevertDownload: ModelPin? {
        guard let probation else { return nil }
        let dir = installRoot.appendingPathComponent(probation.previous)
        guard !ModelInstaller.isCompleteModelDir(dir) else { return nil }
        return ModelPin.known.first { $0.version == probation.previous }
    }

    /// Flip `current` back and refuse the version that failed. No network: this is one `rename()`.
    ///
    /// Verified before the flip — a previous version that has since been deleted or truncated would
    /// take a working-but-suspect model and replace it with nothing at all, which is strictly worse
    /// than what we're recovering from. When those bytes are gone but the pin list still knows the
    /// version, the probation record is LEFT ARMED so ``revertInstallingIfNeeded()`` can fetch it;
    /// when nothing can bring it back, probation is cleared rather than retried forever.
    @discardableResult
    public func revert() -> Bool {
        guard let probation else { return false }
        let previousDir = installRoot.appendingPathComponent(probation.previous)
        guard ModelInstaller.isCompleteModelDir(previousDir) else {
            if pendingRevertDownload != nil {
                log.error("model \(probation.previous, privacy: .public) is not on disk; a revert needs to re-download it")
                return false
            }
            log.error("cannot revert model to \(probation.previous, privacy: .public): that version is gone")
            confirm()
            return false
        }
        do {
            try ModelInstaller.pointCurrent(at: probation.previous, in: installRoot)
        } catch {
            log.error("model revert failed: \(String(describing: error), privacy: .public)")
            return false
        }
        finishRevert(from: probation.new, to: probation.previous)
        return true
    }

    /// Revert, downloading the previous version first if its bytes are no longer on disk.
    ///
    /// The flip path is the normal one and costs no network. The fetch path exists because a pin
    /// list can name a version this machine never installed — it becomes reachable the first time a
    /// second model version ships, and until then it is dead code kept honest by the tests.
    @discardableResult
    public func revertInstallingIfNeeded(session: URLSession? = nil) async -> Bool {
        if revert() { return true }
        guard let probation, let pin = pendingRevertDownload else { return false }
        do {
            _ = try await ModelInstaller(pin: pin, installRoot: installRoot, session: session)
                .ensureLatest(force: true)
        } catch {
            log.error("model revert download failed: \(String(describing: error), privacy: .public)")
            return false
        }
        finishRevert(from: probation.new, to: probation.previous)
        return true
    }

    private func finishRevert(from new: String, to previous: String) {
        block(new)
        confirm()
        ModelStore.invalidate()
        log.notice("model reverted to \(previous, privacy: .public); \(new, privacy: .public) blocked")
    }
}

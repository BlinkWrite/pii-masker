import Foundation
import Testing

@testable import PIIMasker

/// Hermetic tests for `ModelRollback`: fake versioned model dirs under a temp install root and a
/// scratch defaults suite. No model, no network.
///
/// The probe half — does the *real* model still find PII — is in `ModelProbeTests`, which needs
/// weights on disk.
@Suite("Model rollback")
struct ModelRollbackTests {

    /// A version dir holding every file the loader opens — enough for `isCompleteModelDir`, which is
    /// what the revert checks before flipping back to it.
    @discardableResult
    static func installVersion(_ version: String, marker: String, into root: URL) throws -> URL {
        let dir = root.appendingPathComponent(version, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try marker.data(using: .utf8)!.write(to: dir.appendingPathComponent("model.onnx"))
        for f in ModelInstaller.requiredModelFiles where f != "model.onnx" {
            try "{}".data(using: .utf8)!.write(to: dir.appendingPathComponent(f))
        }
        return dir
    }

    static func currentVersion(in root: URL) -> String? {
        try? FileManager.default.destinationOfSymbolicLink(
            atPath: root.appendingPathComponent("current").path)
    }

    /// A fresh install root plus its own defaults suite, so tests never see each other's records.
    static func fixture() throws -> (install: URL, defaults: UserDefaults, rollback: ModelRollback, trash: Trash) {
        let root = TestPublish.scratch()
        let install = root.appendingPathComponent("install", isDirectory: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        let suite = "com.github.pii-masker.tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        return (install, defaults, ModelRollback(installRoot: install, defaults: defaults), Trash(root))
    }

    // MARK: - Arming

    /// Arming needs somewhere to go back to: a first install and a same-version repair do not
    /// qualify, so a failure there is a reinstall rather than a rollback.
    @Test func armingNeedsSomewhereToGoBackTo() throws {
        let f = try Self.fixture()
        try Self.installVersion("v1", marker: "model-v1", into: f.install)
        try ModelInstaller.pointCurrent(at: "v1", in: f.install)

        f.rollback.arm(new: "v1", previous: "v1")
        #expect(!f.rollback.isPending)

        f.rollback.arm(new: "v2", previous: "gone")
        #expect(!f.rollback.isPending)
        _ = f.trash
    }

    /// A real version change arms, and only the launch counter advances until a verdict lands.
    @Test func aVersionChangeArmsAndCountsLaunches() throws {
        let f = try Self.fixture()
        try Self.installVersion("v1", marker: "model-v1", into: f.install)
        try Self.installVersion("v2", marker: "model-v2", into: f.install)
        try ModelInstaller.pointCurrent(at: "v2", in: f.install)

        f.rollback.arm(new: "v2", previous: "v1")
        #expect(f.rollback.isPending)
        #expect(f.rollback.probation?.new == "v2")
        #expect(f.rollback.probation?.previous == "v1")
        #expect(f.rollback.resolveAtLaunch() == .proving)
        #expect(f.rollback.resolveAtLaunch() == .proving)
        _ = f.trash
    }

    /// Out of launches: revert without probing. This is the crash case — nothing in-process ever
    /// gets to report a verdict, so the launch counter is the only signal there is.
    @Test func outOfLaunchesRevertsUnprobed() throws {
        let f = try Self.fixture()
        try Self.installVersion("v1", marker: "model-v1", into: f.install)
        try Self.installVersion("v2", marker: "model-v2", into: f.install)
        try ModelInstaller.pointCurrent(at: "v2", in: f.install)
        f.rollback.arm(new: "v2", previous: "v1")
        _ = f.rollback.resolveAtLaunch()
        _ = f.rollback.resolveAtLaunch()

        #expect(f.rollback.resolveAtLaunch() == .rolledBack(from: "v2"))
        #expect(Self.currentVersion(in: f.install) == "v1")
        #expect(f.rollback.blockedVersions == ["v2"])
        #expect(!f.rollback.isPending)
        #expect(f.rollback.resolveAtLaunch() == .none)
        _ = f.trash
    }

    // MARK: - The blocklist

    /// A blocked version is not reinstalled over a working model…
    @Test func aBlockedVersionIsSkippedOverAWorkingModel() async throws {
        let f = try Self.fixture()
        let dist = f.install.deletingLastPathComponent().appendingPathComponent("dist")
        try FileManager.default.createDirectory(at: dist, withIntermediateDirectories: true)
        try Self.installVersion("v1", marker: "model-v1", into: f.install)
        try ModelInstaller.pointCurrent(at: "v1", in: f.install)
        f.rollback.block("v2")

        let pin = try TestPublish.publish(version: "v2", marker: "model-v2", into: dist)
        let skipped = try await ModelInstaller(pin: pin, installRoot: f.install)
            .ensureLatest(skipping: f.rollback.blockedVersions)
        #expect(!skipped.installed)
        #expect(Self.currentVersion(in: f.install) == "v1")

        // …and an explicit retry clears the refusal: the user outranks our verdict, and otherwise
        // the only pinned version stays refused forever with no way to say "try again".
        f.rollback.clearBlocklist()
        #expect(f.rollback.blockedVersions.isEmpty)
        let retried = try await ModelInstaller(pin: pin, installRoot: f.install)
            .ensureLatest(skipping: f.rollback.blockedVersions)
        #expect(retried.installed)
        #expect(Self.currentVersion(in: f.install) == "v2")
        _ = f.trash
    }

    // MARK: - Reverting

    /// A revert with nowhere to go back to must not strand the host on no model at all — and the
    /// record is dropped rather than retried forever.
    @Test func aRevertWithNowhereToGoIsAbandoned() throws {
        let f = try Self.fixture()
        try Self.installVersion("v2", marker: "model-v2", into: f.install)
        try ModelInstaller.pointCurrent(at: "v2", in: f.install)
        // Plant the record directly: `arm` would refuse, which is the previous assertion.
        f.defaults.set(
            ["new": "v2", "previous": "v1"],
            forKey: ModelRollback.SettingsKeys.default.probation)

        #expect(!f.rollback.revert())
        #expect(Self.currentVersion(in: f.install) == "v2")
        #expect(!f.rollback.isPending)
        _ = f.trash
    }

    /// The second revert mode. When the previous version's bytes are gone but the pin list still
    /// names it, the record is LEFT ARMED so it can be fetched rather than abandoned — the branch
    /// that becomes reachable the first time a second model version ships.
    @Test func aRevertToAKnownPinIsHeldForDownload() throws {
        let f = try Self.fixture()
        let known = ModelPin.current.version
        try Self.installVersion("next", marker: "model-next", into: f.install)
        try ModelInstaller.pointCurrent(at: "next", in: f.install)
        f.defaults.set(
            ["new": "next", "previous": known],
            forKey: ModelRollback.SettingsKeys.default.probation)

        #expect(f.rollback.pendingRevertDownload?.version == known)
        #expect(!f.rollback.revert())
        // Still armed: unlike the vanished-version case, this one has a recovery.
        #expect(f.rollback.isPending)
        #expect(Self.currentVersion(in: f.install) == "next")
        _ = f.trash
    }

    /// …and `arm` accepts such a pair, where it would refuse a version nothing can bring back.
    @Test func armAcceptsAKnownPinNotOnDisk() throws {
        let f = try Self.fixture()
        f.rollback.arm(new: "next", previous: ModelPin.current.version)
        #expect(f.rollback.isPending)

        f.rollback.confirm()
        f.rollback.arm(new: "next", previous: "never-published")
        #expect(!f.rollback.isPending)
        _ = f.trash
    }

    // MARK: - Verdicts

    /// A probe that could not run for reasons that say nothing about the model — the store not
    /// resolving, an install repointing `current` underneath it — must not blocklist a version. Only
    /// the loader having opened the files and found they were not a model is a verdict.
    @Test func anUnprobeableModelIsInconclusive() async throws {
        let f = try Self.fixture()
        try Self.installVersion("m1", marker: "x", into: f.install)
        try ModelInstaller.pointCurrent(at: "m1", in: f.install)
        f.rollback.arm(new: "m2", previous: "m1")
        try FileManager.default.removeItem(at: f.install.appendingPathComponent("current"))

        let verdict = await f.rollback.verify { _ in
            PrivacyFilter.ProbeResult(found: [], missed: PrivacyFilter.probeAnchors, ran: false)
        }
        #expect(verdict == .inconclusive, "\(verdict)")
        #expect(f.rollback.isPending)
        #expect(f.rollback.blockedVersions.isEmpty)
        _ = f.trash
    }

    /// THE failure this exists for: a model that loads perfectly and quietly detects less. No crash,
    /// no error — the fail-closed gate simply stops closing and PII flows onward. It cannot be built
    /// on demand (it needs a valid graph that under-detects), so the probe is handed in.
    @Test func aModelThatUnderDetectsIsReverted() async throws {
        let f = try Self.fixture()
        try Self.installVersion("m8", marker: "masks-properly", into: f.install)
        try Self.installVersion("m9", marker: "under-detecting", into: f.install)
        try ModelInstaller.pointCurrent(at: "m9", in: f.install)
        f.rollback.arm(new: "m9", previous: "m8")
        #expect(f.rollback.isPending)

        let verdict = await f.rollback.verify { _ in
            PrivacyFilter.ProbeResult(
                found: [PrivacyFilter.probeAnchors[0]],
                missed: Array(PrivacyFilter.probeAnchors.dropFirst()), ran: true)
        }
        #expect(verdict == .reverted, "\(verdict)")
        #expect(Self.currentVersion(in: f.install) == "m8")
        #expect(f.rollback.blockedVersions.contains("m9"))
        _ = f.trash
    }

    /// Two of three is model jitter, not a broken model — that one is kept.
    @Test func twoOfThreeAnchorsStillPasses() async throws {
        let f = try Self.fixture()
        try Self.installVersion("m8", marker: "masks-properly", into: f.install)
        try Self.installVersion("m9", marker: "fine", into: f.install)
        try ModelInstaller.pointCurrent(at: "m9", in: f.install)
        f.rollback.arm(new: "m9", previous: "m8")

        let verdict = await f.rollback.verify { _ in
            PrivacyFilter.ProbeResult(
                found: Array(PrivacyFilter.probeAnchors.prefix(2)),
                missed: [PrivacyFilter.probeAnchors[2]], ran: true)
        }
        #expect(verdict == .verified, "\(verdict)")
        #expect(Self.currentVersion(in: f.install) == "m9")
        #expect(!f.rollback.blockedVersions.contains("m9"))
        #expect(!f.rollback.isPending)
        _ = f.trash
    }

    /// Nothing armed is not a verdict about anything.
    @Test func verifyWithNothingArmedIsIdle() async throws {
        let f = try Self.fixture()
        let verdict = await f.rollback.verify { _ in
            PrivacyFilter.ProbeResult(found: [], missed: PrivacyFilter.probeAnchors, ran: true)
        }
        #expect(verdict == .idle, "\(verdict)")
        _ = f.trash
    }

    /// The keys are the caller's, so a host can keep its own namespace — and an existing host can
    /// keep reading records it wrote under its old key.
    @Test func settingsKeysAreInjectable() throws {
        let f = try Self.fixture()
        let keys = ModelRollback.SettingsKeys(probation: "custom.probation", failedVersions: "custom.failed")
        let custom = ModelRollback(installRoot: f.install, defaults: f.defaults, keys: keys)
        try Self.installVersion("a", marker: "a", into: f.install)
        custom.arm(new: "b", previous: "a")
        custom.block("b")

        #expect(f.defaults.dictionary(forKey: "custom.probation") != nil)
        #expect(f.defaults.stringArray(forKey: "custom.failed") == ["b"])
        // The default-keyed view sees none of it.
        #expect(!f.rollback.isPending)
        #expect(f.rollback.blockedVersions.isEmpty)
        _ = f.trash
    }
}

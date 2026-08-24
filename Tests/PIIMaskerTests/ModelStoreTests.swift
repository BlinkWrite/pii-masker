import Foundation
import Testing

@testable import PIIMasker

/// `ModelStore` is process-wide by design, so these run serialized — a parallel test reconfiguring
/// the location underneath another would make both flaky for reasons that have nothing to do with
/// the code.
///
/// No weights needed: every case here is about which directory is chosen and what the loader's
/// verdict is, not about inference.
@Suite("Model store", .serialized)
struct ModelStoreTests {

    /// Restores whatever the store was configured with, so one test can't leak into the next.
    static func withLocation(_ location: ModelStore.Location, _ body: () throws -> Void) rethrows {
        let saved = ModelStore.location
        defer { ModelStore.configure(saved) }
        ModelStore.configure(location)
        try body()
    }

    static func completeDir(_ url: URL, marker: String = "weights") throws -> URL {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try marker.data(using: .utf8)!.write(to: url.appendingPathComponent("model.onnx"))
        for f in ModelInstaller.requiredModelFiles where f != "model.onnx" {
            try "{}".data(using: .utf8)!.write(to: url.appendingPathComponent(f))
        }
        return url
    }

    /// A host that never configured the store resolves nothing — so the fail-closed gate holds
    /// everything. Safe, and it looks exactly like a missing download, which is the correct story.
    @Test func unconfiguredResolvesNothing() {
        Self.withLocation(.unconfigured) {
            #expect(ModelStore.resolvedModelDirectory() == nil)
            #expect(!ModelStore.modelIsInstalled())
        }
    }

    /// The install root's `current` wins over every fallback, and it is resolved THROUGH the symlink
    /// so a caller holds a concrete version directory: a refresh flipping `current` afterwards
    /// cannot change what an in-flight load is reading.
    @Test func installRootWinsAndResolvesThroughTheSymlink() throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let install = root.appendingPathComponent("install")
        let bundled = root.appendingPathComponent("bundled")
        _ = try Self.completeDir(install.appendingPathComponent("v1"), marker: "installed")
        _ = try Self.completeDir(bundled, marker: "bundled")
        try ModelInstaller.pointCurrent(at: "v1", in: install)

        Self.withLocation(ModelStore.Location(installRoot: install, fallbacks: [bundled])) {
            let dir = ModelStore.resolvedModelDirectory()
            #expect(dir?.lastPathComponent == "v1", "\(dir?.path ?? "nil")")
            let weights = (try? String(
                contentsOf: dir!.appendingPathComponent("model.onnx"), encoding: .utf8)) ?? ""
            #expect(weights == "installed", "\(weights)")
        }
        _ = trash
    }

    /// Fallbacks are searched in order, and only a COMPLETE directory counts. Answering per file
    /// would let a directory missing one of them still report installed by resolving that file from
    /// a different candidate — and the loader, which reads them from a single folder, would then
    /// fail with nothing to retry it.
    @Test func fallbacksAreOrderedAndMustBeComplete() throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let incomplete = root.appendingPathComponent("incomplete")
        try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
        try "x".data(using: .utf8)!.write(to: incomplete.appendingPathComponent("model.onnx"))
        let good = try Self.completeDir(root.appendingPathComponent("good"), marker: "good")

        Self.withLocation(
            ModelStore.Location(installRoot: nil, fallbacks: [incomplete, good])
        ) {
            #expect(ModelStore.resolvedModelDirectory()?.lastPathComponent == "good")
            #expect(ModelStore.modelIsInstalled())
        }
        _ = trash
    }

    /// An install root whose `current` is incomplete falls through to the fallbacks rather than
    /// reporting a model that cannot load.
    @Test func anIncompleteCurrentFallsThrough() throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let install = root.appendingPathComponent("install")
        let broken = install.appendingPathComponent("v1")
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
        try "x".data(using: .utf8)!.write(to: broken.appendingPathComponent("model.onnx"))
        try ModelInstaller.pointCurrent(at: "v1", in: install)
        let bundled = try Self.completeDir(root.appendingPathComponent("bundled"), marker: "bundled")

        Self.withLocation(ModelStore.Location(installRoot: install, fallbacks: [bundled])) {
            #expect(ModelStore.resolvedModelDirectory()?.lastPathComponent == "bundled")
        }
        _ = trash
    }

    /// Reconfiguring invalidates: the bytes the loader last judged are not the bytes at the new
    /// location, so a cached "installed" answer from the old one must not carry over.
    @Test func reconfiguringInvalidates() throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let good = try Self.completeDir(root.appendingPathComponent("good"))

        Self.withLocation(ModelStore.Location(installRoot: nil, fallbacks: [good])) {
            #expect(ModelStore.modelIsInstalled())
            ModelStore.configure(.unconfigured)
            #expect(!ModelStore.modelIsInstalled(), "the cached positive must not survive")
        }
        _ = trash
    }

    /// A load failure has to reach the host the moment it happens. The load runs asynchronously off
    /// launch, so whether it beats the host's last badge refresh is a race — and it loses whenever
    /// the ONNX session builds and only the tokenizer fails. Without this push the user sees a
    /// healthy-looking app with no badge and no retry.
    @Test func stateChangesArePushedOnce() {
        let saved = ModelStore.onStateChange
        defer { ModelStore.onStateChange = saved; ModelStore.invalidate() }

        ModelStore.invalidate()
        let pushes = Flag()
        ModelStore.onStateChange = { pushes.value = true }

        ModelStore.markUnusable()
        #expect(pushes.value)

        pushes.value = false
        ModelStore.markUnusable()
        #expect(!pushes.value, "an unchanged state should push nothing")

        ModelStore.invalidate()
        #expect(pushes.value, "clearing it should push too")
    }

    /// `markUnusable` also drops the installed cache: the files are there, so a cached positive
    /// would keep reporting a healthy install for a model nothing can open.
    @Test func markUnusableClearsTheInstalledCache() throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        let good = try Self.completeDir(root.appendingPathComponent("good"))

        Self.withLocation(ModelStore.Location(installRoot: nil, fallbacks: [good])) {
            #expect(ModelStore.modelIsInstalled())
            ModelStore.markUnusable()
            #expect(ModelStore.state == .unusable)
            // The files are still there, so it re-checks disk and answers true again — but it did
            // re-check, which is the point.
            #expect(ModelStore.modelIsInstalled())
        }
        ModelStore.invalidate()
        _ = trash
    }

    // MARK: - The fail-closed gate

    /// With no model, `maskFields` returns nil rather than the values it was handed. This is the
    /// whole contract: the caller's only correct response to nil is to send nothing.
    @Test func maskFieldsHoldsWhenNoModelIsInstalled() async {
        let saved = ModelStore.location
        defer { ModelStore.configure(saved) }
        ModelStore.configure(.unconfigured)

        let masker = PrivacyFilter()
        let out = await masker.maskFields(["a@b.com", "call 415-555-0142"], timeout: 1)
        #expect(out == nil)
    }

    /// Nothing to mask is success, not a drop — a caller with no populated fields must not be told
    /// to abandon the request.
    @Test func noValuesIsSuccess() async {
        let masker = PrivacyFilter()
        let out = await masker.maskFields([], timeout: 1)
        #expect(out?.masked.isEmpty == true)
        #expect(out?.restore.isEmpty == true)
    }
}

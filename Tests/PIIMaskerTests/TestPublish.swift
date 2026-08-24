import CryptoKit
import Foundation

@testable import PIIMasker

/// Builds a fake published model — a real `.tar.gz` holding stand-in files — and the ``ModelPin``
/// that names it, with a `file://` URL. Everything the installer does to a real publish it does to
/// one of these: checksum the archive, unpack it, checksum the weights, swap the symlink.
enum TestPublish {

    static func sha256(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// - Parameters:
    ///   - marker: the body of the stand-in `model.onnx`, so a test can tell versions apart.
    ///   - forcedArchiveSHA: fakes a corrupted or tampered archive.
    ///   - forcedWeightsSHA: fakes an archive that checksums correctly but carries the wrong weights
    ///     — the case only the post-unpack check can catch.
    ///   - omitting: leave one required file out, for the cases that assert an incomplete archive is
    ///     refused.
    static func publish(
        version: String, marker: String, into dist: URL,
        forcedArchiveSHA: String? = nil, forcedWeightsSHA: String? = nil,
        omitting: String? = nil
    ) throws -> ModelPin {
        let fm = FileManager.default
        let src = dist.appendingPathComponent("src-\(version)")
        try? fm.removeItem(at: src)
        try fm.createDirectory(at: src, withIntermediateDirectories: true)
        // Mirrors what a real publish contains — `tokenizer_config.json` included, because the
        // loader opens it too and an archive without it produces an install that can never load.
        var files = ModelInstaller.requiredModelFiles
        if let omitting { files.removeAll { $0 == omitting } }
        try marker.data(using: .utf8)!.write(to: src.appendingPathComponent("model.onnx"))
        for f in files where f != "model.onnx" {
            try "{}".data(using: .utf8)!.write(to: src.appendingPathComponent(f))
        }

        let archive = dist.appendingPathComponent("gliner-\(version).tar.gz")
        try? fm.removeItem(at: archive)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        p.arguments = ["-czf", archive.path, "-C", src.path] + files
        try p.run()
        p.waitUntilExit()

        let archiveSHA = try forcedArchiveSHA ?? sha256(archive)
        let weightsSHA = try forcedWeightsSHA ?? sha256(src.appendingPathComponent("model.onnx"))
        let bytes = (try fm.attributesOfItem(atPath: archive.path)[.size] as? Int) ?? 0
        return ModelPin(
            version: version, sourceURL: archive, archiveSHA256: archiveSHA,
            weightsSHA256: weightsSHA, bytes: Int64(bytes), maxWidth: 12, maxSequenceLength: 768)
    }

    /// A fresh temp directory that cleans itself up when the test's `Trash` goes out of scope.
    static func scratch() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("pii-masker-test-\(UUID().uuidString)")
    }
}

/// Removes a directory when the test that owns it finishes.
final class Trash: @unchecked Sendable {
    let url: URL
    init(_ url: URL) { self.url = url }
    deinit { try? FileManager.default.removeItem(at: url) }
}

/// Lets a `@Sendable` progress closure record a signal without a mutable capture.
final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    var value: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); defer { lock.unlock() }; _value = newValue }
    }
}

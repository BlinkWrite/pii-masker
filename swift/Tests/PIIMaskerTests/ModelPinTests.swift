import Foundation
import Testing

@testable import PIIMasker

/// The pin is stated twice on purpose — once in Swift, once in `model.json` for a reader to diff
/// without building anything. Two statements of the same fact drift; this makes drift a red build.
@Suite("Model pin")
struct ModelPinTests {

    /// `swift/Tests/PIIMaskerTests/ModelPinTests.swift` → the repository root.
    ///
    /// Four levels, not three: `model.json` is shared with the .NET target, so it stays at the
    /// repository root while the Swift sources live under `swift/`.
    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // PIIMaskerTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // swift
            .deletingLastPathComponent()   // <repository root>
    }

    struct PinJSON: Decodable {
        let version: String
        let sourceURL: String
        let archiveSHA256: String
        let weightsSHA256: String
        let bytes: Int64
        let maxWidth: Int
        let maxSequenceLength: Int
    }

    @Test func modelJSONMatchesTheCurrentPin() throws {
        let data = try Data(contentsOf: Self.packageRoot.appendingPathComponent("model.json"))
        let json = try JSONDecoder().decode(PinJSON.self, from: data)
        let pin = ModelPin.current

        #expect(json.version == pin.version)
        #expect(json.sourceURL == pin.sourceURL.absoluteString)
        #expect(json.archiveSHA256 == pin.archiveSHA256)
        #expect(json.weightsSHA256 == pin.weightsSHA256)
        #expect(json.bytes == pin.bytes)
        #expect(json.maxWidth == pin.maxWidth)
        #expect(json.maxSequenceLength == pin.maxSequenceLength)
    }

    /// Every entry has to be installable: the version becomes a directory name, the hashes have to
    /// be hashes, and the URL has to be immutable — a `resolve/main` URL would let the bytes under a
    /// published pin change, which is the whole thing pinning exists to prevent.
    @Test func everyKnownPinIsWellFormed() {
        #expect(!ModelPin.known.isEmpty)
        for pin in ModelPin.known {
            #expect(InstallSupport.isSafePathComponent(pin.version), "\(pin.version)")
            #expect(pin.archiveSHA256.count == 64, "\(pin.version) archive hash")
            #expect(pin.weightsSHA256.count == 64, "\(pin.version) weights hash")
            let archiveIsHex = pin.archiveSHA256.allSatisfy { $0.isHexDigit }
            let weightsIsHex = pin.weightsSHA256.allSatisfy { $0.isHexDigit }
            #expect(archiveIsHex, "\(pin.version) archive hash")
            #expect(weightsIsHex, "\(pin.version) weights hash")
            #expect(pin.bytes > 0 && pin.bytes < ModelInstaller.maxPlausibleArchiveBytes, "\(pin.version) bytes")
            #expect(pin.maxWidth > 0, "\(pin.version) maxWidth")
            // A window this small could not fit the label preamble, let alone any text — so a
            // value that low is a typo, not a conservative choice.
            #expect(pin.maxSequenceLength > 64, "\(pin.version) maxSequenceLength")
            #expect(!pin.sourceURL.absoluteString.contains("/resolve/main/"), "\(pin.version) is not pinned to a commit")
        }
    }

    /// `known` is oldest → newest, and `current` is the newest. Rollback walks this ordering, so an
    /// entry appended in the wrong place would revert forwards.
    @Test func knownIsOrderedAndCurrentIsTheNewest() {
        #expect(ModelPin.current == ModelPin.known[ModelPin.known.count - 1])
        #expect(Set(ModelPin.known.map(\.version)).count == ModelPin.known.count, "duplicate versions")
        #expect(ModelPin.predecessor(of: ModelPin.known[0].version) == nil)
        #expect(ModelPin.predecessor(of: "never-published") == nil)
        for (i, pin) in ModelPin.known.enumerated() where i > 0 {
            #expect(ModelPin.predecessor(of: pin.version) == ModelPin.known[i - 1])
        }
    }
}

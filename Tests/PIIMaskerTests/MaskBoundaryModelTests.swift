import Foundation
import Testing

@testable import PIIMasker

@Suite("Masked boundaries with the installed model", .serialized, .enabled(if: ModelTier.isAvailable))
struct MaskBoundaryModelTests {
    @Test(arguments: ["sk-abc123def456ghi789", "AKIAIOSFODNN7EXAMPLE"])
    func repeatedCredentialAcrossContextAndLastMessage(_ key: String) async throws {
        let values = [
            "SENDER|MESSAGE\nMorgan|Can you create an API key for the test service?\nRiley|The API key is \(key)",
            "The API key is \(key)",
        ]
        let result = try #require(await ModelTier.masker().maskFields(values, timeout: 4))
        #expect(result.masked.count == values.count)
        for (field, original) in zip(result.masked, values) {
            #expect(!field.contains(key))
            #expect(PrivacyFilter.restore(field, with: result.restore) == original)
        }
    }
}

import Foundation
import Testing

@testable import PIIMasker

@Suite("Masked field boundaries")
struct MaskBoundaryTests {
    /// Synthetic version of a Slack context whose last credential is duplicated into lastMessage.
    @Test func credentialAcrossTwoFieldsKeepsBothFieldsMasked() throws {
        let values = ["SENDER|MESSAGE\nMorgan|synthetic-key-123", "synthetic-key-123"]
        let joined = PrivacyFilter.joinForMasking(values)
        let detected = "synthetic-key-123" + PrivacyFilter.maskSep + "synthetic-key-123"
        try check(values, spans: [span(joined, detected)])
    }

    @Test func credentialAndTrailingGuardRemainSeparate() throws {
        let values = ["Use synthetic-key-123"]
        let joined = PrivacyFilter.joinForMasking(values)
        try check(values, spans: [span(joined, "synthetic-key-123" + PrivacyFilter.maskSep + PrivacyFilter.maskGuard)])
    }

    @Test func credentialAndLeadingGuardRemainSeparate() throws {
        let values = ["synthetic-key-123 is the test credential"]
        let joined = PrivacyFilter.joinForMasking(values)
        try check(values, spans: [span(joined, PrivacyFilter.maskGuard + PrivacyFilter.maskSep + "synthetic-key-123")])
    }

    @Test func overlappingDetectionsCannotMergeAcrossMarkers() throws {
        let values = ["synthetic-key-123", "synthetic-key-456", "synthetic-key-789"]
        let joined = PrivacyFilter.joinForMasking(values)
        try check(values, spans: [
            span(joined, values[0] + PrivacyFilter.maskSep + values[1]),
            span(joined, values[1] + PrivacyFilter.maskSep + values[2]),
        ])
    }

    @Test func unicodeWhitespaceAndPunctuationSurvive() throws {
        let values = ["🙂 Cafe\u{301}: (synthetic-key-123)? \n", "\t‘synthetic-key-456’! "]
        let joined = PrivacyFilter.joinForMasking(values)
        try check(values, spans: [span(joined, "synthetic-key-123)? \n" + PrivacyFilter.maskSep + "\t‘synthetic-key-456")])
    }

    @Test func wholeBatchDetectionPreservesEmptyFieldsAndGuards() throws {
        let values = ["synthetic-key-123", "", "  ", "synthetic-key-456"]
        let joined = PrivacyFilter.joinForMasking(values)
        try check(values, spans: [(0, joined.count, "api key")])
    }

    @Test func markerSubstringInsideASecretIsNotExempt() {
        let text = "prefixMARKERsuffix"
        let spans = [(start: 0, end: text.count, label: "api key")]
        let protected = PrivacyFilter.dropProtectedSpans(text, spans, protecting: ["MARKER"])
        let result = PrivacyFilter.mask(text, spans: protected)
        #expect(result.text == "[API_KEY_1]")
        #expect(PrivacyFilter.restore(result.text, with: result.restore) == text)
    }

    private func span(_ text: String, _ value: String) -> (start: Int, end: Int, label: String) {
        let range = text.range(of: value)!
        return (text.distance(from: text.startIndex, to: range.lowerBound),
                text.distance(from: text.startIndex, to: range.upperBound), "api key")
    }

    private func check(_ values: [String], spans: [(start: Int, end: Int, label: String)]) throws {
        let joined = PrivacyFilter.joinForMasking(values)
        let protected = PrivacyFilter.dropProtectedSpans(joined, spans, protecting: PrivacyFilter.maskMarkerWords)
        let result = PrivacyFilter.mask(joined, spans: protected)
        let fields = try #require(PrivacyFilter.splitMaskedFields(result.text, count: values.count))
        #expect(fields.count == values.count)
        for (field, original) in zip(fields, values) {
            #expect(PrivacyFilter.restore(field, with: result.restore) == original)
            #expect(!field.contains("synthetic-key-"))
        }
        #expect(!result.restore.isEmpty)
        #expect(!result.restore.values.contains { $0.contains(PrivacyFilter.maskGuard) })
    }
}

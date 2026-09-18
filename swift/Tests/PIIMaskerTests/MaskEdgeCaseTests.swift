import Foundation
import Testing

@testable import PIIMasker

/// The branches the original suite never reached: span merging, grapheme offsets, restore
/// collisions, and the safety-net strip's own false positive.
@Suite("Masking edge cases")
struct MaskEdgeCaseTests {

    static func span(_ text: String, _ word: String, _ label: String)
        -> (start: Int, end: Int, label: String)
    {
        MaskRoundTripTests.span(text, word, label)
    }

    // MARK: - Span merging

    /// Overlapping spans merge into one placeholder rather than splicing over each other. Two
    /// labels claiming overlapping text is normal — GLiNER emits a span per label per width — and
    /// masking both would corrupt the offsets of everything after them.
    @Test func overlappingSpansMerge() {
        let text = "reach me at 415-555-0142 ok"
        let (masked, restore) = PrivacyFilter.mask(
            text,
            spans: [
                Self.span(text, "415-555-0142", "phone number"),
                Self.span(text, "555-0142", "phone number"),
            ])
        #expect(restore.count == 1, "\(restore)")
        #expect(masked == "reach me at [PHONE_NUMBER_1] ok", "\(masked)")
        #expect(PrivacyFilter.restore(masked, with: restore) == text)
    }

    /// Adjacent-but-not-overlapping spans stay separate: the merge is `span.start <= last.end`, so
    /// a span starting exactly where the previous ended is absorbed, while a gap keeps them apart.
    @Test func adjacentSpansAbsorbAndGappedSpansDoNot() {
        let text = "aaabbb ccc"
        let merged = PrivacyFilter.mask(
            text, spans: [(0, 3, "tax id"), (3, 6, "tax id")]).restore
        #expect(merged.count == 1, "touching spans should merge: \(merged)")

        let separate = PrivacyFilter.mask(
            text, spans: [(0, 3, "tax id"), (7, 10, "tax id")]).restore
        #expect(separate.count == 2, "gapped spans should stay apart: \(separate)")
    }

    /// Spans arrive in whatever order the model produced them; numbering is reading order.
    @Test func numberingFollowsReadingOrderNotInputOrder() {
        let text = "first 111 then 222"
        let (masked, restore) = PrivacyFilter.mask(
            text,
            spans: [
                Self.span(text, "222", "pin code"),
                Self.span(text, "111", "pin code"),
            ])
        #expect(masked == "first [PIN_CODE_1] then [PIN_CODE_2]", "\(masked)")
        #expect(restore["[PIN_CODE_1]"] == "111", "\(restore)")
        #expect(restore["[PIN_CODE_2]"] == "222", "\(restore)")
    }

    /// A degenerate or out-of-range span is skipped rather than trapping. `detect` should never
    /// produce one, but `mask` is public and its offsets are plain Ints.
    @Test func degenerateSpansAreSkipped() {
        let text = "short"
        let (masked, restore) = PrivacyFilter.mask(
            text, spans: [(3, 3, "tax id"), (99, 120, "tax id")])
        #expect(masked == text, "\(masked)")
        #expect(restore.isEmpty, "\(restore)")
    }

    // MARK: - Grapheme offsets

    /// Offsets are Character counts, so a span past an emoji, a combining mark or an RTL run has to
    /// land on the same characters `detect` measured. Byte or UTF-16 offsets would slip here.
    @Test func offsetsAreGraphemeClustersNotUnits() {
        let cases: [(String, String)] = [
            ("hi 👋 mail me at a@b.com ok", "a@b.com"),
            ("cafe\u{301} then a@b.com ok", "a@b.com"),              // combining acute
            ("family 👨‍👩‍👧‍👦 then a@b.com ok", "a@b.com"),             // ZWJ sequence
            ("שלום עולם a@b.com ok", "a@b.com"),                     // RTL
            ("flag 🇷🇴 a@b.com ok", "a@b.com"),                       // regional indicators
        ]
        for (text, target) in cases {
            let (masked, restore) = PrivacyFilter.mask(
                text, spans: [Self.span(text, target, "email address")])
            #expect(masked.contains("[EMAIL_ADDRESS_1]"), "\(text) → \(masked)")
            #expect(!masked.contains(target), "\(text) → \(masked)")
            #expect(PrivacyFilter.restore(masked, with: restore) == text, "\(text) → \(masked)")
        }
    }

    /// …and trimming measures the same way.
    @Test func trimMeasuresGraphemeClusters() {
        let text = "wave 👋 (a@b.com)."
        let raw = Self.span(text, "(a@b.com).", "email address")
        let t = PrivacyFilter.trimSpan(text, start: raw.start, end: raw.end)!
        let lo = text.index(text.startIndex, offsetBy: t.start)
        let hi = text.index(text.startIndex, offsetBy: t.end)
        #expect(String(text[lo..<hi]) == "a@b.com", "\(String(text[lo..<hi]))")
    }

    // MARK: - Restore collisions

    /// Two spans covering identical text still get distinct tokens, and both restore. `restore`
    /// iterates a dictionary, so the order is undefined — the result must not depend on it.
    @Test func identicalOriginalsGetDistinctTokens() {
        let text = "call 555 or 555 again"
        let (masked, restore) = PrivacyFilter.mask(
            text, spans: [(5, 8, "phone number"), (12, 15, "phone number")])
        #expect(restore.count == 2, "\(restore)")
        #expect(masked == "call [PHONE_NUMBER_1] or [PHONE_NUMBER_2] again", "\(masked)")
        #expect(PrivacyFilter.restore(masked, with: restore) == text)
    }

    /// Past ten spans of one label, `restore` faces `[PHONE_NUMBER_1]` and `[PHONE_NUMBER_11]` in
    /// the same map — and it substitutes by iterating a dictionary, so it may reach either first.
    /// The closing bracket is what keeps that safe: `[PHONE_NUMBER_1]` is not a substring of
    /// `[PHONE_NUMBER_11]`, so neither can eat the other's prefix. Asserted because the token shape
    /// is what makes it true, and a change to that shape would break this silently.
    @Test func tokenNumberingHasNoPrefixCollision() {
        let words = (1...12).map { "n\($0)" }
        let text = words.joined(separator: " ")
        var spans: [(start: Int, end: Int, label: String)] = []
        var cursor = 0
        for w in words {
            spans.append((cursor, cursor + w.count, "phone number"))
            cursor += w.count + 1
        }
        let (masked, restore) = PrivacyFilter.mask(text, spans: spans)
        #expect(restore.count == 12, "\(restore)")
        #expect(masked.contains("[PHONE_NUMBER_1]"), "\(masked)")
        #expect(masked.contains("[PHONE_NUMBER_11]"), "\(masked)")
        #expect(restore["[PHONE_NUMBER_1]"] == "n1", "\(restore)")
        #expect(restore["[PHONE_NUMBER_11]"] == "n11", "\(restore)")
        #expect(PrivacyFilter.restore(masked, with: restore) == text)
    }

    /// An original that is ITSELF shaped like a placeholder does NOT survive the round trip: the
    /// substitution puts it back, and the safety-net strip — which runs afterwards over the whole
    /// string — cannot tell it from a token the model invented, so it deletes it.
    ///
    /// Documented, not fixed. The alternative order (strip first, or substitute and strip in one
    /// scan) would fix it, and this test is here so that change is a deliberate one rather than an
    /// accident. The trade the current code makes is: never let a redaction marker reach the user,
    /// at the cost of mangling user text that looks exactly like one.
    @Test func anOriginalShapedLikeAPlaceholderIsEatenByTheSafetyNet() {
        let text = "the key is [API_KEY_1] literally"
        let (masked, restore) = PrivacyFilter.mask(
            text, spans: [Self.span(text, "[API_KEY_1]", "api key")])
        let restored = PrivacyFilter.restore(masked, with: restore)
        #expect(restored == "the key is  literally", "\(restored)")
        #expect(restored != text, "if this now round-trips, the strip was reordered on purpose")
    }

    /// The known cost of that safety net: legitimate user text of the placeholder shape is stripped
    /// when it was never masked. Documented here rather than fixed — a leaked redaction marker is
    /// worse than a mangled `[FOO_1]` — so a change in this behaviour is a deliberate one.
    @Test func theSafetyNetAlsoEatsLookalikeUserText() {
        let stripped = PrivacyFilter.restore(
            "see [SECTION_2] of the doc", with: ["[IP_ADDRESS_1]": "10.0.0.1"])
        #expect(stripped == "see  of the doc", "\(stripped)")

        // …but only with a non-empty map. An empty map is a strict no-op, so a caller that masked
        // nothing never has its text touched.
        let untouched = PrivacyFilter.restore("see [SECTION_2] of the doc", with: [:])
        #expect(untouched == "see [SECTION_2] of the doc", "\(untouched)")

        // And the shape is narrow: lowercase, or no trailing number, is left alone.
        let narrow = PrivacyFilter.restore(
            "see [section_2] and [NOTES] here", with: ["[IP_ADDRESS_1]": "10.0.0.1"])
        #expect(narrow == "see [section_2] and [NOTES] here", "\(narrow)")
    }

    // MARK: - Round-trip property

    /// Mask-then-restore is the identity for arbitrary non-overlapping span sets over arbitrary
    /// text — the property every individual case above is an instance of.
    ///
    /// The one documented exception is excluded from the corpus: text that ALREADY contains a
    /// placeholder-shaped substring does not round-trip, because the safety-net strip removes it.
    /// See `anOriginalShapedLikeAPlaceholderIsEatenByTheSafetyNet`.
    @Test func maskThenRestoreIsIdentity() {
        let corpus = [
            "plain text with no entities at all",
            "a@b.com 415-555-0142 192.168.4.21 all together",
            "punctuation, (parens), \"quotes\" and — dashes",
            "emoji 👋 mixed 🇷🇴 with שלום rtl",
            "   leading and trailing whitespace   ",
        ]
        let labels = ["email address", "phone number", "ip address", "api key"]
        var rng = SystemRandomNumberGenerator()

        for text in corpus {
            let n = text.count
            guard n > 2 else { continue }
            for _ in 0..<40 {
                // Non-overlapping spans, left to right.
                var spans: [(start: Int, end: Int, label: String)] = []
                var cursor = 0
                while cursor < n {
                    let gap = Int.random(in: 0...3, using: &rng)
                    let start = cursor + gap
                    guard start < n else { break }
                    let end = min(n, start + Int.random(in: 1...4, using: &rng))
                    spans.append((start, end, labels.randomElement(using: &rng)!))
                    cursor = end
                }
                let (masked, restore) = PrivacyFilter.mask(text, spans: spans)
                #expect(
                    PrivacyFilter.restore(masked, with: restore) == text,
                    "spans \(spans.map { "\($0.start)..<\($0.end)" }) → \(masked)")
            }
        }
    }

    // MARK: - The probe verdict

    /// Two of three is the pass mark: one miss is model jitter, two is a model that is not doing its
    /// job. And a probe that never ran is never a pass, however many anchors it claims.
    @Test func probeVerdictIsTwoOfThree() {
        func result(_ found: Int, ran: Bool) -> PrivacyFilter.ProbeResult {
            PrivacyFilter.ProbeResult(
                found: Array(PrivacyFilter.probeAnchors.prefix(found)),
                missed: Array(PrivacyFilter.probeAnchors.dropFirst(found)), ran: ran)
        }
        #expect(!result(0, ran: true).passes)
        #expect(!result(1, ran: true).passes)
        #expect(result(2, ran: true).passes)
        #expect(result(3, ran: true).passes)
        #expect(!result(3, ran: false).passes)
    }

    /// The probe text carries all three anchors, or it cannot measure what it claims to.
    @Test func probeTextContainsEveryAnchor() {
        for anchor in PrivacyFilter.probeAnchors {
            #expect(PrivacyFilter.probeText.contains(anchor), "\(anchor)")
        }
    }

    // MARK: - The timeout race

    @Test func withTimeoutReturnsAFastValue() async {
        let outcome = await PrivacyFilter.withTimeout(seconds: 5) { "ok" }
        guard case .value(let v) = outcome else {
            Issue.record("a fast operation should yield its value")
            return
        }
        #expect(v == "ok")
    }

    @Test func withTimeoutTripsOnASlowOperation() async {
        let outcome = await PrivacyFilter.withTimeout(seconds: 0.05) { () async -> String in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            return "too late"
        }
        guard case .timedOut = outcome else {
            Issue.record("a slow operation should time out")
            return
        }
    }

    /// The invariant, stated as a pure function: only text the masker produced is ever sent.
    @Test func onlyMaskerProducedTextIsSent() {
        let safe = PrivacyFilter.Sanitized(text: "masked", restore: [:])
        #expect(PrivacyFilter.resolveMask(.value(safe)) == .send(safe))
        #expect(PrivacyFilter.resolveMask(.value(nil)) == .dropFilterFailed)
        #expect(PrivacyFilter.resolveMask(.timedOut) == .dropTimedOut)
    }
}

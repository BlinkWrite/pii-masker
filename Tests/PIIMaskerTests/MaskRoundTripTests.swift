import Foundation
import Testing

@testable import PIIMasker

/// Hermetic tests for the reversible masking (`PrivacyFilter.mask` / `.restore`) — the pure
/// round-trip behind "a false positive must never leak a redaction marker into user-visible text".
///
/// The bug these come from: GLiNER tagged the ordinary word "ip" in a draft as an `ip address`, the
/// masker replaced it with `*****`, and the model copied the stars straight into the user's text.
/// Masking is now reversible: the span becomes a labelled token the model reasons around, and
/// `restore` swaps the original back into the response — so even a wrong detection round-trips
/// cleanly. None of this needs the model, so the contract is locked regardless of which weights are
/// installed.
@Suite("Mask round-trip")
struct MaskRoundTripTests {

    /// Build an Int char span (start, end, label) for the first occurrence of `word` in `text`,
    /// mirroring how `detect` hands word ranges to `mask`.
    static func span(_ text: String, _ word: String, _ label: String)
        -> (start: Int, end: Int, label: String)
    {
        let r = text.range(of: word)!
        let start = text.distance(from: text.startIndex, to: r.lowerBound)
        let end = text.distance(from: text.startIndex, to: r.upperBound)
        return (start, end, label)
    }

    /// A masked span becomes a labelled, reversible token — never `*****`.
    @Test func replacesWithTokenNotStars() {
        let text = "my ip is here"
        let (masked, restore) = PrivacyFilter.mask(
            text, spans: [Self.span(text, "ip", "ip address")])
        #expect(!masked.contains("*"), "\(masked)")
        #expect(masked.contains("[IP_ADDRESS_1]"), "\(masked)")
        #expect(restore["[IP_ADDRESS_1]"] == "ip", "\(restore)")
    }

    /// The reported bug: "ip" in a draft, wrongly flagged, must survive a full
    /// mask → (model preserves token) → restore round-trip with no marker left.
    @Test func falsePositiveRoundTrips() {
        let draft = "whats this for specifically? hoewe foes it help ip"
        let (masked, restore) = PrivacyFilter.mask(
            draft, spans: [Self.span(draft, "ip", "ip address")])
        // The model sees only the token and returns a corrected paragraph that preserves it verbatim.
        let modelOutput = "what's this for specifically? how does it help [IP_ADDRESS_1]"
        let restored = PrivacyFilter.restore(modelOutput, with: restore)
        #expect(restored.hasSuffix("it help ip"), "\(restored)")
        #expect(!restored.contains("*"), "\(restored)")
        #expect(!restored.contains("["), "\(restored)")
        #expect(!masked.contains(" ip"), "\(masked)")
    }

    /// Multiple detected spans get distinct numbered tokens and all restore.
    @Test func multipleEntitiesNumberedAndRestored() {
        let text = "call 555 or 777 now"
        let (masked, restore) = PrivacyFilter.mask(
            text,
            spans: [
                Self.span(text, "555", "phone number"),
                Self.span(text, "777", "phone number"),
            ])
        #expect(masked.contains("[PHONE_NUMBER_1]"), "\(masked)")
        #expect(masked.contains("[PHONE_NUMBER_2]"), "\(masked)")
        #expect(restore.count == 2, "\(restore)")
        let restored = PrivacyFilter.restore(masked, with: restore)
        #expect(restored == text, "\(restored)")
    }

    /// Helper: the substring a trimmed span covers, or nil if dropped.
    static func trimmedText(_ text: String, _ word: String) -> String? {
        let raw = span(text, word, "ip address")
        guard let t = PrivacyFilter.trimSpan(text, start: raw.start, end: raw.end)
        else { return nil }
        let lo = text.index(text.startIndex, offsetBy: t.start)
        let hi = text.index(text.startIndex, offsetBy: t.end)
        return String(text[lo..<hi])
    }

    /// The reported "ip??" bug: GLiNER's whitespace word is "ip?" (the "?" rides along). Trimming
    /// the span boundary drops the "?" so the placeholder covers only "ip" and the "?" stays as
    /// visible text — the model can't double it.
    @Test func trimsBoundaryPunctuation() {
        #expect(Self.trimmedText("how does it help ip?", "ip?") == "ip")

        // Internal punctuation (a real IP's dots) survives — only edges trim.
        #expect(Self.trimmedText("ping 192.168.0.1?", "192.168.0.1?") == "192.168.0.1")

        // Leading/trailing quotes and brackets trim from both ends.
        #expect(Self.trimmedText("call (5551234).", "(5551234).") == "5551234")

        // A span that is only punctuation trims to nothing → dropped.
        #expect(Self.trimmedText("wait ... go", "...") == nil)

        // Full path: mask the trimmed "ip" span, model echoes the token verbatim (the "?" is already
        // visible to it), restore → a single "?", not "ip??".
        let draft = "how does it help ip?"
        let raw = Self.span(draft, "ip?", "ip address")
        let t = PrivacyFilter.trimSpan(draft, start: raw.start, end: raw.end)!
        let (masked, restore) = PrivacyFilter.mask(
            draft, spans: [(start: t.start, end: t.end, label: "ip address")])
        #expect(masked == "how does it help [IP_ADDRESS_1]?", "\(masked)")
        let restored = PrivacyFilter.restore(masked, with: restore)
        #expect(restored == "how does it help ip?", "\(restored)")
    }

    /// No detected spans → text unchanged, empty map.
    @Test func noSpansIsIdentity() {
        let text = "nothing private here"
        let (masked, restore) = PrivacyFilter.mask(text, spans: [])
        #expect(masked == text, "\(masked)")
        #expect(restore.isEmpty, "\(restore)")
    }

    /// Safety net: if the model garbles/drops a token so it can't be matched, the leftover marker is
    /// stripped rather than shown to the user.
    @Test func garbledTokenStripped() {
        // Map has a token the output doesn't contain; output has a stray one.
        let restored = PrivacyFilter.restore(
            "the value is [IP_ADDRESS_2] done", with: ["[IP_ADDRESS_1]": "ip"])
        #expect(!restored.contains("["), "\(restored)")
        #expect(restored.contains("the value is"), "\(restored)")
    }

    /// An empty restore map is a no-op (no spurious stripping of real text).
    @Test func emptyRestoreMapIsIdentity() {
        let text = "an array index a[0] stays"
        #expect(PrivacyFilter.restore(text, with: [:]) == text)
    }

    // MARK: - The join

    /// The single-pass join/split that lets several fields share one restore map. Valid round-trip
    /// recovers every field in order.
    @Test func sentinelRoundTrip() {
        let values = ["context here", "the draft ", "reply target"]
        let joined = PrivacyFilter.joinForMasking(values)
        let back = PrivacyFilter.splitMaskedFields(joined, count: values.count)
        #expect(back == values, "\(String(describing: back))")
    }

    /// A field that itself contains the separator over-splits → count mismatch → drop (nil), never a
    /// silent mis-assignment of masked text onto the wrong field.
    @Test func sentinelSeparatorCollisionDrops() {
        let evil = "a" + PrivacyFilter.maskSep + "b"
        let joined = PrivacyFilter.joinForMasking(["safe", evil])
        #expect(PrivacyFilter.splitMaskedFields(joined, count: 2) == nil)
    }

    /// The count can match while the outer guards don't — only the guard-integrity check catches it.
    @Test func sentinelGuardTamperDrops() {
        let tampered = "NOTGUARD" + PrivacyFilter.maskSep + "field1" + PrivacyFilter.maskSep + "NOTGUARD"
        #expect(PrivacyFilter.splitMaskedFields(tampered, count: 1) == nil)
    }

    /// Too few parts (no sentinels at all) → drop.
    @Test func sentinelCountMismatchDrops() {
        #expect(PrivacyFilter.splitMaskedFields("just text", count: 2) == nil)
    }

    /// The reported drop: with an email-heavy context, GLiNER false-flagged the join's own guard
    /// word as an `email address`, masked it, and the guard-integrity split then failed — dropping
    /// the whole request before it reached the model. `dropProtectedSpans` shields the markers so a
    /// false positive on one is ignored while real PII still masks and every field round-trips.
    @Test func protectedMarkersSurviveFalsePositive() {
        let values = ["a real entity 555 here", "the draft"]
        let joined = PrivacyFilter.joinForMasking(values)

        // GLiNER false-flags BOTH guard words (one at each end) plus a real number.
        var spans: [(start: Int, end: Int, label: String)] = []
        var from = joined.startIndex
        while let r = joined.range(
            of: PrivacyFilter.maskGuard, range: from..<joined.endIndex)
        {
            spans.append((
                joined.distance(from: joined.startIndex, to: r.lowerBound),
                joined.distance(from: joined.startIndex, to: r.upperBound),
                "email address"))
            from = r.upperBound
        }
        spans.append(Self.span(joined, "555", "phone number"))

        // Without shielding, masking the guards rewrites them → split fails → drop.
        let unshielded = PrivacyFilter.mask(joined, spans: spans).text
        #expect(
            PrivacyFilter.splitMaskedFields(unshielded, count: values.count) == nil, "\(unshielded)")

        // With shielding, guard false-positives are dropped; the real PII still masks and every
        // field round-trips.
        let kept = PrivacyFilter.dropProtectedSpans(
            joined, spans, protecting: PrivacyFilter.maskMarkerWords)
        let (masked, restore) = PrivacyFilter.mask(joined, spans: kept)
        let fields = PrivacyFilter.splitMaskedFields(masked, count: values.count)
        #expect(fields != nil, "\(masked)")
        #expect(!restore.values.contains(PrivacyFilter.maskGuard), "\(restore)")
        if let f = fields {
            #expect(f[0].contains("[PHONE_NUMBER_1]"), "\(f[0])")
            #expect(f[1] == "the draft", "\(f[1])")
        }
    }
}

import Foundation

// The fail-closed gate: everything between "the caller has some strings" and "these are safe to
// send, or send nothing". The join markers, the timeout race, the round-trip integrity check and
// the not-installed early-out all live here rather than in the caller, because each of them is a
// way to leak or corrupt text and none of them is the caller's business to get right.

extension PrivacyFilter {

    /// The outcome of racing an async operation against a timeout.
    public enum TimeoutOutcome<T: Sendable>: Sendable {
        case value(T)
        case timedOut
    }

    /// What to do with text after racing the masker against its timeout. The invariant: only text
    /// the masker produced is ever `.send`; any failure or timeout maps to a `.drop`, never to
    /// sending the raw text.
    public enum MaskOutcome: Sendable, Equatable {
        case send(Sanitized)
        case dropFilterFailed
        case dropTimedOut
    }

    /// Map a sanitize-with-timeout result to the send/drop decision. Pure, so the "never send
    /// unmasked" invariant is testable without a model: `.value(.some)` → the masker produced text
    /// → send it; `.value(.none)` → the masker ran but failed → drop; `.timedOut` → drop.
    public static func resolveMask(
        _ outcome: TimeoutOutcome<Sanitized?>
    ) -> MaskOutcome {
        switch outcome {
        case .value(.some(let masked)): return .send(masked)
        case .value(.none): return .dropFilterFailed
        case .timedOut: return .dropTimedOut
        }
    }

    /// Run `operation`, returning `.timedOut` if it does not finish within `seconds`. The losing
    /// child task is cancelled.
    public static func withTimeout<T: Sendable>(
        seconds: TimeInterval, _ operation: @escaping @Sendable () async -> T
    ) async -> TimeoutOutcome<T> {
        await withTaskGroup(of: TimeoutOutcome<T>.self) { group in
            group.addTask { .value(await operation()) }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
                return .timedOut
            }
            let first = await group.next() ?? .timedOut
            group.cancelAll()
            return first
        }
    }

    // MARK: - The join

    /// Brackets the joined list. The masker trims outer whitespace, and without a sentinel that
    /// trim eats the leading or trailing edge of a real field — a draft's trailing space, say,
    /// which a continuation prompt may depend on.
    public static let maskGuard = "\u{1E}\u{1E}"

    /// Separates fields inside the join. Space-padded so it is a whitespace-delimited word to the
    /// masker, which splits on whitespace.
    public static let maskSep = " \u{1E}\u{1E}\u{1E} "

    /// The structural marker "words" ``joinForMasking(_:)`` introduces into the text the masker
    /// sees. The masker must never PII-mask these — a false-positive hit on one breaks the split
    /// round-trip and drops the whole request — so they are handed to
    /// ``sanitize(_:protecting:userNameMasking:)`` to shield them.
    public static let maskMarkerWords: Set<String> = [
        maskGuard, maskSep.trimmingCharacters(in: .whitespaces),
    ]

    /// Join field values for a single masking pass.
    public static func joinForMasking(_ values: [String]) -> String {
        ([maskGuard] + values + [maskGuard]).joined(separator: maskSep)
    }

    /// Inverse of ``joinForMasking(_:)``: recover the `count` field values from the masked text.
    ///
    /// Returns nil unless the split yields exactly `count` fields between two INTACT guards — a
    /// field that contained the separator, or any guard tampering, fails the check so the caller
    /// drops. The guard-equality check (not just the count) is the integrity boundary: it is what
    /// prevents mis-aligned masked text being reassigned onto fields.
    public static func splitMaskedFields(_ maskedText: String, count: Int) -> [String]? {
        let parts = maskedText.components(separatedBy: maskSep)
        guard parts.count == count + 2,
            parts.first == maskGuard,
            parts.last == maskGuard
        else { return nil }
        return Array(parts[1...count])
    }

    // MARK: - The gate

    /// How long a masking pass may run before the caller drops the request.
    ///
    /// The masker is a local NER pass on a serialized actor, so a pathological input — a huge log
    /// or terminal dump — can make one call run for tens of seconds and block every request behind
    /// it. On timeout the request is abandoned entirely; there is no fallback to sending the
    /// unmasked text.
    public static let defaultMaskTimeout: TimeInterval = 4.0

    /// Mask several field values in ONE inference pass so they share a single token map — separate
    /// passes restart the counter, so a later field's `[PERSON_1]` would collide with an earlier
    /// one on restore.
    ///
    /// Returns nil if the model is missing, inference fails, the pass exceeds `timeout`, or the
    /// masked text does not round-trip. Never partial, never raw. **The caller's only correct
    /// response to nil is to send nothing.**
    ///
    /// `values` empty → `([], [:])`, not nil: nothing to mask is success.
    ///
    /// This does not mask names. A caller that knows which field holds whose text should apply
    /// ``UserNameMask`` per field first, with the scope that field deserves; this call cannot tell
    /// the fields apart once they are joined.
    public func maskFields(
        _ values: [String], timeout: TimeInterval
    ) async -> (masked: [String], restore: [String: String])? {
        guard !values.isEmpty else { return ([], [:]) }

        // The download hasn't landed yet (fresh install, failed download): hold rather than risk
        // sending unmasked text. `sanitize` would fail-and-drop anyway; this is the explicit,
        // quiet early-out, and it is part of the gate rather than a caller optimisation.
        guard hasResolvableModel else { return nil }

        let joined = Self.joinForMasking(values)
        let outcome = await Self.withTimeout(seconds: timeout) {
            await self.sanitize(
                joined, protecting: Self.maskMarkerWords, userNameMasking: false)
        }
        switch Self.resolveMask(outcome) {
        case .send(let sanitized):
            guard let fields = Self.splitMaskedFields(sanitized.text, count: values.count) else {
                return nil
            }
            return (fields, sanitized.restore)
        case .dropFilterFailed, .dropTimedOut:
            return nil
        }
    }
}

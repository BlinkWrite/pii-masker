import Foundation

/// The inference knobs. Immutable for the lifetime of a masker.
public struct MaskerConfig: Sendable, Equatable {
    /// The entity types to look for, fed to GLiNER at inference time.
    ///
    /// **Order is the output index space.** The model returns an index into this exact array, so
    /// the array used for inference must be the one used to read the result back.
    ///
    /// **A label is not just a filter — it names the placeholder.** The label is uppercased with
    /// every non-alphanumeric mapped to `_`, so `"email address"` becomes `[EMAIL_ADDRESS_1]`.
    /// Renaming a label changes the token text the model sees, and invalidates any restore map
    /// built before the change.
    public let labels: [String]

    /// Sigmoid probability floor for keeping a span.
    ///
    /// A recall/precision dial whose two errors are not equally bad. A miss is a leak: PII goes
    /// out, permanently, into whatever the text is sent to. A false positive only costs a noisier
    /// prompt, since masking here is reversible and ``restore`` puts the text back. So the floor
    /// belongs below the middle, and the 0.1 default sits there.
    ///
    /// Recall falls as the surrounding text grows — a bare sentence is saturated at any usable
    /// threshold, long prose is not. A host masking long context, and willing to trade prompt noise
    /// for recall, should go lower; the dial is here to be turned.
    public let threshold: Float

    /// The most text one masking call may carry, counted in tokens (the model's own units, not
    /// characters — 1,800 characters is 379 tokens of English prose but 1,752 of CJK, so a
    /// character budget means something different in every language).
    ///
    /// This is a COST limit, distinct from ``ModelPin/maxSequenceLength``, which is a correctness
    /// limit the weights impose. Input longer than one window is split into several and masked in
    /// full, so nothing is lost — but each window is another inference pass on a serialized actor,
    /// and unbounded input means unbounded passes with every queued request waiting behind them.
    ///
    /// It is a caller's choice rather than a fact about the weights, which is why it lives here and
    /// not on the pin. Raise it if you would rather wait than lose long inputs; lower it to keep
    /// worst-case latency tight. At roughly 1 ms per word, the 2,000 default is about three windows
    /// and ~1.8s worst case, inside ``PrivacyFilter/defaultMaskTimeout``.
    ///
    /// Over the budget, ``PrivacyFilter/maskFields(_:timeout:)`` returns nil — the same fail-closed
    /// answer as every other failure, so the caller sends nothing.
    public let maxInputTokens: Int

    public init(
        labels: [String] = .defaultPII, threshold: Float = 0.1, maxInputTokens: Int = 2_000
    ) {
        self.labels = labels
        self.threshold = threshold
        self.maxInputTokens = maxInputTokens
    }

    public static let `default` = MaskerConfig()
}

extension Array where Element == String {
    /// The 17 labels this library ships with. GLiNER takes labels at inference time, so any label
    /// set works with any model — unlike ``ModelPin/maxWidth``, which must match the weights.
    public static let defaultPII: [String] = [
        "phone number",
        "email address",
        "credit card number",
        "address",
        "social security number",
        "date of birth",
        "bank account number",
        "password",
        "pin code",
        "ip address",
        "dollar amount",
        "passport number",
        "driver license number",
        "tax id",
        "api key",
        "access token",
        "secret key",
    ]
}

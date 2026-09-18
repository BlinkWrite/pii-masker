using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace PIIMasker;

/// <summary>The inference knobs. Immutable for the lifetime of a masker.</summary>
public sealed record MaskerConfig
{
    /// <summary>The entity types to look for, fed to GLiNER at inference time.</summary>
    /// <remarks>
    /// <para>
    /// <b>Order is the output index space.</b> The model returns an index into this exact array, so
    /// the array used for inference must be the one used to read the result back.
    /// </para>
    /// <para>
    /// <b>A label is not just a filter — it names the placeholder.</b> The label is uppercased with
    /// every non-alphanumeric mapped to <c>_</c>, so <c>"email address"</c> becomes
    /// <c>[EMAIL_ADDRESS_1]</c>. Renaming a label changes the token text the model sees, and
    /// invalidates any restore map built before the change.
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> Labels { get; init; } = DefaultPII;

    /// <summary>Sigmoid probability floor for keeping a span.</summary>
    /// <remarks>
    /// A recall/precision dial whose two errors are not equally bad. A miss is a leak: PII goes out,
    /// permanently, into whatever the text is sent to. A false positive only costs a noisier prompt,
    /// since masking here is reversible and the restore map puts the text back. So the floor belongs
    /// below the middle, and the 0.1 default sits there. Recall falls as the surrounding text grows —
    /// a bare sentence is saturated at any usable threshold, long prose is not.
    /// </remarks>
    public float Threshold { get; init; } = 0.1f;

    /// <summary>
    /// The most text one masking call may carry, counted in tokens (the model's own units, not
    /// characters — 1,800 characters is 379 tokens of English prose but 1,752 of CJK, so a character
    /// budget means something different in every language).
    /// </summary>
    /// <remarks>
    /// A COST limit, distinct from <see cref="ModelPin.MaxSequenceLength"/>, which is a correctness
    /// limit the weights impose. Input longer than one window is split into several and masked in
    /// full, so nothing is lost — but each window is another inference pass, and unbounded input
    /// means unbounded passes with every queued request waiting behind them. It is a caller's choice
    /// rather than a fact about the weights, which is why it lives here and not on the pin.
    /// </remarks>
    public int MaxInputTokens { get; init; } = 2_000;

    // DECLARATION ORDER IS LOAD-BEARING. Static initializers run top to bottom, and `Default`'s
    // constructor reads `DefaultPII` for its `Labels`. Declaring `Default` first left every
    // `MaskerConfig.Default` with a NULL label set — which is not a crash but a total, silent
    // failure: masking throws on every call, the fail-closed gate holds every request, and the
    // product simply stops producing suggestions with nothing in the log to say why.
    // `LabelsAreNeverNull` pins this.

    /// <summary>
    /// The 17 labels this library ships with. GLiNER takes labels at inference time, so any label
    /// set works with any model — unlike <see cref="ModelPin.MaxWidth"/>, which must match the weights.
    /// </summary>
    public static IReadOnlyList<string> DefaultPII { get; } = new ReadOnlyCollection<string>(
    [
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
    ]);

    /// <summary>The default configuration: the shipped labels, a 0.1 floor, a 2,000-token budget.</summary>
    /// <remarks>Declared AFTER <see cref="DefaultPII"/> on purpose — see the note above.</remarks>
    public static MaskerConfig Default { get; } = new();
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The tier that needs real weights: tokenization → ONNX inference → masking.
/// </summary>
/// <remarks>
/// <para>
/// Opt in by pointing <c>PII_MASKER_MODEL_DIR</c> at a directory holding <c>model.onnx</c>,
/// <c>tokenizer.json</c> and <c>tokenizer_config.json</c>. Without it every test here is skipped,
/// so <c>dotnet test</c> on a machine with no model is green and means it.
/// </para>
/// <code>
/// PII_MASKER_MODEL_DIR=/path/to/gliner dotnet test
/// </code>
/// <para>
/// The counterpart of the Swift target's <c>ModelTierTests</c>, and the answer to the same problem:
/// every other test in this project is pure, so without this tier the entire detection path —
/// tokenizing, windowing, inference, span decoding, offset translation — ships unexercised, and a
/// change that quietly stopped detecting anything would still be green.
/// </para>
/// <para>
/// Not ported from Swift: the tests covering a masker that reloads after a store invalidation and
/// the ones racing concurrent loads. Both exercise Swift's actor reload semantics, which this
/// target does not share — its loading is guarded by a semaphore and a warm-up task, covered
/// without weights.
/// </para>
/// </remarks>
public sealed class ModelTierTests
{
    /// <summary>A phrase set per label, and the label it is expected to be caught by.</summary>
    /// <remarks>
    /// The assertion is deliberately "something was masked", not "masked under this exact label":
    /// the model sometimes files a span under a neighbouring label (a PIN as a password), and from
    /// a privacy standpoint both are the same answer — the content did not go out in the clear.
    /// </remarks>
    public static TheoryData<string, string[]> DetectionCases() => new()
    {
        { "PHONE_NUMBER", ["Please call me at 555-867-5309 tomorrow", "My number is +1 (212) 555-0198", "Reach me on 07700 900461"] },
        { "EMAIL_ADDRESS", ["Send it to alice.jones@company.com please", "Contact support@example.org for help", "Email me at bob123@gmail.com"] },
        { "CREDIT_CARD_NUMBER", ["My card is 4111-1111-1111-1111", "Use card number 5500 0000 0000 0004"] },
        { "ADDRESS", ["I live at 742 Evergreen Terrace, Springfield", "Ship to 1600 Pennsylvania Avenue, Washington DC"] },
        { "SOCIAL_SECURITY_NUMBER", ["My SSN is 123-45-6789", "Social security number 987-65-4321"] },
        { "DATE_OF_BIRTH", ["My date of birth is March 15, 1990", "Date of birth: 01/15/1985", "DOB: 1990-03-15", "She was born on 12/25/1988"] },
        { "BANK_ACCOUNT_NUMBER", ["My bank account number is 12345678901234", "Wire to account 9876543210 at Chase"] },
        { "PASSWORD", ["My password is hunter2", "The login credentials are admin / P@ssw0rd123", "Password: xK9#mQ2!vL7"] },
        { "PIN_CODE", ["My PIN code is 4829", "Enter PIN: 13334 to proceed", "The ATM pin is 9021"] },
        { "IP_ADDRESS", ["The server IP is 192.168.1.100", "Connect to 10.0.0.1 on port 443", "Blocked IP address 203.0.113.42"] },
        { "DOLLAR_AMOUNT", ["The total is $250,000", "I owe $15,000 in bills", "Paid $99.99 for the subscription"] },
        { "PASSPORT_NUMBER", ["My passport number is AB1234567", "Passport: X12345678 issued in London"] },
        { "DRIVER_LICENSE_NUMBER", ["Driver license number D123-4567-8901", "My driver license is S550-2400-1234"] },
        { "TAX_ID", ["My tax id is 12-3456789", "EIN / tax id: 98-7654321"] },
        { "API_KEY", ["The API key is sk-abc123def456ghi789", "Set your api key: AKIAIOSFODNN7EXAMPLE"] },
        { "ACCESS_TOKEN", ["Bearer access token eyJhbGciOiJIUzI1NiJ9.abc.xyz", "Use this access token: ghp_xxxxxxxxxxxxxxxxxxxx"] },
        { "SECRET_KEY", ["AWS secret key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "The secret key is 0x4a2b3c4d5e6f7890"] },
    };

    /// <summary>Every label category detects the entity it names, in at least one of its phrasings.</summary>
    [ModelTheory]
    [MemberData(nameof(DetectionCases))]
    public async Task DetectsEachLabelCategory(string label, string[] phrases)
    {
        using var masker = ModelTier.Masker();
        var reports = new List<string>();
        var anyMasked = false;
        foreach (var phrase in phrases)
        {
            var result = await masker.MaskAsync([phrase]);
            Assert.NotNull(result);
            var masked = result!.Fields[0];
            if (masked != phrase) anyMasked = true;
            reports.Add($"  {phrase}\n    → {masked}");
            // Whatever it did mask has to come back exactly, or the round trip is broken.
            Assert.Equal(phrase, PrivacyFilter.Restore(masked, result.Restore));
        }
        Assert.True(anyMasked, $"{label}: nothing masked in any phrasing\n{string.Join("\n", reports)}");
    }

    /// <summary>The shipped health probe passes against the installed model.</summary>
    /// <remarks>
    /// The one test that exercises the check a host is told to run after every install. If this
    /// fails, either the model is not the pinned one or the probe's own anchors have drifted.
    /// </remarks>
    [ModelFact]
    public async Task TheInstalledModelPassesItsProbe()
    {
        using var masker = ModelTier.Masker();
        var probe = await masker.ProbeAsync();

        Assert.True(probe.Ran, "the probe did not run at all");
        Assert.True(probe.Passes,
            $"probe found {probe.Found.Count} of {PrivacyFilter.ProbeAnchors.Count} anchors");
    }

    /// <summary>Several fields, one inference pass, one shared token map, and every field recovers.</summary>
    [ModelFact]
    public async Task MaskingBatchesAndRoundTrips()
    {
        using var masker = ModelTier.Masker();
        string[] fields =
        [
            "Call me on 415-555-0142 when you get in.",
            "My email is dana.brooks@example.com.",
            "the draft ",
        ];

        var result = await masker.MaskAsync(fields);
        Assert.NotNull(result);
        Assert.Equal(fields.Length, result!.Fields.Count);
        // The guarded trailing space survives: it is inside a field, not part of the framing.
        Assert.Equal("the draft ", result.Fields[2]);
        for (var index = 0; index < fields.Length; index++)
            Assert.Equal(fields[index], PrivacyFilter.Restore(result.Fields[index], result.Restore));

        Assert.NotEmpty(result.Restore);
        var sent = string.Join("\n", result.Fields);
        foreach (var original in result.Restore.Values)
            Assert.DoesNotContain(original, sent, StringComparison.Ordinal);
    }

    /// <summary>
    /// Input past one window is split into several passes rather than dropped, and PII at the very
    /// end — where detection died first — is still found.
    /// </summary>
    [ModelFact]
    public async Task LongInputIsWindowedAndStillFindsPII()
    {
        using var masker = ModelTier.Masker();
        const string anchor = "415-555-0142";
        var text = $"{ModelTier.Prose(1150)} please call me on {anchor} tomorrow.";

        var result = await masker.MaskAsync([text]);
        Assert.NotNull(result);
        Assert.DoesNotContain(anchor, result!.Fields[0], StringComparison.Ordinal);
        // The offsets survived the windowing, or the round trip would not close.
        Assert.Equal(text, PrivacyFilter.Restore(result.Fields[0], result.Restore));
    }

    /// <summary>
    /// An entity landing ON a window boundary is still found, because windows overlap.
    /// </summary>
    /// <remarks>
    /// Without the overlap a cut through <c>415-555-0142</c> hides it from both neighbours, and the
    /// leak returns in a form no length check would catch. Sweeping the filler length walks the
    /// anchor across the boundary rather than guessing where it falls.
    /// </remarks>
    [ModelFact]
    public async Task AnEntityOnAWindowBoundaryIsStillFound()
    {
        using var masker = ModelTier.Masker();
        const string anchor = "415-555-0142";
        var leaked = new List<int>();
        var brokenRoundTrip = new List<int>();

        for (var filler = 580; filler <= 720; filler += 20)
        {
            var text = $"{ModelTier.Prose(filler)} please call me on {anchor} tomorrow. ";
            var result = await masker.MaskAsync([text]);
            if (result == null) { leaked.Add(filler); continue; }
            if (result.Fields[0].Contains(anchor, StringComparison.Ordinal)) leaked.Add(filler);
            if (PrivacyFilter.Restore(result.Fields[0], result.Restore) != text)
                brokenRoundTrip.Add(filler);
        }

        Assert.True(leaked.Count == 0, $"the anchor survived at filler lengths: {string.Join(", ", leaked)}");
        Assert.True(brokenRoundTrip.Count == 0,
            $"the round trip broke at filler lengths: {string.Join(", ", brokenRoundTrip)}");
    }

    /// <summary>The cost ceiling is a caller's choice, and exceeding it drops rather than truncates.</summary>
    [ModelFact]
    public async Task TheInputBudgetIsConfigurableAndDrops()
    {
        using var masker = ModelTier.Masker(MaskerConfig.Default with { MaxInputTokens = 5 });

        Assert.Null(await masker.MaskAsync([ModelTier.Prose(200)]));
    }

    /// <summary>A deadline that cannot be met drops the pass rather than returning partial text.</summary>
    [ModelFact]
    public async Task AnImpossibleTimeoutDrops()
    {
        using var masker = new PrivacyFilter(
            ModelTier.ModelDirectory!, timeout: TimeSpan.FromMilliseconds(1));

        Assert.Null(await masker.MaskAsync([ModelTier.Prose(400)]));
    }
}

/// <summary>Where the weights are, and the filler the windowing tests run on.</summary>
internal static class ModelTier
{
    /// <summary>A directory holding the three files the loader opens, or null.</summary>
    internal static string? ModelDirectory
    {
        get
        {
            var path = Environment.GetEnvironmentVariable("PII_MASKER_MODEL_DIR");
            if (string.IsNullOrEmpty(path)) return null;
            return ModelInstaller.IsCompleteModelDir(path) ? path : null;
        }
    }

    internal static bool IsAvailable => ModelDirectory != null;

    internal static PrivacyFilter Masker(MaskerConfig? config = null) =>
        // Sixty seconds: the windowing tests run a thousand words through several passes, and the
        // four-second product default would make this a test of the machine's speed.
        new(ModelDirectory!, config: config, timeout: TimeSpan.FromSeconds(60));

    /// <summary>Varied prose filler, in words.</summary>
    /// <remarks>
    /// Deliberately several different sentences rather than one repeated: the model detects almost
    /// nothing in degenerate repetition, so repeated filler would make a windowing test fail for
    /// recall reasons and vice versa.
    /// </remarks>
    internal static string Prose(int words)
    {
        string[] pool =
        [
            "The quarterly report needs review before Thursday's leadership sync.",
            "I think we should postpone the launch until the metrics stabilise.",
            "Could you take a look at the draft when you get a chance?",
            "There were a few concerns raised about timeline and staffing.",
            "Let me know if the revised numbers change your recommendation.",
            "We agreed to revisit the pricing model after the pilot ends.",
        ];
        var output = new List<string>();
        for (var index = 0; output.Count < words; index++)
            output.AddRange(pool[index % pool.Length].Split(' '));
        return string.Join(' ', output.Take(words));
    }
}

/// <summary>A fact that is skipped unless real weights are available.</summary>
/// <remarks>
/// xunit 2 cannot decide <c>Skip</c> at run time, so it is decided here, when the attribute is
/// constructed. The alternative — asserting nothing and returning early — reports as a pass and
/// would let this whole tier silently stop running.
/// </remarks>
internal sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute()
    {
        if (!ModelTier.IsAvailable)
            Skip = "set PII_MASKER_MODEL_DIR to a directory holding the model to run this tier";
    }
}

/// <inheritdoc cref="ModelFactAttribute"/>
internal sealed class ModelTheoryAttribute : TheoryAttribute
{
    public ModelTheoryAttribute()
    {
        if (!ModelTier.IsAvailable)
            Skip = "set PII_MASKER_MODEL_DIR to a directory holding the model to run this tier";
    }
}

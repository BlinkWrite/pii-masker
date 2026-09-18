using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The masking logic that needs no weights: span trimming and merging, placeholder minting, the
/// restore round trip, the separator guard, and the probe's pass mark. Detection itself needs the
/// model and is covered by the model tier.
/// </summary>
public sealed class MaskingTests
{
    // Built numerically for the same reason the production constant is: an escape sequence or a
    // literal control character in source is one careless tool away from being silently rewritten.
    private const char Rs = (char)0x1E;

    private static DetectedEntity Entity(int start, int length, string label = "email address") =>
        new(start, length, label, 0.9f);

    private static string ApplyTo(string text, params DetectedEntity[] detections) =>
        PrivacyFilter.Apply(text, detections,
            new Dictionary<string, string>(StringComparer.Ordinal),
            new Dictionary<string, int>(StringComparer.Ordinal));

    [Fact]
    public void TheGuardAndSeparatorAreBuiltFromTheRecordSeparator()
    {
        Assert.Equal(new string(Rs, 2), PrivacyFilter.MaskGuard);
        Assert.Equal($" {new string(Rs, 3)} ", PrivacyFilter.MaskSeparator);
    }

    [Fact]
    public void ADetectedSpanBecomesANumberedPlaceholder()
    {
        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        var masked = PrivacyFilter.Apply(
            "Mail a@b.com now", [Entity(5, 7)], restore, counts);

        Assert.Equal("Mail [EMAIL_ADDRESS_1] now", masked);
        Assert.Equal("a@b.com", restore["[EMAIL_ADDRESS_1]"]);
    }

    /// <summary>Placeholders number per label, so two emails are 1 and 2 rather than both 1.</summary>
    [Fact]
    public void PlaceholdersNumberPerLabel()
    {
        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        var masked = PrivacyFilter.Apply(
            "a@b.com and c@d.com", [Entity(0, 7), Entity(12, 7)], restore, counts);

        Assert.Equal("[EMAIL_ADDRESS_1] and [EMAIL_ADDRESS_2]", masked);
        Assert.Equal(2, restore.Count);
    }

    /// <summary>
    /// Two labels claiming overlapping characters must mint ONE placeholder. Minting two would make
    /// the second replacement corrupt the first, and the restore map could then never put the
    /// original back.
    /// </summary>
    [Fact]
    public void OverlappingDetectionsAreMergedIntoOnePlaceholder()
    {
        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        var masked = PrivacyFilter.Apply(
            "call 415-555-0142 now",
            [Entity(5, 8, "phone number"), Entity(9, 8, "phone number")],
            restore, counts);

        Assert.Single(restore);
        Assert.Equal("call [PHONE_NUMBER_1] now", masked);
        Assert.Equal("415-555-0142", restore["[PHONE_NUMBER_1]"]);
    }

    /// <summary>
    /// Trailing punctuation must stay outside the placeholder, or the restored text loses the
    /// sentence's full stop and the caller's diff reports it as an edit.
    /// </summary>
    [Theory]
    [InlineData("Mail a@b.com.", 5, 8, "Mail [EMAIL_ADDRESS_1].")]
    [InlineData("Mail (a@b.com)", 5, 9, "Mail ([EMAIL_ADDRESS_1])")]
    [InlineData("Mail a@b.com, ok", 5, 8, "Mail [EMAIL_ADDRESS_1], ok")]
    public void SurroundingPunctuationIsTrimmedOutOfTheSpan(
        string text, int start, int length, string expected)
    {
        Assert.Equal(expected, ApplyTo(text, Entity(start, length)));
    }

    [Fact]
    public void ASpanThatTrimsToNothingIsDropped()
    {
        Assert.Equal("hello .  ", ApplyTo("hello .  ", Entity(5, 4)));
    }

    [Theory]
    [InlineData(-1, 5)]
    [InlineData(0, 0)]
    [InlineData(3, 99)]
    public void AnOutOfRangeSpanIsIgnoredRatherThanThrowing(int start, int length)
    {
        Assert.Equal("hello", ApplyTo("hello", Entity(start, length)));
    }

    [Fact]
    public void RestorePutsTheOriginalsBack()
    {
        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var masked = PrivacyFilter.Apply("Mail a@b.com now", [Entity(5, 7)], restore, counts);

        Assert.Equal("Mail a@b.com now", PrivacyFilter.Restore(masked, restore));
    }

    /// <summary>
    /// Longest key first: restoring <c>[EMAIL_ADDRESS_1]</c> before <c>[EMAIL_ADDRESS_11]</c> would
    /// eat the latter's prefix and leave a stray <c>1]</c> in the text.
    /// </summary>
    [Fact]
    public void RestoreDoesNotLetAShortPlaceholderEatALongerOne()
    {
        var restore = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["[EMAIL_ADDRESS_1]"] = "first@example.com",
            ["[EMAIL_ADDRESS_11]"] = "eleventh@example.com",
        };

        var restored = PrivacyFilter.Restore("[EMAIL_ADDRESS_11] and [EMAIL_ADDRESS_1]", restore);

        Assert.Equal("eleventh@example.com and first@example.com", restored);
    }

    /// <summary>
    /// A placeholder the model echoed that the map cannot account for must not reach the user, so it
    /// is scrubbed.
    /// </summary>
    [Fact]
    public void AnUnaccountedPlaceholderIsScrubbedFromTheReply()
    {
        var restored = PrivacyFilter.Restore(
            "Contact [EMAIL_ADDRESS_9] soon", new Dictionary<string, string>(StringComparer.Ordinal));

        Assert.Equal("Contact  soon", restored);
    }

    /// <summary>
    /// The scrub is scoped to labels this masker can mint. A general sweep of bracketed tokens would
    /// delete text the user typed themselves — and a caller diffing the reply against the user's
    /// unmasked draft would then report the deletion as a correction and write it into their document.
    /// </summary>
    [Fact]
    public void BracketedTextTheUserWroteThemselvesSurvives()
    {
        var restored = PrivacyFilter.Restore(
            "See [FIGURE_2] and [TODO_1] for the numbers.",
            new Dictionary<string, string>(StringComparer.Ordinal));

        Assert.Equal("See [FIGURE_2] and [TODO_1] for the numbers.", restored);
    }

    /// <summary>
    /// Input already containing the field separator would cross field boundaries on the way back,
    /// attributing one field's text to another. It must be refused — and refused BEFORE the model is
    /// loaded, which is what lets this be tested without weights.
    /// </summary>
    [Fact]
    public async Task InputContainingTheFieldSeparatorIsRefusedWithoutLoadingTheModel()
    {
        using var filter = new PrivacyFilter(
            modelDirectory: @"Z:\no\model\here", timeout: TimeSpan.FromSeconds(30));

        var result = await filter.MaskAsync(["harmless", $"smuggled{PrivacyFilter.MaskSeparator}field"]);

        Assert.Null(result);
    }

    [Fact]
    public async Task NoFieldsMasksToNoFields()
    {
        using var filter = new PrivacyFilter(@"Z:\no\model\here");

        var result = await filter.MaskAsync([]);

        Assert.NotNull(result);
        Assert.Empty(result.Fields);
        Assert.Empty(result.Restore);
    }

    /// <summary>A missing model must fail closed — null, never partly-masked text.</summary>
    [Fact]
    public async Task AMissingModelMasksNothingRatherThanReturningTheInput()
    {
        using var filter = new PrivacyFilter(@"Z:\no\model\here", timeout: TimeSpan.FromSeconds(30));

        Assert.Null(await filter.MaskAsync(["Email me at jane@example.com"]));
        Assert.Equal(ModelProbeResult.DidNotRun, await filter.ProbeAsync());
    }

    /// <summary>
    /// Two of three anchors, not all three: one miss is model jitter, two is a model that has stopped
    /// working. Requiring a clean sweep would revert healthy models over noise; requiring one would
    /// pass a model that has almost entirely failed.
    /// </summary>
    [Theory]
    [InlineData(true, 3, true)]
    [InlineData(true, 2, true)]
    [InlineData(true, 1, false)]
    [InlineData(true, 0, false)]
    [InlineData(false, 3, false)]
    public void TheProbePassMarkIsTwoOfThreeAnchors(bool ran, int foundCount, bool expected)
    {
        var found = PrivacyFilter.ProbeAnchors.Take(foundCount).ToList();
        var missed = PrivacyFilter.ProbeAnchors.Skip(foundCount).ToList();

        Assert.Equal(expected, new ModelProbeResult(ran, found, missed).Passes);
    }

    /// <summary>
    /// The regression for a static-initialisation-order bug that cost nothing to introduce and
    /// everything to hit: <c>Default</c> was declared before <c>DefaultPII</c>, so its constructor
    /// read a null label set. Nothing crashed at startup — masking simply threw on every call, the
    /// fail-closed gate held every request, and the product stopped producing suggestions with no
    /// error anywhere that named the cause. Reordering the two declarations fixes it; this makes
    /// reintroducing it a red build.
    /// </summary>
    [Fact]
    public void LabelsAreNeverNull()
    {
        Assert.NotNull(MaskerConfig.Default.Labels);
        Assert.NotEmpty(MaskerConfig.Default.Labels);
        Assert.Equal(MaskerConfig.DefaultPII, MaskerConfig.Default.Labels);
        Assert.NotNull(new MaskerConfig().Labels);
        Assert.NotEmpty(new MaskerConfig().Labels);
    }

    [Fact]
    public void TheProbeTextContainsEveryAnchorItChecksFor()
    {
        foreach (var anchor in PrivacyFilter.ProbeAnchors)
            Assert.Contains(anchor, PrivacyFilter.ProbeText, StringComparison.Ordinal);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// What happens to a detection that crosses the structural framing of a batched pass.
/// </summary>
/// <remarks>
/// <para>
/// Several fields are masked in one inference pass, joined by a guard and a separator built from
/// control characters. The detector knows nothing about that framing, so it can and does report a
/// span running from inside one field, across a separator, into the next — most easily when the
/// same credential appears in two fields, which is exactly what a chat context does when its last
/// message is also carried as <c>lastMessage</c>.
/// </para>
/// <para>
/// Such a span used to be DROPPED whole, on the reasoning that masking a marker would corrupt the
/// round trip. It would — but dropping the span leaves the detected secret unmasked on BOTH sides
/// of the marker, and it is then sent. The span is split around the marker instead, so the framing
/// survives and every detected fragment is still masked.
/// </para>
/// <para>
/// Ported from the Swift target's <c>MaskBoundaryTests</c>, case for case — the fix landed there
/// first, on the <c>fix/masked-field-boundaries</c> branch.
/// </para>
/// </remarks>
public sealed class MaskBoundaryTests
{
    private const string Secret1 = "synthetic-key-123";
    private const string Secret2 = "synthetic-key-456";
    private const string Secret3 = "synthetic-key-789";

    /// <summary>A span reaching from the FIRST occurrence of one substring to the LAST of another.</summary>
    /// <remarks>
    /// Last, not first: the point of every case here is a span that crosses the framing, and the
    /// two ends are often the same text repeated in two fields.
    /// </remarks>
    private static DetectedEntity Span(string joined, string from, string to)
    {
        var start = joined.IndexOf(from, StringComparison.Ordinal);
        var last = joined.LastIndexOf(to, StringComparison.Ordinal);
        var end = last + to.Length;
        Assert.True(start >= 0 && last >= 0 && end > start, $"span '{from}'..'{to}' not found");
        return new DetectedEntity(start, end - start, "api key", 1f);
    }

    /// <summary>
    /// Mask the batch with the given detections and assert the framing and every field survived.
    /// </summary>
    private static void Check(string[] values, params DetectedEntity[] detections)
    {
        var joined = PrivacyFilter.JoinForMasking(values);
        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var masked = PrivacyFilter.Apply(
            joined, PrivacyFilter.DropProtectedSpans(joined, detections), restore, counts);

        var fields = PrivacyFilter.SplitFields(masked, values.Length);
        Assert.NotNull(fields);
        Assert.Equal(values.Length, fields!.Length);
        for (var index = 0; index < values.Length; index++)
        {
            // Every field round-trips to exactly what came in...
            Assert.Equal(values[index], PrivacyFilter.Restore(fields[index], restore));
            // ...and no fragment of a secret is still sitting in what would be sent.
            Assert.DoesNotContain("synthetic-key-", fields[index], StringComparison.Ordinal);
        }
        Assert.NotEmpty(restore);
        // The framing never ends up inside a placeholder's value, or restoring would rebuild it
        // into the reply and the split would find guards where the text should be.
        Assert.DoesNotContain(restore.Values,
            value => value.Contains(PrivacyFilter.MaskGuard, StringComparison.Ordinal));
    }

    /// <summary>
    /// A credential duplicated into a second field, detected as one span across the separator.
    /// This is the shape that leaked: the whole span was dropped and both copies were sent.
    /// </summary>
    [Fact]
    public void ACredentialAcrossTwoFieldsKeepsBothFieldsMasked()
    {
        string[] values = ["SENDER|MESSAGE\nMorgan|" + Secret1, Secret1];
        var joined = PrivacyFilter.JoinForMasking(values);
        Check(values, Span(joined, Secret1, Secret1));
    }

    /// <summary>A span running off the end of the last field into the trailing guard.</summary>
    [Fact]
    public void ACredentialAndTheTrailingGuardRemainSeparate()
    {
        string[] values = ["Use " + Secret1];
        var joined = PrivacyFilter.JoinForMasking(values);
        Check(values, Span(joined, Secret1, PrivacyFilter.MaskGuard + ""));
    }

    /// <summary>And the same from the leading guard into the first field.</summary>
    [Fact]
    public void ACredentialAndTheLeadingGuardRemainSeparate()
    {
        string[] values = [Secret1 + " is the test credential"];
        var joined = PrivacyFilter.JoinForMasking(values);
        var start = joined.IndexOf(PrivacyFilter.MaskGuard, StringComparison.Ordinal);
        var end = joined.IndexOf(Secret1, StringComparison.Ordinal) + Secret1.Length;
        Check(values, new DetectedEntity(start, end - start, "api key", 1f));
    }

    /// <summary>
    /// Two spans that each cross a different separator and overlap in the middle field. The pieces
    /// must not merge across a marker — a merged placeholder would swallow the framing.
    /// </summary>
    [Fact]
    public void OverlappingDetectionsCannotMergeAcrossMarkers()
    {
        string[] values = [Secret1, Secret2, Secret3];
        var joined = PrivacyFilter.JoinForMasking(values);
        Check(values, Span(joined, Secret1, Secret2), Span(joined, Secret2, Secret3));
    }

    /// <summary>
    /// The padding around the separator, and any punctuation next to a secret, stay outside the
    /// placeholder — including when the field is unicode.
    /// </summary>
    [Fact]
    public void UnicodeWhitespaceAndPunctuationSurvive()
    {
        string[] values = ["\U0001F642 Café: (" + Secret1 + ")? \n", "\t‘" + Secret2 + "’! "];
        var joined = PrivacyFilter.JoinForMasking(values);
        Check(values, Span(joined, Secret1, Secret2));
    }

    /// <summary>
    /// One detection covering the entire batch — guards, separators, empty fields and all.
    /// </summary>
    /// <remarks>
    /// The degenerate case, and the one that proves the split is driven by the markers rather than
    /// by where the span happens to start: an empty field has to come back empty, not vanish.
    /// </remarks>
    [Fact]
    public void AWholeBatchDetectionPreservesEmptyFieldsAndGuards()
    {
        string[] values = [Secret1, "", "  ", Secret2];
        var joined = PrivacyFilter.JoinForMasking(values);
        Check(values, new DetectedEntity(0, joined.Length, "api key", 1f));
    }

    /// <summary>
    /// A field that carries the separator itself is refused before inference, so nothing downstream
    /// has to reason about a marker the caller planted.
    /// </summary>
    /// <remarks>
    /// The Swift target needs a different guard here: its markers are ordinary whitespace-delimited
    /// words, so it checks that a marker SUBSTRING inside a secret is not exempt. This target frames
    /// with a control character no user types, and rejects the input outright — covered by
    /// <c>MaskingTests.InputContainingTheFieldSeparatorIsRefusedWithoutLoadingTheModel</c>. Asserted
    /// here too, from the framing's side: a smuggled separator does not survive a round trip.
    /// </remarks>
    [Fact]
    public void AFieldCarryingTheSeparatorDoesNotSurviveTheRoundTrip()
    {
        string[] values = ["before" + PrivacyFilter.MaskSeparator + "after"];
        var joined = PrivacyFilter.JoinForMasking(values);

        Assert.Null(PrivacyFilter.SplitFields(joined, values.Length));
    }
}

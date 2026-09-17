using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace PIIMasker;

/// <summary>
/// Replaces the authenticated user's own name with a <c>[USER]</c> token. Pure — no model, no network.
/// </summary>
/// <remarks>
/// <para>
/// Two rules, both learned the same way: masking a name by matching each of its parts separately is
/// wrong twice over.
/// </para>
/// <para>
/// <b>A name fragment only ever matches as the GIVEN name, never as the surname.</b> A surname is
/// exactly what families and colleagues share, so matching it alone rewrites other people's names.
/// If the user is "A B" and a message mentions "C B", a surname match turns it into "C [USER]" —
/// and a downstream model that echoes the token back has it restored to the user's full name, so
/// the text a user reads names the wrong person. Matching only the given name is also how people
/// actually address someone ("Hi A", "@A can you check this").
/// </para>
/// <para>
/// <b>The masker never writes the sender column.</b> Who wrote a turn is settled before this runs.
/// A bare word match inside the sender field would override that verdict and tell a downstream
/// model the user said something another participant said.
/// </para>
/// <para>
/// First and last name are taken separately, not as one string: a full-name string carries no
/// reliable order — plenty of systems store it surname-first — and telling given name from surname
/// is the whole point. A host holding only a display name splits it itself, so that the guess stays
/// visible at the call site rather than buried in here.
/// </para>
/// </remarks>
public static class UserNameMask
{
    /// <summary>The placeholder a matched name becomes.</summary>
    /// <remarks>
    /// <c>[USER]</c> by convention; a caller whose prompts are written against a different token
    /// passes its own.
    /// </remarks>
    public const string DefaultToken = "[USER]";

    /// <summary>Which forms of the name may match.</summary>
    public enum Scope
    {
        /// <summary>
        /// Whole-name forms only, either order. For fields that round-trip verbatim back into the
        /// user's own text, where a wrong match lands in what they wrote.
        /// </summary>
        FullNameOnly,

        /// <summary>
        /// Whole-name forms plus the given name alone. For conversation text the user does not own,
        /// where being addressed by first name is the common case.
        /// </summary>
        GivenNameToo,
    }

    /// <summary>Mask every recognised spelling of the user's name.</summary>
    /// <param name="text">The text to mask.</param>
    /// <param name="firstName">The given name. Empty means no pattern, so nothing is masked.</param>
    /// <param name="lastName">The surname, possibly several words. Middle names ride along with it.</param>
    /// <param name="scope">Which forms may match — see <see cref="Scope"/>.</param>
    /// <param name="token">The placeholder to substitute. Defaults to <see cref="DefaultToken"/>.</param>
    public static string Mask(
        string text, string firstName, string lastName, Scope scope, string token = DefaultToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        var pattern = Pattern(firstName, lastName, scope);
        return pattern == null ? text : Apply(pattern, text, token);
    }

    /// <summary>Mask a transcript: message bodies only, sender column left alone.</summary>
    /// <remarks>
    /// <paramref name="senderHeader"/> is the transcript's first line (for example
    /// <c>"SENDER|MESSAGE"</c>). When the text starts with it, each line is split at the first
    /// <c>|</c> and only the part after it is masked. Text that is not in that format — or a null
    /// header — has no column to protect and is masked whole.
    /// </remarks>
    /// <param name="text">The transcript to mask.</param>
    /// <param name="firstName">The given name.</param>
    /// <param name="lastName">The surname.</param>
    /// <param name="senderHeader">The transcript's header line, or null for unstructured text.</param>
    /// <param name="token">The placeholder to substitute.</param>
    public static string MaskContext(
        string text, string firstName, string lastName, string? senderHeader = null,
        string token = DefaultToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        // Always the wider scope: conversation text is not the user's own writing, and being
        // addressed by first name is the common case there.
        var pattern = Pattern(firstName, lastName, Scope.GivenNameToo);
        if (pattern == null) return text;
        if (senderHeader == null || !text.StartsWith(senderHeader, StringComparison.Ordinal))
            return Apply(pattern, text, token);

        return string.Join('\n', text.Split('\n').Select(line =>
        {
            var bar = line.IndexOf('|');
            return bar < 0
                ? Apply(pattern, line, token)
                : line[..(bar + 1)] + Apply(pattern, line[(bar + 1)..], token);
        }));
    }

    /// <summary>
    /// A given name trailed by a capitalised word is someone else's full name ("@Alex Popescu"), so
    /// the fragment does not match there — the whole-name alternatives come first in the alternation
    /// and still claim the user's own full name.
    /// </summary>
    /// <remarks>
    /// Case sensitivity is forced back ON inside it: the pattern matches names caselessly, and under
    /// a caseless flag <c>\p{Lu}</c> folds too, so the guard would fire on any letter and swallow
    /// every fragment match. Best-effort: caseless scripts have no <c>\p{Lu}</c> to test.
    /// </remarks>
    private const string FollowedByAnotherName = @"(?-i:(?!\s+\p{Lu}))";

    /// <summary>Build the alternation, or null when there is no given name to anchor it.</summary>
    private static Regex? Pattern(string firstName, string lastName, Scope scope)
    {
        var given = (firstName ?? string.Empty).Trim();
        if (given.Length == 0) return null;

        // Middle names ride along with the surname, as they do in most profiles.
        var family = (lastName ?? string.Empty)
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(Regex.Escape)
            .ToList();
        var escapedGiven = Regex.Escape(given);
        var all = new[] { escapedGiven }.Concat(family).ToList();

        var alternatives = new List<string>();
        if (family.Count > 0)
        {
            var surname = family[^1];
            // "Given Family", "Given  Family"
            alternatives.Add(string.Join(@"\s+", all));
            // Surname first, comma optional: "Family, Given", "Family Given"
            alternatives.Add(string.Join(@",?\s+", family.Append(escapedGiven)));
            // "G. Family", "G Family"
            alternatives.Add(Regex.Escape(given[..1]) + @"\.?\s+" + surname);
            // Handles and email locals: "GivenFamily", "given.family"
            alternatives.Add(string.Concat(all));
            alternatives.Add(string.Join(@"\.", all));
        }

        // With no surname on the profile the given name IS the whole name, so it stands in both
        // scopes — still guarded, so someone else with the same given name keeps their name.
        //
        // The length floor gates THIS alternative only, never the whole pattern: a two-letter given
        // name matches too much ordinary text to stand alone, but "Jo Smith" is still the user's
        // own name, and refusing to mask it would send it in full.
        if ((scope == Scope.GivenNameToo || family.Count == 0) && given.Length >= 3)
            alternatives.Add(escapedGiven + FollowedByAnotherName);

        if (alternatives.Count == 0) return null;

        // Caseless via an inline flag rather than the option, so FollowedByAnotherName can turn it
        // back off for its uppercase test.
        //
        // The "(you)" marker some chat surfaces append sits OUTSIDE the word boundary, and applies
        // to every alternative. Swift writes it inside the first one, where it cannot fire: the
        // group ends on ")", and the trailing \b then needs a word character on one side, so the
        // engine backtracks to the bare name and leaves the marker behind. Swift's comment says the
        // marker goes with the name, which is the behaviour here — the divergence is against that
        // target's code, not its intent, and this target had it working already.
        return new Regex(@"\b(?i:" + string.Join("|", alternatives) + @")\b(?:\s*\(you\))?",
            RegexOptions.CultureInvariant);
    }

    // A `$` is a substitution in a .NET replacement template, so a caller's token carrying one
    // would be read as a group reference rather than as text.
    private static string Apply(Regex pattern, string text, string token) =>
        pattern.Replace(text, (token ?? DefaultToken).Replace("$", "$$", StringComparison.Ordinal));
}

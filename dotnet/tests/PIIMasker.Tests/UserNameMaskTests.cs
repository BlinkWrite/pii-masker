using Xunit;
using Scope = PIIMasker.UserNameMask.Scope;

namespace PIIMasker.Tests;

/// <summary>
/// The two rules in <see cref="UserNameMask"/>, both from the same failure.
/// </summary>
/// <remarks>
/// <para>
/// The user is "Alex Rivera"; a group member is "Sam Rivera"; the draft is the tag
/// "@Sam Rivera wha day i". A masker that matches each name part on its own matches the shared
/// SURNAME inside the other person's name, sends "@Sam [USER] wha day i" to the model, and the
/// response's token is restored to the user's full name — so the text that lands in the user's
/// composer reads "@Sam Rivera Alex wha day i". The same pass rewrites the sender column "Alex|"
/// to "[USER]|", telling the model the user asked a question another participant asked.
/// </para>
/// <para>
/// So: a name FRAGMENT only ever matches as the given name, never as the surname; and the masker
/// never writes the sender column.
/// </para>
/// <para>
/// Ported case for case from the Swift target's suite, because both targets have to make the same
/// promise about the same text — see <c>swift/Tests/PIIMaskerTests/UserNameMaskTests.swift</c>.
/// </para>
/// </remarks>
public sealed class UserNameMaskTests
{
    private const string Header = "SENDER|MESSAGE";

    /// <summary>
    /// A name fragment the user shares with someone else is that someone else. The surname alone
    /// must never match, in any field.
    /// </summary>
    [Fact]
    public void ASharedSurnameIsNotTheUser()
    {
        const string draft = "@Sam Rivera wha day i ";
        foreach (var scope in new[] { Scope.FullNameOnly, Scope.GivenNameToo })
            Assert.Equal(draft, UserNameMask.Mask(draft, "Alex", "Rivera", scope));

        const string prose = "I'll ask Sam Rivera about it";
        Assert.Equal(prose, UserNameMask.Mask(prose, "Alex", "Rivera", Scope.GivenNameToo));
    }

    /// <summary>
    /// The draft round-trips verbatim into the user's own composer, so only a whole-name match may
    /// be replaced there — either order, comma optional.
    /// </summary>
    [Theory]
    [InlineData("thanks, Alex Rivera", "thanks, [USER]")]
    [InlineData("thanks, Rivera Alex", "thanks, [USER]")]
    [InlineData("thanks, Rivera, Alex", "thanks, [USER]")]
    [InlineData("ask Rivera about it", "ask Rivera about it")]
    [InlineData("hi Alex how are you", "hi Alex how are you")]
    public void ADraftMatchesOnlyTheWholeName(string input, string expected)
    {
        Assert.Equal(expected, UserNameMask.Mask(input, "Alex", "Rivera", Scope.FullNameOnly));
    }

    /// <summary>
    /// Conversation text the user does not own: the given name alone is how people actually address
    /// them, and prompts commonly reason about <c>@[USER]</c> mentions.
    /// </summary>
    [Theory]
    [InlineData("Hi Alex, can you check this?", "Hi [USER], can you check this?")]
    [InlineData("@Alex can you check this", "@[USER] can you check this")]
    [InlineData("alex rivera said so", "[USER] said so")]
    public void TheGivenNameAloneMasksInConversationText(string input, string expected)
    {
        Assert.Equal(expected, UserNameMask.Mask(input, "Alex", "Rivera", Scope.GivenNameToo));
    }

    /// <summary>A given name trailed by a different surname is a different person.</summary>
    [Theory]
    [InlineData("@Alex Chen is on it")]
    [InlineData("Alex Chen is on it")]
    public void AGivenNameFollowedByAnotherSurnameIsSomeoneElse(string input)
    {
        Assert.Equal(input, UserNameMask.Mask(input, "Alex", "Rivera", Scope.GivenNameToo));
    }

    /// <summary>
    /// The same guard applies to a profile carrying no surname, where the given name is all there is.
    /// </summary>
    /// <remarks>
    /// This target used to match the bare name unguarded in that case, turning "Alex Chen is on it"
    /// into "[USER] Chen is on it" — someone else's name reported to the model as the user's.
    /// </remarks>
    [Fact]
    public void AGivenNameOnlyProfileIsGuardedTheSameWay()
    {
        Assert.Equal("Alex Chen is on it",
            UserNameMask.Mask("Alex Chen is on it", "Alex", "", Scope.GivenNameToo));
        Assert.Equal("Hi [USER], ready?",
            UserNameMask.Mask("Hi Alex, ready?", "Alex", "", Scope.GivenNameToo));
    }

    /// <summary>
    /// Who wrote a turn is decided before masking runs. A bare name match must not override that
    /// verdict — masking touches message bodies only.
    /// </summary>
    [Fact]
    public void TheSenderColumnIsNeverMasked()
    {
        const string transcript = """
            SENDER|MESSAGE
            Alex|Hey guys, what's the project status?
            [USER]|I am about to start testing
            Sam|Hey! Let me know how it goes
            """;
        Assert.Equal(transcript,
            UserNameMask.MaskContext(transcript, "Alex", "Rivera", Header));

        const string bodies = "SENDER|MESSAGE\nSam|Hi Alex, ready?\nAlex|yes";
        Assert.Equal("SENDER|MESSAGE\nSam|Hi [USER], ready?\nAlex|yes",
            UserNameMask.MaskContext(bodies, "Alex", "Rivera", Header));
    }

    /// <summary>
    /// Unstructured text has no sender column — mask it whole. Both when the caller declares a
    /// header the text does not carry, and when it declares none at all.
    /// </summary>
    [Fact]
    public void RawContextHasNoSenderColumn()
    {
        Assert.Equal("Hi [USER], ready?",
            UserNameMask.MaskContext("Hi Alex, ready?", "Alex", "Rivera", Header));
        Assert.Equal("Hi [USER], ready?",
            UserNameMask.MaskContext("Hi Alex, ready?", "Alex", "Rivera"));
    }

    /// <summary>
    /// A caller with no header declared gets NO column protection, even on text that happens to be
    /// pipe-shaped — the header is what identifies a transcript, and guessing would silently rewrite
    /// attribution for anyone whose text contains a pipe.
    /// </summary>
    [Fact]
    public void ColumnProtectionRequiresTheHeader()
    {
        Assert.Equal("SENDER|MESSAGE\n[USER]|hello",
            UserNameMask.MaskContext("SENDER|MESSAGE\nAlex|hello", "Alex", "Rivera"));
    }

    /// <summary>The token is the caller's to choose; the prompts a host writes may not use <c>[USER]</c>.</summary>
    [Fact]
    public void TheTokenIsConfigurable()
    {
        Assert.Equal("Hi <<ME>>",
            UserNameMask.Mask("Hi Alex", "Alex", "Rivera", Scope.GivenNameToo, "<<ME>>"));
    }

    /// <summary>
    /// A token carrying a dollar sign is text, not a substitution.
    /// </summary>
    /// <remarks>
    /// No Swift counterpart: <c>NSRegularExpression</c> escapes the template for the caller, while
    /// .NET reads <c>$1</c> in a replacement as a group reference. Left unescaped, a host whose
    /// token contained one would silently emit a capture — or nothing.
    /// </remarks>
    [Fact]
    public void ATokenContainingADollarSignIsNotReadAsASubstitution()
    {
        Assert.Equal("Hi $USER$",
            UserNameMask.Mask("Hi Alex", "Alex", "Rivera", Scope.GivenNameToo, "$USER$"));
    }

    /// <summary>
    /// No given name means no pattern, so nothing is masked — an unauthenticated caller must not
    /// have every capitalised word rewritten.
    /// </summary>
    [Theory]
    [InlineData("", "Rivera")]
    [InlineData("  ", "")]
    public void AnEmptyGivenNameMasksNothing(string first, string last)
    {
        const string text = "Hi Alex Rivera";
        Assert.Equal(text, UserNameMask.Mask(text, first, last, Scope.GivenNameToo));
    }

    /// <summary>Every shape the name is written in, from the .NET target's original suite.</summary>
    [Theory]
    [InlineData("Ask Jane Doe about it", "Ask [USER] about it")]
    [InlineData("Ask Doe, Jane about it", "Ask [USER] about it")]
    [InlineData("Ask J. Doe about it", "Ask [USER] about it")]
    [InlineData("Ask J Doe about it", "Ask [USER] about it")]
    [InlineData("Ask JaneDoe about it", "Ask [USER] about it")]
    [InlineData("Ask jane.doe about it", "Ask [USER] about it")]
    [InlineData("Ask JANE DOE about it", "Ask [USER] about it")]
    [InlineData("Ask jane doe about it", "Ask [USER] about it")]
    public void EveryShapeOfTheNameIsMasked(string input, string expected)
    {
        Assert.Equal(expected, UserNameMask.Mask(input, "Jane", "Doe", Scope.FullNameOnly));
    }

    /// <summary>Chat surfaces append "(you)" to the signed-in user; it goes with the name.</summary>
    /// <remarks>
    /// The one place this target's output differs from Swift's. Swift puts the marker inside the
    /// first alternative, where it cannot fire — the group ends on ")", so the pattern's trailing
    /// word boundary fails and the engine backtracks to the bare name, leaving " (you)" in the
    /// text. Its comment says the marker goes with the name. This target does that, for every
    /// alternative, and had it working before the port was reviewed; reproducing the Swift defect
    /// would have been a regression dressed up as parity.
    /// </remarks>
    [Theory]
    [InlineData("Jane Doe (you) said so", "[USER] said so")]
    [InlineData("Jane Doe(you) said so", "[USER] said so")]
    [InlineData("Doe, Jane (you) said so", "[USER] said so")]
    public void AYouMarkerIsConsumedWithTheName(string input, string expected)
    {
        Assert.Equal(expected, UserNameMask.Mask(input, "Jane", "Doe", Scope.FullNameOnly));
    }

    /// <summary>
    /// A name inside a longer word is not the user. Masking it would corrupt unrelated text — and
    /// the restore map cannot put back something that was never a placeholder.
    /// </summary>
    [Theory]
    [InlineData("The doeskin gloves")]
    [InlineData("Janeway reporting")]
    [InlineData("undoerror")]
    public void ANameInsideALongerWordIsLeftAlone(string input)
    {
        Assert.Equal(input, UserNameMask.Mask(input, "Jane", "Doe", Scope.GivenNameToo));
    }

    /// <summary>
    /// A given name under three characters never stands alone: it matches too much ordinary text.
    /// </summary>
    [Theory]
    [InlineData("Al")]
    [InlineData("Jo")]
    [InlineData("A")]
    public void AVeryShortGivenNameNeverStandsAlone(string name)
    {
        const string text = "Al and Jo went to A road";
        Assert.Equal(text, UserNameMask.Mask(text, name, "", Scope.GivenNameToo));
    }

    /// <summary>
    /// The length floor suppresses the bare-name rule, never the whole name.
    /// </summary>
    /// <remarks>
    /// This target used to return early on a short given name and mask NOTHING, so a user called
    /// "Jo Smith" had their full name sent verbatim — a leak, and the opposite of what the floor is
    /// for. The floor exists because "Jo" alone matches ordinary prose, not because "Jo Smith" is
    /// unrecognisable.
    /// </remarks>
    [Theory]
    [InlineData("Ask Jo Smith about it", "Ask [USER] about it")]
    [InlineData("Ask Smith, Jo about it", "Ask [USER] about it")]
    [InlineData("Ask jo.smith about it", "Ask [USER] about it")]
    [InlineData("Jo went to the shop", "Jo went to the shop")]
    public void AShortGivenNameStillMasksTheWholeName(string input, string expected)
    {
        Assert.Equal(expected, UserNameMask.Mask(input, "Jo", "Smith", Scope.GivenNameToo));
    }

    /// <summary>A single-word name is matched on its own, since there is no family name to pair it with.</summary>
    [Fact]
    public void ASingleWordNameIsMaskedOnItsOwn()
    {
        Assert.Equal("Ask [USER] about it",
            UserNameMask.Mask("Ask Madonna about it", "Madonna", "", Scope.FullNameOnly));
    }

    [Fact]
    public void EmptyTextIsReturnedUnchanged()
    {
        Assert.Equal("", UserNameMask.Mask("", "Jane", "Doe", Scope.GivenNameToo));
    }

    /// <summary>A middle name rides along with the surname, as it does in most profiles.</summary>
    [Fact]
    public void AMiddleNameRidesAlongWithTheSurname()
    {
        Assert.Equal("Ask [USER] about it",
            UserNameMask.Mask("Ask Jane Q Doe about it", "Jane", "Q Doe", Scope.FullNameOnly));
    }
}

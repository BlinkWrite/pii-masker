import Foundation
import Testing

@testable import PIIMasker

/// The two rules in `UserNameMask`, both from the same failure.
///
/// The user is "Alex Rivera"; a group member is "Sam Rivera"; the draft is the tag
/// "@Sam Rivera wha day i". A masker that matches each name part on its own matches the shared
/// SURNAME inside the other person's name, sends "@Sam [USER] wha day i" to the model, and the
/// response's token is restored to the user's full name — so the text that lands in the user's
/// composer reads "@Sam Rivera Alex wha day i". The same pass rewrites the sender column
/// "Alex|" to "[USER]|", telling the model the user asked a question another participant asked.
///
/// So: a name FRAGMENT only ever matches as the given name, never as the surname; and the masker
/// never writes the sender column.
@Suite("User-name masking")
struct UserNameMaskTests {
    static let header = "SENDER|MESSAGE"

    /// A name fragment the user shares with someone else is that someone else. The surname alone
    /// must never match, in any field.
    @Test func sharedSurnameIsNotTheUser() {
        let draft = "@Sam Rivera wha day i "
        for scope in [UserNameMask.Scope.fullNameOnly, .givenNameToo] {
            let got = UserNameMask.mask(
                draft, firstName: "Alex", lastName: "Rivera", scope: scope)
            #expect(got == draft, "\(scope): \(got)")
        }
        let prose = "I'll ask Sam Rivera about it"
        let got = UserNameMask.mask(
            prose, firstName: "Alex", lastName: "Rivera", scope: .givenNameToo)
        #expect(got == prose, "\(got)")
    }

    /// The draft round-trips verbatim into the user's own composer, so only a whole-name match may
    /// be replaced there — either order, comma optional.
    @Test func draftMatchesOnlyTheWholeName() {
        let cases: [(String, String)] = [
            ("thanks, Alex Rivera", "thanks, [USER]"),
            ("thanks, Rivera Alex", "thanks, [USER]"),
            ("thanks, Rivera, Alex", "thanks, [USER]"),
            ("ask Rivera about it", "ask Rivera about it"),
            ("hi Alex how are you", "hi Alex how are you"),
        ]
        for (input, want) in cases {
            let got = UserNameMask.mask(
                input, firstName: "Alex", lastName: "Rivera", scope: .fullNameOnly)
            #expect(got == want, "\"\(input)\" → \(got)")
        }
    }

    /// Conversation text the user does not own: the given name alone is how people actually address
    /// them, and prompts commonly reason about `@[USER]` mentions.
    @Test func givenNameAloneMasksInConversationText() {
        let cases: [(String, String)] = [
            ("Hi Alex, can you check this?", "Hi [USER], can you check this?"),
            ("@Alex can you check this", "@[USER] can you check this"),
            ("alex rivera said so", "[USER] said so"),
        ]
        for (input, want) in cases {
            let got = UserNameMask.mask(
                input, firstName: "Alex", lastName: "Rivera", scope: .givenNameToo)
            #expect(got == want, "\"\(input)\" → \(got)")
        }
    }

    /// A given name trailed by a different surname is a different person.
    @Test func givenNameFollowedByAnotherSurnameIsSomeoneElse() {
        for input in ["@Alex Chen is on it", "Alex Chen is on it"] {
            let got = UserNameMask.mask(
                input, firstName: "Alex", lastName: "Rivera", scope: .givenNameToo)
            #expect(got == input, "\(got)")
        }
    }

    /// Who wrote a turn is decided before masking runs. A bare name match must not override that
    /// verdict — masking touches message bodies only.
    @Test func senderColumnIsNeverMasked() {
        let ctx = """
            SENDER|MESSAGE
            Alex|Hey guys, what's the project status?
            [USER]|I am about to start testing
            Sam|Hey! Let me know how it goes
            """
        let got = UserNameMask.maskContext(
            ctx, firstName: "Alex", lastName: "Rivera", senderHeader: Self.header)
        #expect(got == ctx, "\(got)")

        let bodies = "SENDER|MESSAGE\nSam|Hi Alex, ready?\nAlex|yes"
        let want = "SENDER|MESSAGE\nSam|Hi [USER], ready?\nAlex|yes"
        let gotBodies = UserNameMask.maskContext(
            bodies, firstName: "Alex", lastName: "Rivera", senderHeader: Self.header)
        #expect(gotBodies == want, "\(gotBodies)")
    }

    /// Unstructured text has no sender column — mask it whole. Both when the caller declares a
    /// header the text doesn't carry, and when it declares none at all.
    @Test func rawContextHasNoSenderColumn() {
        let raw = "Hi Alex, ready?"
        let withHeader = UserNameMask.maskContext(
            raw, firstName: "Alex", lastName: "Rivera", senderHeader: Self.header)
        #expect(withHeader == "Hi [USER], ready?", "\(withHeader)")

        let noHeader = UserNameMask.maskContext(raw, firstName: "Alex", lastName: "Rivera")
        #expect(noHeader == "Hi [USER], ready?", "\(noHeader)")
    }

    /// A caller with no header declared gets NO column protection, even on text that happens to be
    /// pipe-shaped — the header is what identifies a transcript, and guessing would silently rewrite
    /// attribution for anyone whose text contains a pipe.
    @Test func columnProtectionRequiresTheHeader() {
        let transcript = "SENDER|MESSAGE\nAlex|hello"
        let unprotected = UserNameMask.maskContext(
            transcript, firstName: "Alex", lastName: "Rivera")
        #expect(unprotected == "SENDER|MESSAGE\n[USER]|hello", "\(unprotected)")
    }

    /// The token is the caller's to choose; the prompts a host writes may not use `[USER]`.
    @Test func tokenIsConfigurable() {
        let got = UserNameMask.mask(
            "Hi Alex", firstName: "Alex", lastName: "Rivera", scope: .givenNameToo,
            token: "<<ME>>")
        #expect(got == "Hi <<ME>>", "\(got)")
    }

    /// No given name means no pattern, so nothing is masked — an unauthenticated caller must not
    /// have every capitalised word rewritten.
    @Test func emptyGivenNameMasksNothing() {
        let text = "Hi Alex Rivera"
        #expect(UserNameMask.mask(text, firstName: "", lastName: "Rivera", scope: .givenNameToo) == text)
        #expect(UserNameMask.mask(text, firstName: "  ", lastName: "", scope: .fullNameOnly) == text)
    }
}

import Foundation

/// Replaces the authenticated user's own name with a `[USER]` token. Pure — no model, no network.
///
/// Two rules, both learned the same way: masking a name by matching each of its parts separately
/// is wrong twice over.
///
/// **A name fragment only ever matches as the GIVEN name, never as the surname.** A surname is
/// exactly what families and colleagues share, so matching it alone rewrites other people's names.
/// If the user is "A B" and a message mentions "C B", a surname match turns it into "C [USER]" —
/// and a downstream model that echoes the token back has it restored to the user's full name, so
/// the text a user reads names the wrong person. Matching only the given name is also how people
/// actually address someone ("Hi A", "@A can you check this").
///
/// **The masker never writes the sender column.** Who wrote a turn is settled before this runs. A
/// bare word match inside the sender field would override that verdict and tell a downstream model
/// the user said something another participant said.
///
/// First and last name are taken separately, not as one string: a full-name string carries no
/// reliable order — plenty of systems store it surname-first — and telling given name from surname
/// is the whole point.
public enum UserNameMask {
    /// The placeholder a matched name becomes. `[USER]` by convention; a caller whose prompts are
    /// written against a different token passes its own.
    public static let defaultToken = "[USER]"

    /// Which forms of the name may match.
    public enum Scope {
        /// Whole-name forms only, either order. For fields that round-trip verbatim back into the
        /// user's own text, where a wrong match lands in what they wrote.
        case fullNameOnly
        /// Whole-name forms plus the given name alone. For conversation text the user doesn't own,
        /// where being addressed by first name is the common case.
        case givenNameToo
    }

    public static func mask(
        _ text: String, firstName: String, lastName: String, scope: Scope,
        token: String = defaultToken
    ) -> String {
        guard let re = pattern(firstName: firstName, lastName: lastName, scope: scope) else {
            return text
        }
        return apply(re, to: text, token: token)
    }

    /// Mask a transcript: message bodies only, sender column left alone.
    ///
    /// `senderHeader` is the transcript's first line (e.g. `"SENDER|MESSAGE"`). When the text
    /// starts with it, each line is split at the first `|` and only the part after it is masked.
    /// Text that isn't in that format — or a nil header — has no column to protect and is masked
    /// whole.
    public static func maskContext(
        _ text: String, firstName: String, lastName: String, senderHeader: String? = nil,
        token: String = defaultToken
    ) -> String {
        guard
            let re = pattern(firstName: firstName, lastName: lastName, scope: .givenNameToo)
        else { return text }
        guard let senderHeader, text.hasPrefix(senderHeader) else {
            return apply(re, to: text, token: token)
        }
        return text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                guard let bar = line.firstIndex(of: "|") else {
                    return apply(re, to: String(line), token: token)
                }
                return String(line[...bar])
                    + apply(re, to: String(line[line.index(after: bar)...]), token: token)
            }
            .joined(separator: "\n")
    }

    /// A given name trailed by a capitalized word is someone else's full name ("@Alex Popescu"),
    /// so the fragment doesn't match there — the whole-name alternatives come first in the
    /// alternation and still claim the user's own full name. Case sensitivity is forced back ON
    /// inside it: the pattern matches names caselessly, and under a global caseless flag ICU folds
    /// `\p{Lu}` too, so the guard would fire on any letter and swallow every fragment match.
    /// Best-effort: caseless scripts have no `\p{Lu}` to test.
    static let followedByAnotherName = "(?-i:(?!\\s+\\p{Lu}))"

    static func pattern(firstName: String, lastName: String, scope: Scope)
        -> NSRegularExpression?
    {
        let given = firstName.trimmingCharacters(in: .whitespaces)
        guard !given.isEmpty else { return nil }
        // Middle names ride along with the surname, as they do in most profiles.
        let family = lastName.split(separator: " ").map(String.init)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .map { NSRegularExpression.escapedPattern(for: $0) }
        let g = NSRegularExpression.escapedPattern(for: given)
        let all = [g] + family

        var alternatives: [String] = []
        if let surname = family.last {
            // "Given Family", "Given  Family", "Given Family(you)"
            alternatives.append(all.joined(separator: "\\s+") + "(?:\\s*\\(you\\))?")
            // Surname first, comma optional: "Family, Given", "Family Given"
            alternatives.append((family + [g]).joined(separator: ",?\\s+"))
            // "G. Family", "G Family"
            alternatives.append(
                NSRegularExpression.escapedPattern(for: String(given.prefix(1)))
                    + "\\.?\\s+" + surname)
            // Handles and email locals: "GivenFamily", "given.family"
            alternatives.append(all.joined())
            alternatives.append(all.joined(separator: "\\."))
        }
        // With no surname on the profile the given name IS the whole name, so it stands in both
        // scopes — still guarded, so someone else with the same given name keeps their name.
        if scope == .givenNameToo || family.isEmpty, given.count >= 3 {
            alternatives.append(g + followedByAnotherName)
        }
        guard !alternatives.isEmpty else { return nil }

        // Caseless via an inline flag rather than the option, so `followedByAnotherName` can turn
        // it back off for its uppercase test.
        return try? NSRegularExpression(
            pattern: "\\b(?i:" + alternatives.joined(separator: "|") + ")\\b")
    }

    private static func apply(_ re: NSRegularExpression, to text: String, token: String) -> String {
        re.stringByReplacingMatches(
            in: text, range: NSRange(text.startIndex..., in: text),
            withTemplate: NSRegularExpression.escapedTemplate(for: token))
    }
}

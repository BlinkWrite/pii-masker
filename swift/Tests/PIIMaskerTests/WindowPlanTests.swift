import Foundation
import Testing

@testable import PIIMasker

/// The window planner, with no model on disk.
///
/// Every other windowing test lives in ``ModelTierTests`` and is gated on `PII_MASKER_MODEL_DIR`,
/// so without this suite the split — the mechanism that exists to stop a silent under-masking leak
/// — is untested on a machine with no weights. The arithmetic needs nothing but each word's token
/// count, so it is pinned exactly here: the cut positions, the overlap width, the fail-closed cases
/// and the termination argument.
///
/// What is asserted here is the PLAN. Whether the model then finds an entity inside a window is
/// recall, and it belongs with the weights.
@Suite("Window planning")
struct WindowPlanTests {
    static let maxWidth = 12   // ModelPin.current.maxWidth

    static func plan(_ counts: [Int], capacity: Int, maxWidth: Int = maxWidth) -> [Range<Int>]? {
        PrivacyFilter.planWindowRanges(
            tokenCounts: counts, capacity: capacity, maxWidth: maxWidth)
    }

    // MARK: - Exact cuts

    /// The cut positions themselves, not a behaviour that implies them. One token per word and a
    /// capacity of 40 makes every number here readable by hand: each window takes 40 words, then
    /// steps back 11 so the next one re-reads the tail.
    @Test func cutsLandWhereTheArithmeticSays() {
        let got = Self.plan(Array(repeating: 1, count: 100), capacity: 40)
        #expect(got == [0..<40, 29..<69, 58..<98, 87..<100], "\(got.map(Array.init) ?? [])")
    }

    /// A realistic shape: input long enough for three windows, against the capacity the 17 default
    /// labels leave of a 768-token window. Two windows is the case every other test covers, and it
    /// is the one where a middle window — bounded on both sides by an overlap — never appears.
    @Test func aRealisticInputSplitsIntoThreeWindows() {
        let got = Self.plan(Array(repeating: 1, count: 1_618), capacity: 711)
        #expect(got == [0..<711, 700..<1411, 1400..<1618], "\(got.map(Array.init) ?? [])")
    }

    /// One window when everything fits, and no phantom second window at exactly capacity.
    @Test func inputWithinOneWindowIsNotSplit() {
        #expect(Self.plan(Array(repeating: 1, count: 711), capacity: 711) == [0..<711])
        #expect(Self.plan(Array(repeating: 1, count: 712), capacity: 711)?.count == 2)
        #expect(Self.plan([], capacity: 711) == [])
    }

    // MARK: - The guarantee the overlap buys

    /// The reason the overlap exists: a GLiNER span is at most `maxWidth` words wide, so every run
    /// of `maxWidth` consecutive words must sit WHOLLY inside at least one window. A cut through
    /// `alice@ example.com` otherwise hides the email from both neighbours, and that leak is
    /// invisible — no length check catches it, and the pass reports success.
    ///
    /// Checked over uneven token counts, because equal-width words are the one case where a cut
    /// can never land awkwardly.
    @Test func everyRunOfMaxWidthWordsSitsInsideOneWindow() {
        var rng = LCG(seed: 0x5EED)
        for trial in 0..<200 {
            let n = Int.random(in: 40...600, using: &rng)
            // Word lengths a tokenizer actually produces: mostly 1–3 subwords, occasionally more.
            let counts = (0..<n).map { _ in Int.random(in: 1...4, using: &rng) }
            let capacity = Int.random(in: 60...400, using: &rng)
            guard let windows = Self.plan(counts, capacity: capacity) else {
                Issue.record("trial \(trial): plan failed on counts that all fit")
                continue
            }
            // Only meaningful where windows are at least `maxWidth` words wide — a narrower window
            // cannot hold a `maxWidth`-word span in the first place, so the model could not emit
            // one. `narrowWindowsShrinkTheOverlap` pins that case separately.
            guard windows.allSatisfy({ $0.count >= Self.maxWidth }) else { continue }
            for start in 0...(max(0, n - Self.maxWidth)) {
                let run = start..<min(start + Self.maxWidth, n)
                let covered = windows.contains { $0.lowerBound <= run.lowerBound && run.upperBound <= $0.upperBound }
                #expect(covered, "trial \(trial): words \(run) fall in no single window")
                if !covered { break }
            }
        }
    }

    /// Consecutive windows step back by exactly `maxWidth - 1` words whenever the window is wide
    /// enough to allow it. Fewer would reopen the split-entity hole; more would just cost passes.
    @Test func consecutiveWindowsOverlapByMaxWidthMinusOne() {
        var rng = LCG(seed: 0xC0FFEE)
        for _ in 0..<200 {
            let counts = (0..<Int.random(in: 40...600, using: &rng))
                .map { _ in Int.random(in: 1...4, using: &rng) }
            guard let w = Self.plan(counts, capacity: Int.random(in: 60...400, using: &rng)),
                  w.count > 1
            else { continue }
            for (a, b) in zip(w, w.dropFirst()) where a.count > Self.maxWidth {
                #expect(a.upperBound - b.lowerBound == Self.maxWidth - 1,
                        "overlap \(a.upperBound - b.lowerBound) between \(a) and \(b)")
            }
        }
    }

    // MARK: - Invariants that must hold for every plan

    /// No word may fall outside every window, and no window may exceed the capacity it was planned
    /// against. The first is a leak; the second is the over-length input the whole mechanism exists
    /// to prevent.
    @Test func everyWordIsCoveredAndNoWindowExceedsCapacity() {
        var rng = LCG(seed: 0xBEEF)
        for trial in 0..<300 {
            let n = Int.random(in: 1...500, using: &rng)
            let counts = (0..<n).map { _ in Int.random(in: 1...6, using: &rng) }
            // The low end of this range is below the largest word, so some trials genuinely hit
            // the oversized-word branch rather than only ever exercising the happy path.
            let capacity = Int.random(in: 3...200, using: &rng)
            guard let windows = Self.plan(counts, capacity: capacity) else {
                // Only legitimate when some single word cannot fit a whole window.
                #expect(counts.contains { $0 > capacity }, "trial \(trial): plan failed but every word fits")
                continue
            }
            var seen = Set<Int>()
            for w in windows {
                #expect(counts[w].reduce(0, +) <= capacity, "trial \(trial): window \(w) over capacity")
                #expect(!w.isEmpty, "trial \(trial): empty window")
                seen.formUnion(w)
            }
            #expect(seen.count == n, "trial \(trial): \(n - seen.count) words fell outside every window")
            // Windows advance; a plan that revisits ground would loop.
            for (a, b) in zip(windows, windows.dropFirst()) {
                #expect(b.lowerBound > a.lowerBound, "trial \(trial): \(b) does not advance past \(a)")
            }
        }
    }

    /// A window too narrow to hold `maxWidth` words still covers everything and still terminates —
    /// the step-back shrinks to what the window allows instead of stalling. The `maxWidth`-run
    /// guarantee genuinely degrades here, and that is a property of the capacity, not a bug in the
    /// plan: the model cannot emit a span wider than the window it ran on.
    @Test func narrowWindowsShrinkTheOverlap() {
        let counts = Array(repeating: 1, count: 50)
        let got = Self.plan(counts, capacity: 4)
        #expect(got == [0..<4, 1..<5, 2..<6, 3..<7, 4..<8, 5..<9, 6..<10, 7..<11, 8..<12, 9..<13,
                        10..<14, 11..<15, 12..<16, 13..<17, 14..<18, 15..<19, 16..<20, 17..<21,
                        18..<22, 19..<23, 20..<24, 21..<25, 22..<26, 23..<27, 24..<28, 25..<29,
                        26..<30, 27..<31, 28..<32, 29..<33, 30..<34, 31..<35, 32..<36, 33..<37,
                        34..<38, 35..<39, 36..<40, 37..<41, 38..<42, 39..<43, 40..<44, 41..<45,
                        42..<46, 43..<47, 44..<48, 45..<49, 46..<50],
                "\(got.map(Array.init) ?? [])")
    }

    /// Every word exactly filling a whole window leaves no room to step back at all. The plan must
    /// still advance one word at a time rather than loop forever.
    @Test func wordsThatEachFillAWindowStillTerminate() {
        let got = Self.plan(Array(repeating: 20, count: 5), capacity: 20)
        #expect(got == [0..<1, 1..<2, 2..<3, 3..<4, 4..<5], "\(got.map(Array.init) ?? [])")
    }

    // MARK: - Fail closed

    /// A single word that does not fit a whole window — a key, a token, an unbroken blob. No cut
    /// rescues it, and it is exactly the string that must not go out unmasked, so the plan fails
    /// rather than masking around it.
    @Test func aWordLargerThanAWindowFailsClosed() {
        #expect(Self.plan([1, 1, 500, 1], capacity: 100) == nil)
        #expect(Self.plan([101], capacity: 100) == nil)
        // One token under is fine — the boundary is `<=`, not `<`.
        #expect(Self.plan([100], capacity: 100) == [0..<1])
    }

    /// A window with no room for text — every token spent on the label preamble — cannot be split
    /// into anything usable, so it fails rather than returning an empty plan that masks nothing.
    @Test func noCapacityFailsClosed() {
        #expect(Self.plan([1, 1, 1], capacity: 0) == nil)
        #expect(Self.plan([1, 1, 1], capacity: -30) == nil)
    }
}

/// Reproducible randomness: a failing trial has to be re-runnable from the seed alone.
struct LCG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed &* 6_364_136_223_846_793_005 &+ 1 }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

import Foundation
import Hub
import OnnxRuntimeBindings
import Tokenizers
import os

/// Local, reversible PII masker. Detects entities with GLiNER (ONNX INT8, run on-device through
/// ONNX Runtime) and replaces each with a unique labelled placeholder, so text can be sent to a
/// remote model and the originals swapped back into its response locally.
///
/// Masking is reversible by design: a detected span becomes `[IP_ADDRESS_1]`, not an opaque
/// `*****`. A false positive therefore costs a placeholder in text the caller restores, never a
/// redaction marker bleeding into what a user reads — and the remote model still sees a labelled
/// token it can reason around.
///
/// The gate is fail-closed. Every entry point returns nil rather than raw text when the model is
/// missing, the load fails, or inference throws.
public actor PrivacyFilter {
    private var session: ORTSession?
    private var tokenizer: Tokenizer?
    private var isLoaded = false
    private var loadedGeneration = -1

    private let maxWidth: Int
    /// See ``ModelPin/maxSequenceLength``. An input that tokenizes past this is dropped, not masked
    /// partially — the one place this library could otherwise fail open.
    private let maxSequenceLength: Int
    private let piiLabels: [String]
    private let threshold: Float
    /// See ``MaskerConfig/maxInputTokens``. The cost ceiling, as opposed to `maxSequenceLength`'s
    /// correctness ceiling.
    private let maxInputTokens: Int
    private let logging: MaskerLogging
    let log: os.Logger

    // GLiNER special token IDs (from the exported model)
    private let clsTokenId: Int64 = 1
    private let sepTokenId: Int64 = 2
    private let entTokenId: Int64 = 128001
    private let entSepTokenId: Int64 = 128002

    private let firstName: String
    private let lastName: String
    private let senderHeader: String?
    /// Pins this masker to one model directory instead of resolving the install root. The rollback
    /// probe needs it: it judges a specific store, and a verdict formed against the configured
    /// location while reverting some other root would revert a model that was never probed.
    private let modelDirectory: URL?

    /// - Parameters:
    ///   - firstName: The authenticated user's given name, for the `[USER]` pass. Taken apart from
    ///     the surname rather than as one string, because a single name has no reliable order and
    ///     telling given name from surname is the whole point — see ``UserNameMask``. Empty
    ///     disables name masking.
    ///   - config: Labels, score threshold and the input-token budget. See ``MaskerConfig``.
    ///   - pin: The model this masker expects, for its ``ModelPin/maxWidth`` and
    ///     ``ModelPin/maxSequenceLength``. It must match the weights actually on disk: a wrong
    ///     `maxWidth` reads the logits tensor with the wrong stride, and a too-high
    ///     `maxSequenceLength` lets over-length input through to a model that silently stops
    ///     detecting.
    ///   - senderHeader: First line of a `SENDER|MESSAGE`-style transcript. When the text starts
    ///     with it, the name pass rewrites message bodies only and leaves the sender column alone.
    ///     nil (the default) masks the text whole.
    ///   - modelDirectory: Load from exactly this directory instead of resolving one.
    public init(
        firstName: String = "",
        lastName: String = "",
        config: MaskerConfig = .default,
        pin: ModelPin = .current,
        logging: MaskerLogging = .silent,
        senderHeader: String? = nil,
        modelDirectory: URL? = nil
    ) {
        self.firstName = firstName
        self.lastName = lastName
        self.piiLabels = config.labels
        self.threshold = config.threshold
        self.maxInputTokens = config.maxInputTokens
        self.maxWidth = pin.maxWidth
        self.maxSequenceLength = pin.maxSequenceLength
        self.logging = logging
        self.log = logging.logger("privacy")
        self.senderHeader = senderHeader
        self.modelDirectory = modelDirectory
    }

    /// Whether this masker has a model to load at all — its own pinned directory, or one the
    /// process-wide store resolves. Asking the store alone would be wrong for a pinned masker: it
    /// can load perfectly while the store is unconfigured, and the gate would then drop every
    /// request forever.
    var hasResolvableModel: Bool {
        modelDirectory != nil || ModelStore.modelIsInstalled()
    }

    /// Build the ONNX session and tokenizer off the caller's path, so the first real request does
    /// not pay for the load.
    public nonisolated func warmUp() {
        Task.detached { [weak self] in
            _ = await self?.ensureLoaded()
        }
    }

    /// Outcome of sanitizing a prompt: the masked `text` that may be sent to the (possibly remote)
    /// model, plus the `restore` map (placeholder → original) used to swap real values back into
    /// that model's *response*, locally.
    public struct Sanitized: Sendable, Equatable {
        public let text: String
        public let restore: [String: String]
        public init(text: String, restore: [String: String]) {
            self.text = text
            self.restore = restore
        }
    }

    /// Detect PII spans in the raw text and replace each with a unique, reversible placeholder.
    /// The current user's name is replaced with `[USER]` (regex, before GLiNER) so a downstream
    /// model can still tell whose text is whose. The returned map puts the originals back.
    ///
    /// - Parameters:
    ///   - protecting: Structural marker words a caller embedded so several fields can be masked
    ///     in one pass (the join's guard/separator). GLiNER can false-positive a marker as PII;
    ///     masking it would corrupt the caller's round-trip. Spans crossing a marker are split
    ///     around it so the detected content on both sides is still masked.
    ///   - userNameMasking: `false` skips the `[USER]` pass — for a caller that joined several
    ///     fields into one blob and already masked each by ``UserNameMask``'s per-field rules,
    ///     which this method can no longer tell apart.
    /// - Returns: nil on a load or inference failure. Never partial, never raw.
    public func sanitize(
        _ rawContext: String, protecting: Set<String> = [], userNameMasking: Bool = true
    ) async -> Sanitized? {
        guard let (text, spans) = await detect(rawContext, userNameMasking: userNameMasking)
        else { return nil }
        guard !spans.isEmpty else { return Sanitized(text: text, restore: [:]) }
        // GLiNER spans are whitespace-delimited words, so trailing sentence punctuation rides into
        // the span ("ip?" → one word). Masking it whole hides the "?" inside the placeholder; the
        // model, seeing a bare token, re-adds its own "?", and `restore` brings the original back —
        // yielding "ip??". Trim boundary punctuation so the placeholder covers only the entity and
        // the punctuation stays as visible literal text.
        let labelled = spans.compactMap {
            span -> (start: Int, end: Int, label: String)? in
            guard let t = Self.trimSpan(text, start: span.start, end: span.end)
            else { return nil }
            return (start: t.start, end: t.end, label: piiLabels[span.labelIdx])
        }
        let shielded = Self.dropProtectedSpans(text, labelled, protecting: protecting)
        guard !shielded.isEmpty else { return Sanitized(text: text, restore: [:]) }
        let (masked, restore) = Self.mask(text, spans: shielded)
        logging.trace(
            "privacy masked (\(masked.count) chars, \(restore.count) tokens): \(String(masked.prefix(400)))"
        )
        return Sanitized(text: masked, restore: restore)
    }

    /// Shared GLiNER inference: returns the user-name-masked text plus the detected entity char
    /// spans into it (empty when nothing is detected), or nil on a load/inference failure.
    ///
    /// Long input is run in several overlapping windows rather than one oversized pass — see
    /// ``planWindows(_:capacity:)``. Every window reports spans as character offsets into the same
    /// original string, so the windows are an inference detail: nothing downstream of here knows
    /// how many there were, and the masking, numbering and restore map are unchanged.
    private func detect(_ rawContext: String, userNameMasking: Bool = true)
        async -> (text: String, spans: [(start: Int, end: Int, labelIdx: Int)])?
    {
        let trimmed = rawContext.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard await ensureLoaded() else { return nil }

        let start = Date()

        // 0. Replace current user's name with [USER] before GLiNER
        let userMasked = userNameMasking ? maskUserName(trimmed) : trimmed

        // 1. Split into words
        let words = Self.splitWords(userMasked)
        guard !words.isEmpty else { return (userMasked, []) }

        guard let tok = tokenizer, let sess = session else { return nil }

        // 2. Tokenize once. Both the window plan and the inference need the per-word token counts,
        //    and tokenizing twice would double the only non-inference cost on this path.
        let tokenized = tokenize(words, tokenizer: tok)
        let textTokens = tokenized.reduce(0) { $0 + $1.tokens.count }

        // 2a. The resource cap: how much work one call may cost.
        //
        // Distinct from the per-window limit below, and for a different reason. That one is about
        // correctness — the model cannot read past it. This one is about cost: windows are cheap
        // individually (~1 ms per word) but unbounded input means unbounded passes on a serialized
        // actor, and every request queued behind it waits. Refusing here costs microseconds; a
        // timeout costs the whole budget and then drops anyway.
        //
        // It lives in ``MaskerConfig`` rather than on the pin because it is a caller's choice, not
        // a fact about the weights — new weights do not change what a host can afford.
        guard textTokens <= maxInputTokens else {
            log.error(
                "privacy: input is \(textTokens, privacy: .public) tokens, over the configured \(self.maxInputTokens, privacy: .public)-token budget — dropping"
            )
            return nil
        }

        // 3. Plan the windows. Each pays the label preamble out of its own budget.
        let capacity = maxSequenceLength - preambleTokenCount(tokenizer: tok)
        guard let windows = planWindows(tokenized, capacity: capacity) else {
            log.error(
                "privacy: cannot fit this input into \(self.maxSequenceLength, privacy: .public)-token windows — a single word exceeds one whole window; dropping"
            )
            return nil
        }

        // 4. Run each window. Any window failing fails the whole pass: a partial answer here is
        //    exactly the silent under-masking this design exists to prevent.
        var spans: [(start: Int, end: Int, labelIdx: Int)] = []
        for (i, window) in windows.enumerated() {
            guard let found = runWindow(window, index: i, of: windows.count, session: sess, tokenizer: tok)
            else { return nil }
            spans.append(contentsOf: found)
        }

        if logging.debug != nil {
            let elapsed = Date().timeIntervalSince(start)
            logging.trace(
                "privacy: \(spans.count) spans over \(windows.count) window(s) \(String(format: "%.0f", elapsed * 1000))ms input=\(userMasked.count)chars words=\(words.count) textTokens=\(textTokens)"
            )
        }
        return (userMasked, spans)
    }

    // MARK: - Windowing

    /// A word with its subword token ids, tokenized once and reused by the planner and the encoder.
    private struct TokenizedWord {
        let word: Word
        let tokens: [Int]
    }

    private func tokenize(_ words: [Word], tokenizer: Tokenizer) -> [TokenizedWord] {
        words.map { w in
            // Emoji-only words become a neutral token ("-") so the tokenizer doesn't emit garbled
            // byte-fallback tokens, which used to confuse the model into false detections around
            // emoji. The original text is kept on the `Word`, so masking is unaffected.
            let encodable = Self.isEmojiOnly(w.text) ? "-" : w.text
            return TokenizedWord(
                word: w, tokens: tokenizer.encode(text: encodable, addSpecialTokens: false))
        }
    }

    /// Fixed per-window overhead: the CLS token, one `<<ENT>>` plus its subwords for every label,
    /// the label/text separator and the trailing SEP. Cached because it depends only on the label
    /// set, which is immutable for a masker's lifetime — and it is ~67 tokens of a 768 budget, so
    /// it is worth knowing exactly rather than estimating.
    private var cachedPreambleTokens: Int?

    private func preambleTokenCount(tokenizer: Tokenizer) -> Int {
        if let cachedPreambleTokens { return cachedPreambleTokens }
        var n = 2   // CLS, and the separator between the labels and the text
        for label in piiLabels {
            n += 1  // <<ENT>>
            n += tokenizer.encode(text: label, addSpecialTokens: false).count
        }
        n += 1      // trailing SEP
        cachedPreambleTokens = n
        return n
    }

    /// Slice the words into windows that each fit `capacity` text tokens.
    ///
    /// Windows OVERLAP by `maxWidth - 1` words, and that number is not a guess: a GLiNER span is at
    /// most `maxWidth` words wide, so overlapping by one less than that puts every run of
    /// `maxWidth` consecutive words wholly inside at least one window. Without it a cut landing
    /// mid-entity would hide that entity from both neighbours — `alice@` in one window and
    /// `example.com` in the next, neither looking like an email. The duplicate hits the overlap
    /// produces cost nothing: ``mask(_:spans:)`` already folds overlapping spans into one.
    ///
    /// Returns nil when a SINGLE word does not fit a whole window — a long unbroken blob, which is
    /// usually a key or a token. No split rescues it, and it is precisely the kind of string that
    /// must not go out unmasked, so the pass fails closed rather than skipping it.
    private func planWindows(_ tokenized: [TokenizedWord], capacity: Int) -> [[TokenizedWord]]? {
        guard
            let ranges = Self.planWindowRanges(
                tokenCounts: tokenized.map { $0.tokens.count }, capacity: capacity,
                maxWidth: maxWidth)
        else { return nil }
        return ranges.map { Array(tokenized[$0]) }
    }

    /// The plan itself, as index ranges over the words.
    ///
    /// Separate from ``planWindows(_:capacity:)`` because the arithmetic needs nothing but each
    /// word's token count — no model, no tokenizer, no actor. That makes the cut positions, the
    /// overlap width and the termination argument testable with no weights on disk, which every
    /// other windowing test requires.
    static func planWindowRanges(tokenCounts: [Int], capacity: Int, maxWidth: Int)
        -> [Range<Int>]?
    {
        guard capacity > 0 else { return nil }
        var windows: [Range<Int>] = []
        var i = 0
        while i < tokenCounts.count {
            var used = 0
            var j = i
            while j < tokenCounts.count, used + tokenCounts[j] <= capacity {
                used += tokenCounts[j]
                j += 1
            }
            guard j > i else { return nil }
            windows.append(i..<j)
            if j >= tokenCounts.count { break }
            // Step back so the next window re-reads the tail. `overlap` can never reach `j - i`, so
            // `i` strictly increases and this terminates even on pathological token counts.
            let overlap = min(maxWidth - 1, j - i - 1)
            i = max(i + 1, j - overlap)
        }
        return windows
    }

    /// One inference pass over one window. Spans come back as character offsets into the ORIGINAL
    /// text, because each ``TokenizedWord`` carries the offsets ``splitWords(_:)`` measured — so
    /// there is no window-relative arithmetic to get wrong.
    private func runWindow(
        _ window: [TokenizedWord], index: Int, of total: Int,
        session sess: ORTSession, tokenizer tok: Tokenizer
    ) -> [(start: Int, end: Int, labelIdx: Int)]? {
        let started = Date()
        let numWords = window.count
        let (inputIds, wordsMask, textLen) = buildInputs(window, tokenizer: tok)
        let seqLen = inputIds.count

        // The correctness limit. GLiNER's position embeddings are relative, so an over-length
        // sequence throws nothing — it quietly detects less, reaching zero detections around 1,250
        // tokens, and an empty result is indistinguishable from "this text is clean". The planner
        // above should make this unreachable; it stays as the backstop, because the failure it
        // guards against is silent and every other failure here is loud.
        guard seqLen <= maxSequenceLength else {
            log.error(
                "privacy: window \(index, privacy: .public) is \(seqLen, privacy: .public) tokens, over the model's \(self.maxSequenceLength, privacy: .public)-token window — dropping rather than masking partially"
            )
            return nil
        }

        let (spanIdx, spanMask) = generateSpans(numWords: numWords, maxWidth: maxWidth)
        let numSpans = spanIdx.count

        do {
            let inputIdsData = inputIds.withUnsafeBufferPointer { Data(buffer: $0) }
            let wordsMaskData = wordsMask.withUnsafeBufferPointer { Data(buffer: $0) }
            let attMask = [Int64](repeating: 1, count: seqLen)
            let attMaskData = attMask.withUnsafeBufferPointer { Data(buffer: $0) }
            let textLenArr: [Int64] = [Int64(textLen)]
            let textLenData = textLenArr.withUnsafeBufferPointer { Data(buffer: $0) }
            let flatSpanIdx = spanIdx.flatMap { [$0.0, $0.1] }
            let spanIdxData = flatSpanIdx.withUnsafeBufferPointer { Data(buffer: $0) }
            let spanMaskI64 = spanMask.map { Int64($0 ? 1 : 0) }
            let spanMaskData = spanMaskI64.withUnsafeBufferPointer { Data(buffer: $0) }

            let idsT = try ORTValue(
                tensorData: NSMutableData(data: inputIdsData),
                elementType: .int64, shape: [1, NSNumber(value: seqLen)])
            let attT = try ORTValue(
                tensorData: NSMutableData(data: attMaskData),
                elementType: .int64, shape: [1, NSNumber(value: seqLen)])
            let wmT = try ORTValue(
                tensorData: NSMutableData(data: wordsMaskData),
                elementType: .int64, shape: [1, NSNumber(value: seqLen)])
            let tlT = try ORTValue(
                tensorData: NSMutableData(data: textLenData),
                elementType: .int64, shape: [1, 1])
            let siT = try ORTValue(
                tensorData: NSMutableData(data: spanIdxData),
                elementType: .int64, shape: [1, NSNumber(value: numSpans), 2])
            let smT = try ORTValue(
                tensorData: NSMutableData(data: spanMaskData),
                elementType: .int64, shape: [1, NSNumber(value: numSpans)])

            let outputs = try sess.run(
                withInputs: [
                    "input_ids": idsT,
                    "attention_mask": attT,
                    "words_mask": wmT,
                    "text_lengths": tlT,
                    "span_idx": siT,
                    "span_mask_int64": smT,
                ],
                outputNames: ["logits"],
                runOptions: nil)

            guard let logitsTensor = outputs["logits"] else {
                log.error("privacy: no logits output")
                return nil
            }

            let logitsData = try logitsTensor.tensorData() as Data
            let numLabels = piiLabels.count
            let cutoff = threshold
            let width = maxWidth

            // Post-process: sigmoid + threshold → entities
            let entities = logitsData.withUnsafeBytes {
                (ptr: UnsafeRawBufferPointer) -> [(
                    wordStart: Int, wordEnd: Int, labelIdx: Int, score: Float
                )] in
                let floats = ptr.bindMemory(to: Float.self)
                var results: [(wordStart: Int, wordEnd: Int, labelIdx: Int, score: Float)] = []
                for sw in 0..<numWords {
                    for w in 0..<width {
                        let ew = sw + w
                        guard ew < numWords else { break }
                        for li in 0..<numLabels {
                            let idx = (sw * width + w) * numLabels + li
                            guard idx < floats.count else { continue }
                            let logit = floats[idx]
                            let prob = 1.0 / (1.0 + exp(-logit))
                            if prob > cutoff {
                                results.append((sw, ew, li, prob))
                            }
                        }
                    }
                }
                return results
            }

            if logging.debug != nil {
                let labelsSummary =
                    entities
                    .reduce(into: [String: Int]()) {
                        $0[piiLabels[$1.labelIdx], default: 0] += 1
                    }
                    .map { "\($0.key):\($0.value)" }
                    .joined(separator: " ")
                logging.trace(
                    "privacy: window \(index + 1)/\(total): \(entities.count) entities [\(labelsSummary)] \(String(format: "%.0f", Date().timeIntervalSince(started) * 1000))ms words=\(numWords) seq=\(seqLen)"
                )
            }

            // Map window word indices → character positions in the original text, dropping spans
            // that cover only emoji-only words (false positives from the neutral-token
            // substitution in `tokenize`).
            return entities.compactMap { e -> (start: Int, end: Int, labelIdx: Int)? in
                let allEmoji = (e.wordStart...e.wordEnd).allSatisfy {
                    Self.isEmojiOnly(window[$0].word.text)
                }
                guard !allEmoji else { return nil }
                return (window[e.wordStart].word.charStart, window[e.wordEnd].word.charEnd, e.labelIdx)
            }
        } catch {
            log.error(
                "privacy: inference failed: \(error.localizedDescription, privacy: .public)"
            )
            return nil
        }
    }

    // MARK: - Probe

    /// A sentence carrying three unmistakable pieces of PII, used to decide whether a
    /// freshly-installed model can be trusted (``ModelRollback``). Deliberately plain and
    /// English-neutral in shape: the point is to catch a model that detects nothing, not to
    /// measure quality.
    public static let probeText =
        "Email me at jane.doe@example.com or call 415-555-0142, my IP is 192.168.4.21."

    /// The anchors a working model has to cover. Label-agnostic on purpose — what matters for the
    /// privacy gate is that the span gets masked at all, not which label it was filed under.
    public static let probeAnchors = ["jane.doe@example.com", "415-555-0142", "192.168.4.21"]

    public struct ProbeResult: Sendable, Equatable {
        public let found: [String]
        public let missed: [String]
        /// A load failure or an inference that threw — distinct from "ran and detected nothing",
        /// which is the same verdict but a different thing to log.
        public let ran: Bool
        /// One miss is model jitter; two is a model that isn't doing its job. A version that fails
        /// this leaves the fail-closed gate open, which is worse than a crash — it leaks silently.
        public var passes: Bool { ran && found.count >= 2 }

        public init(found: [String], missed: [String], ran: Bool) {
            self.found = found
            self.missed = missed
            self.ran = ran
        }
    }

    /// Run the probe against whatever this masker resolves to right now.
    ///
    /// The library ships this check but never calls it. Running it is the host's job: call it once
    /// after ``ModelInstaller/ensureLatest(force:skipping:onProgress:)`` reports new bytes, and
    /// treat ``ProbeResult/passes`` as the pass mark.
    public func probe() async -> ProbeResult {
        guard let (text, spans) = await detect(Self.probeText, userNameMasking: false) else {
            return ProbeResult(found: [], missed: Self.probeAnchors, ran: false)
        }
        let covered = spans.compactMap { span -> String? in
            guard let lo = text.index(text.startIndex, offsetBy: span.start, limitedBy: text.endIndex),
                  let hi = text.index(text.startIndex, offsetBy: span.end, limitedBy: text.endIndex),
                  lo < hi else { return nil }
            return String(text[lo..<hi])
        }
        let found = Self.probeAnchors.filter { anchor in covered.contains { $0.contains(anchor) } }
        return ProbeResult(
            found: found, missed: Self.probeAnchors.filter { !found.contains($0) }, ran: true)
    }

    // MARK: - User name masking

    private func maskUserName(_ text: String) -> String {
        UserNameMask.maskContext(
            text, firstName: firstName, lastName: lastName, senderHeader: senderHeader)
    }

    // MARK: - Preprocessing

    private struct Word {
        let text: String
        let charStart: Int
        let charEnd: Int
    }

    /// Split on whitespace, recording each word's Character offsets — the offsets every later stage
    /// (``trimSpan(_:start:end:)``, ``mask(_:spans:)``) measures in.
    ///
    /// The running `offset` is carried rather than re-measured with `distance(from: startIndex,)`
    /// per word. That call is O(offset) on a String, so re-measuring made this quadratic: a 100 KB
    /// paste — a log dump, a long thread — spent ~2.4s here before inference even started, and 250
    /// KB spent ~14s, blowing ``defaultMaskTimeout`` on a serialized actor and stalling every
    /// request queued behind it. Carrying the offset is the same arithmetic done once.
    private static func splitWords(_ text: String) -> [Word] {
        var words: [Word] = []
        var i = text.startIndex
        var offset = 0
        while i < text.endIndex {
            if text[i].isWhitespace {
                i = text.index(after: i)
                offset += 1
                continue
            }
            let start = i
            let startOffset = offset
            while i < text.endIndex && !text[i].isWhitespace {
                i = text.index(after: i)
                offset += 1
            }
            words.append(
                Word(
                    text: String(text[start..<i]),
                    charStart: startOffset,
                    charEnd: offset))
        }
        return words
    }

    /// Encode one window: CLS, the `<<ENT>> label` preamble, the separator, the window's words,
    /// then SEP. Words arrive already tokenized (see ``tokenize(_:tokenizer:)``) so a multi-window
    /// pass tokenizes the text once rather than once per window.
    ///
    /// The first subword of each word carries that word's 1-based index in `wordsMask`; every other
    /// position is 0. That mask is how the model maps its span indices back to whole words, so the
    /// numbering is window-relative and the caller translates it back through the window's `Word`s.
    private func buildInputs(
        _ window: [TokenizedWord], tokenizer: Tokenizer
    ) -> (inputIds: [Int64], wordsMask: [Int64], textLength: Int) {
        var ids: [Int64] = [clsTokenId]
        var wm: [Int64] = [0]

        for label in piiLabels {
            ids.append(entTokenId)
            wm.append(0)
            for t in tokenizer.encode(text: label, addSpecialTokens: false) {
                ids.append(Int64(t))
                wm.append(0)
            }
        }

        ids.append(entSepTokenId)
        wm.append(0)

        var wordIdx: Int64 = 1
        for tw in window {
            for (j, t) in tw.tokens.enumerated() {
                ids.append(Int64(t))
                wm.append(j == 0 ? wordIdx : 0)
            }
            wordIdx += 1
        }

        ids.append(sepTokenId)
        wm.append(0)

        return (ids, wm, window.count)
    }

    private func generateSpans(numWords: Int, maxWidth: Int)
        -> (spans: [(Int64, Int64)], mask: [Bool])
    {
        var spans: [(Int64, Int64)] = []
        var mask: [Bool] = []
        for i in 0..<numWords {
            for w in 0..<maxWidth {
                let end = i + w
                spans.append((Int64(i), Int64(end)))
                mask.append(end < numWords)
            }
        }
        return (spans, mask)
    }

    // MARK: - Masking

    /// Exclude protected marker words from detections. A marker-only detection is dropped;
    /// a detection crossing a marker is split, retaining the sensitive content on BOTH sides.
    /// Dropping the whole crossing span would leave detected secrets unmasked.
    ///
    /// Match whole whitespace-delimited words, as the detector does, so a marker substring
    /// inside a credential is never exempt. Trim the new pieces so the join's padding and
    /// adjacent punctuation remain outside the placeholders. Offsets remain Character counts.
    public static func dropProtectedSpans(
        _ text: String,
        _ spans: [(start: Int, end: Int, label: String)],
        protecting: Set<String>
    ) -> [(start: Int, end: Int, label: String)] {
        guard !protecting.isEmpty else { return spans }
        let markers = splitWords(text).filter { protecting.contains($0.text) }
        guard !markers.isEmpty else { return spans }
        return spans.flatMap { span -> [(start: Int, end: Int, label: String)] in
            let crossed = markers.filter { $0.charStart < span.end && $0.charEnd > span.start }
            guard !crossed.isEmpty else { return [span] }
            var pieces: [(start: Int, end: Int, label: String)] = []
            var start = span.start
            for marker in crossed {
                if let part = trimSpan(text, start: start, end: marker.charStart) {
                    pieces.append((part.start, part.end, span.label))
                }
                start = max(start, marker.charEnd)
            }
            if let part = trimSpan(text, start: start, end: span.end) {
                pieces.append((part.start, part.end, span.label))
            }
            return pieces
        }
    }

    /// Matches a placeholder token of the exact shape ``mask(_:spans:)`` emits (`[LABEL_<n>]`).
    /// Used by ``restore(_:with:)`` to strip any token the model garbled or dropped, so a
    /// redaction marker can never reach the user.
    private static let unresolvedTokenRegex = try? NSRegularExpression(
        pattern: "\\[[A-Z0-9_]+_[0-9]+\\]")

    /// Replace each detected PII span with a unique, reversible placeholder token (e.g.
    /// `[IP_ADDRESS_1]`) and return the masked text plus the map from token back to the original
    /// substring. Labelled rather than opaque `*****` so the model keeps the sentence's semantic
    /// shape, and numbered so the masking is reversible. Tokens are numbered left-to-right
    /// (reading order) but spliced right-to-left so earlier character indices stay valid.
    public static func mask(_ text: String, spans: [(start: Int, end: Int, label: String)])
        -> (text: String, restore: [String: String])
    {
        let sorted = spans.sorted { $0.start < $1.start }
        var merged: [(start: Int, end: Int, label: String)] = []
        for span in sorted {
            if let last = merged.last, span.start <= last.end {
                merged[merged.count - 1].end = max(last.end, span.end)
            } else {
                merged.append(span)
            }
        }

        // Assign a token per span in reading order so numbering is stable.
        var counters: [String: Int] = [:]
        let tokens: [String] = merged.map { span in
            let key = placeholderLabel(span.label)
            let n = (counters[key] ?? 0) + 1
            counters[key] = n
            return "[\(key)_\(n)]"
        }

        var restore: [String: String] = [:]
        var result = text
        for i in stride(from: merged.count - 1, through: 0, by: -1) {
            let span = merged[i]
            let lo =
                text.index(
                    text.startIndex, offsetBy: span.start,
                    limitedBy: text.endIndex) ?? text.endIndex
            let hi =
                text.index(
                    text.startIndex, offsetBy: span.end,
                    limitedBy: text.endIndex) ?? text.endIndex
            guard lo < hi else { continue }
            restore[tokens[i]] = String(text[lo..<hi])
            result.replaceSubrange(lo..<hi, with: tokens[i])
        }
        return (result, restore)
    }

    /// Put the original PII values back into a model response by swapping each placeholder token
    /// for the text it stood in for. Run locally on the model's output, so the remote model only
    /// ever saw the tokens. Any leftover token of our exact shape (the model garbled or dropped
    /// one) is stripped as a safety net, so a redaction marker can never reach the user.
    public static func restore(_ text: String, with map: [String: String]) -> String {
        guard !map.isEmpty else { return text }
        var result = text
        for (token, original) in map {
            result = result.replacingOccurrences(of: token, with: original)
        }
        if let re = unresolvedTokenRegex {
            let range = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(
                in: result, range: range, withTemplate: "")
        }
        return result
    }

    /// Opening punctuation/quotes that can precede an entity but are never part of it. Trimmed off
    /// a span's leading edge before masking.
    private static let spanTrimLeading: Set<Character> = [
        "(", "[", "{", "\"", "'", "\u{201C}", "\u{2018}", "\u{00AB}", "\u{00BF}", "\u{00A1}",
    ]

    /// Closing/sentence punctuation that can follow an entity but is never part of it. Trimmed off
    /// a span's trailing edge before masking — this is what keeps a flagged "ip?" from masking its
    /// "?" and producing "ip??".
    private static let spanTrimTrailing: Set<Character> = [
        ".", ",", "?", "!", ";", ":", "\u{2026}", ")", "]", "}", "\"", "'",
        "\u{201D}", "\u{2019}", "\u{00BB}",
    ]

    /// Shrink a detected character span inward past boundary whitespace and sentence punctuation,
    /// so the masked placeholder covers only the entity itself. Internal punctuation (the dots of
    /// an IP, the `@` of an email) is untouched — only the span's edges are trimmed. Offsets are
    /// Character counts into `text`. Returns nil when nothing but punctuation/whitespace remains.
    public static func trimSpan(_ text: String, start: Int, end: Int) -> (start: Int, end: Int)? {
        let n = text.count
        var lo = max(0, min(start, n))
        var hi = max(0, min(end, n))
        guard lo < hi else { return nil }
        func char(_ off: Int) -> Character {
            text[text.index(text.startIndex, offsetBy: off)]
        }
        while lo < hi {
            let c = char(lo)
            guard c.isWhitespace || spanTrimLeading.contains(c) else { break }
            lo += 1
        }
        while hi > lo {
            let c = char(hi - 1)
            guard c.isWhitespace || spanTrimTrailing.contains(c) else { break }
            hi -= 1
        }
        return lo < hi ? (lo, hi) : nil
    }

    private static func isEmojiOnly(_ text: String) -> Bool {
        !text.isEmpty && text.allSatisfy { char in
            char.unicodeScalars.contains { $0.properties.isEmoji && !$0.isASCII }
        }
    }

    /// Turn a PII label ("ip address") into a placeholder-safe key ("IP_ADDRESS"): uppercased,
    /// with any non-alphanumeric run as `_`.
    private static func placeholderLabel(_ label: String) -> String {
        String(label.uppercased().map { ($0.isLetter || $0.isNumber) ? $0 : "_" })
    }

    // MARK: - Model loading

    private func ensureLoaded() async -> Bool {
        // Keyed on the generation, not just `isLoaded`: a model rollback replaces the bytes under a
        // masker that already holds a session for them, and that session is precisely the one whose
        // masking was judged unsafe. Without this the reverting launch keeps sending context through
        // the model it just rejected — the leak the probe exists to prevent.
        if isLoaded, loadedGeneration == ModelStore.generation { return true }
        isLoaded = false

        // One directory, resolved once, for every file this load needs. Resolving per file would
        // let a model refresh landing mid-load pair one version's weights with another's tokenizer
        // — which throws nothing (the superseded version dir is kept on purpose) and silently masks
        // PII wrongly. Completeness is checked even for a pinned directory: `resolvedModelDirectory`
        // only ever returns a complete one, and a pinned path that is missing files must fail the
        // same way — "there is no model here", not "these bytes are not a model", which is the
        // distinction a rollback verdict rests on.
        guard let modelDir = modelDirectory ?? ModelStore.resolvedModelDirectory(),
              ModelInstaller.isCompleteModelDir(modelDir)
        else {
            let root = ModelStore.location.installRoot?.path ?? "<no install root configured>"
            log.error(
                "privacy: no complete GLiNER model — need \(ModelInstaller.requiredModelFiles.joined(separator: ", "), privacy: .public) in \(root, privacy: .public)/current"
            )
            return false
        }
        let modelPath = modelDir.appendingPathComponent("model.onnx").path

        logging.trace("privacy: loading GLiNER model from \(modelPath)")
        let start = Date()

        do {
            let env = try ORTEnv(loggingLevel: .warning)
            let opts = try ORTSessionOptions()
            try opts.setIntraOpNumThreads(2)
            session = try ORTSession(
                env: env, modelPath: modelPath, sessionOptions: opts)

            let hub = HubApi()
            let tokData = try hub.configuration(
                fileURL: modelDir.appendingPathComponent("tokenizer.json"))
            let tokConfig = try hub.configuration(
                fileURL: modelDir.appendingPathComponent("tokenizer_config.json"))
            tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: tokConfig, tokenizerData: tokData, strict: false)
        } catch {
            log.error(
                "privacy: failed to load: \(error.localizedDescription, privacy: .public)"
            )
            // Files all present, model still unusable — the fail-closed gate will now hold every
            // request. Nothing else can detect this, so record it: without this the host looks
            // healthy while doing nothing, with no affordance to recover.
            ModelStore.markUnusable()
            return false
        }

        let elapsed = Date().timeIntervalSince(start)
        logging.trace("privacy: GLiNER ready in \(String(format: "%.1f", elapsed))s")
        isLoaded = true
        loadedGeneration = ModelStore.generation
        ModelStore.markLoaded()
        return true
    }

    // MARK: - Model state (forwards to the process-wide store)

    public typealias ModelState = ModelStore.State

    public static var modelState: ModelState {
        get { ModelStore.state }
        set { ModelStore.state = newValue }
    }

    public static var onModelStateChange: (@Sendable () -> Void)? {
        get { ModelStore.onStateChange }
        set { ModelStore.onStateChange = newValue }
    }

    /// The bytes on disk changed, so whatever the loader last concluded no longer describes them —
    /// and neither does any session already built from them.
    public static func invalidateModelState() { ModelStore.invalidate() }

    /// Whether the on-device model is present. See ``ModelStore/modelIsInstalled()``.
    public static func modelIsInstalled() -> Bool { ModelStore.modelIsInstalled() }

    /// The one directory every model file is loaded from. See
    /// ``ModelStore/resolvedModelDirectory()``.
    public static func resolvedModelDirectory() -> URL? { ModelStore.resolvedModelDirectory() }

    /// Resolved `model.onnx` path, for diagnostics and tests.
    public static func resolvedModelPath() -> String? {
        ModelStore.resolvedModelDirectory()?.appendingPathComponent("model.onnx").path
    }
}

import Foundation
import Testing

@testable import PIIMasker

/// The tier that needs real weights: tokenization → ONNX inference → masking, plus the reload
/// behaviour a rollback depends on.
///
/// Opt in by pointing `PII_MASKER_MODEL_DIR` at a directory holding `model.onnx`,
/// `tokenizer.json` and `tokenizer_config.json`. Without it every test here is skipped, so
/// `swift test` on a machine with no model is green and means it.
///
///     PII_MASKER_MODEL_DIR=/path/to/gliner swift test
///
/// Serialized: they reconfigure the process-wide `ModelStore`.
@Suite("Model tier", .serialized, .enabled(if: ModelTier.isAvailable))
struct ModelTierTests {

    // MARK: - Detection

    /// Each label category detects the entity it names. The assertion is deliberately "something
    /// was masked", not "masked under this exact label": the model sometimes files a span under a
    /// neighbouring label (a PIN as a password), and from a privacy standpoint both are the same
    /// answer — the content did not go out in the clear.
    @Test(arguments: ModelTier.detectionCases)
    func detectsEachLabelCategory(_ testCase: ModelTier.DetectionCase) async {
        let masker = ModelTier.masker()
        var anyMasked = false
        var reports: [String] = []
        for phrase in testCase.phrases {
            guard let result = await masker.sanitize(phrase) else {
                reports.append("sanitize returned nil for: \(phrase)")
                continue
            }
            if !result.restore.isEmpty {
                anyMasked = true
                if result.restore.keys.contains(where: { $0.contains(testCase.label) }) { break }
                reports.append("\(testCase.label) → \(result.restore.keys.sorted().joined(separator: ", "))")
            } else {
                reports.append("nothing detected in: \(phrase)")
            }
        }
        #expect(anyMasked, "\(testCase.label): \(reports.joined(separator: " | "))")
    }

    /// The current user's name is masked — by the regex pass, or by the model detecting it. Either
    /// way the name is gone.
    @Test func userNameIsMasked() async {
        let masker = ModelTier.masker(firstName: "Casey", lastName: "Nakamura")
        let result = await masker.sanitize("Hello Casey Nakamura, how are you?")
        #expect(result != nil)
        #expect(result?.text.contains("Casey Nakamura") == false, "\(result?.text ?? "nil")")
    }

    /// Clean text passes through unmasked. A masker that flags everything is useless in a different
    /// way from one that flags nothing, and only this catches it.
    @Test(arguments: [
        "The weather is beautiful today",
        "Can we schedule a meeting for tomorrow",
        "I enjoyed the movie last night",
        "Please review the quarterly report",
        "The new feature looks great",
    ])
    func cleanTextIsNotMasked(_ phrase: String) async {
        let result = await ModelTier.masker().sanitize(phrase)
        #expect(result?.restore.isEmpty == true, "unexpected mask: \(result?.text ?? "nil")")
    }

    /// Several PII types in one string all mask, sharing one token map.
    @Test func mixedEntitiesAllMask() async {
        let mixed = "Call 555-123-4567 or email bob@test.com about the password reset"
        let result = await ModelTier.masker().sanitize(mixed)
        #expect(result != nil)
        #expect((result?.restore.count ?? 0) >= 2,
                "only \(result?.restore.count ?? 0): \(result?.restore.keys.joined(separator: ", ") ?? "")")
    }

    // MARK: - The probe

    /// The shipped model passes its own health probe. This is the gate every published model is
    /// judged by, so a false negative here would roll back good releases.
    @Test func theInstalledModelPassesItsProbe() async {
        let result = await ModelTier.masker().probe()
        #expect(result.ran)
        #expect(result.passes, "missed: \(result.missed.joined(separator: ", "))")
    }

    // MARK: - The batch gate, end to end

    /// The whole gate: several fields, one inference pass, one shared token map, every field
    /// recoverable in order.
    @Test func maskFieldsBatchesAndRoundTrips() async {
        let values = [
            "SENDER|MESSAGE\nSam|my email is dana@example.com\n",
            "reply to dana@example.com and 415-555-0142",
            "the draft ",
        ]
        let masker = ModelTier.masker()
        guard let (masked, restore) = await masker.maskFields(
            values, timeout: PrivacyFilter.defaultMaskTimeout)
        else {
            Issue.record("maskFields returned nil with a model installed")
            return
        }
        #expect(masked.count == values.count)
        // The trailing space on the last field is why the join is guarded.
        #expect(masked[2] == "the draft ", "\(masked[2])")
        #expect(!restore.isEmpty, "nothing was detected in text that plainly carries PII")
        for (i, field) in masked.enumerated() {
            #expect(PrivacyFilter.restore(field, with: restore) == values[i], "field \(i): \(field)")
        }
        // Whatever was masked is really gone from what would be sent.
        for original in restore.values {
            #expect(!masked.joined(separator: "\n").contains(original), "\(original) survived masking")
        }
    }

    /// A timeout the pass cannot meet drops the request rather than sending anything.
    @Test func anImpossibleTimeoutDrops() async {
        let out = await ModelTier.masker().maskFields(["dana@example.com"], timeout: 0)
        #expect(out == nil)
    }

    // MARK: - Long input, windowed

    /// The regression this whole mechanism exists for.
    ///
    /// GLiNER's position embeddings are relative, so an over-length input throws nothing — it
    /// quietly detects less, and past roughly 1,250 tokens it detects nothing at all. Measured on
    /// these weights before windowing: at 1,269 tokens the model returned ZERO entities and
    /// `sanitize` handed back the input unchanged with an empty restore map, which `maskFields`
    /// reported as SUCCESS. The caller would have sent raw PII believing it was masked.
    ///
    /// The anchor is a phone number, not an email, and the filler is varied prose rather than one
    /// sentence repeated. That is deliberate: this asserts what WINDOWING owes you, and picking an
    /// entity the model misses anyway — or degenerate filler it detects nothing in — would test
    /// recall instead. Recall has its own tests above.
    @Test func longInputIsWindowedAndStillFindsPII() async {
        let anchor = "415-555-0142"
        // ~1,200 words: well past one 768-token window, with the PII at the END, which is exactly
        // where detection died first.
        let long = "\(ModelTier.prose(1_150)) please call me on \(anchor) tomorrow."

        guard let out = await ModelTier.masker().sanitize(long) else {
            Issue.record("a long input should be windowed, not dropped")
            return
        }
        #expect(!out.text.contains(anchor), "the phone number survived masking: it leaked")
        // …and the offsets survived the windowing, or the round-trip would not close.
        #expect(PrivacyFilter.restore(out.text, with: out.restore) == long)
    }

    /// The batch gate over input beyond one window: the join, the split and the shared restore map
    /// all have to survive being masked across several inference passes.
    ///
    /// This asserts STRUCTURE, not detection. Whether the model finds a given entity in a
    /// thousand words of filler is recall, and it is genuinely unstable at that length — measured:
    /// the identical field masks its phone number when passed alone, and misses it when 29
    /// characters of unrelated text are appended. Windowing does not change that either way, so
    /// asserting it here would buy a flaky test and no coverage. Detection across a window boundary
    /// is pinned by `anEntityOnAWindowBoundaryIsStillFound`, on content where it is stable.
    @Test func maskFieldsHandlesInputBeyondOneWindow() async {
        let fields = [
            "\(ModelTier.prose(900)) reach me on 415-555-0142 after five. \(ModelTier.prose(150))",
            "the draft ",
        ]
        guard let (masked, restore) = await ModelTier.masker().maskFields(fields, timeout: 60)
        else {
            Issue.record("maskFields should window a long input rather than drop it")
            return
        }
        #expect(masked.count == fields.count)
        #expect(masked[1] == "the draft ", "the guarded trailing space was lost: \(masked[1])")
        // Every field recovers exactly — the guard-integrity split held and no span offset was
        // mistranslated between windows.
        for (i, field) in masked.enumerated() {
            #expect(PrivacyFilter.restore(field, with: restore) == fields[i], "field \(i)")
        }
        // Whatever WAS masked is genuinely gone from what would be sent.
        for original in restore.values {
            #expect(!masked.joined(separator: "\n").contains(original), "\(original) survived masking")
        }
    }

    /// An entity landing ON a window boundary must still be found. Windows overlap by
    /// `maxWidth - 1` words precisely so every run of `maxWidth` words sits wholly inside at least
    /// one of them — without that overlap a cut through `415-555-0142` hides it from both
    /// neighbours, and the leak returns in a form no length check would catch.
    ///
    /// Sweeping the offset walks the anchor across the first boundary rather than guessing where it
    /// falls. Every position must mask AND round-trip; a wrong offset translation would show up as
    /// the latter failing even when the former passes.
    @Test func anEntityOnAWindowBoundaryIsStillFound() async {
        let masker = ModelTier.masker()
        let anchor = "415-555-0142"
        var leaked: [Int] = []
        var brokenRoundTrip: [Int] = []
        for filler in stride(from: 580, through: 720, by: 20) {
            let text =
                "\(ModelTier.prose(filler)) please call me on \(anchor) tomorrow. "
                + ModelTier.prose(120)
            guard let out = await masker.sanitize(text) else {
                leaked.append(filler)
                continue
            }
            if out.text.contains(anchor) { leaked.append(filler) }
            if PrivacyFilter.restore(out.text, with: out.restore) != text {
                brokenRoundTrip.append(filler)
            }
        }
        #expect(leaked.isEmpty, "the anchor leaked at filler widths \(leaked)")
        #expect(brokenRoundTrip.isEmpty, "offsets were mistranslated at \(brokenRoundTrip)")
    }

    // MARK: - The two ceilings

    /// `maxInputTokens` is the COST ceiling and it is the caller's to set. Over it the pass drops,
    /// rather than spending unbounded inference on a serialized actor with every queued request
    /// waiting behind it.
    @Test func theInputBudgetIsConfigurableAndDrops() async {
        let long = "\(ModelTier.prose(1_150)) call 415-555-0142."

        // The default budget (2,000 tokens) accommodates it.
        #expect(await ModelTier.masker().sanitize(long) != nil)

        // A budget below the input refuses it — fail-closed, the same nil as any other failure.
        let stingy = PrivacyFilter(
            config: MaskerConfig(maxInputTokens: 300), modelDirectory: ModelTier.modelDir)
        #expect(await stingy.sanitize(long) == nil, "over the token budget must drop")

        // …and it is a ceiling, not a blanket refusal: short text still masks under the same one.
        let ok = await stingy.sanitize("contact alice@example.com or call 415-555-0142")
        #expect(ok?.restore.isEmpty == false, "\(ok?.text ?? "nil")")
    }

    /// `maxSequenceLength` is the CORRECTNESS ceiling and it comes from the pin, so different
    /// weights get their own window size. Lowering it makes windows smaller and more numerous — it
    /// must not start dropping, because splitting is exactly what it is for.
    @Test func theWindowSizeComesFromThePin() async {
        let text = "\(ModelTier.prose(300)) call 415-555-0142."

        func pin(maxSequenceLength: Int) -> ModelPin {
            ModelPin(
                version: ModelPin.current.version, sourceURL: ModelPin.current.sourceURL,
                archiveSHA256: ModelPin.current.archiveSHA256,
                weightsSHA256: ModelPin.current.weightsSHA256, bytes: ModelPin.current.bytes,
                maxWidth: ModelPin.current.maxWidth, maxSequenceLength: maxSequenceLength)
        }

        // A window barely wider than the ~67-token label preamble forces many tiny windows.
        let narrow = PrivacyFilter(
            pin: pin(maxSequenceLength: 140), modelDirectory: ModelTier.modelDir)
        guard let out = await narrow.sanitize(text) else {
            Issue.record("a narrow window should split further, not drop")
            return
        }
        #expect(PrivacyFilter.restore(out.text, with: out.restore) == text,
                "narrow windows corrupted the round-trip")

        // A window too small to hold the preamble at all has no capacity left for text, so no split
        // can help and the pass fails closed.
        let impossible = PrivacyFilter(
            pin: pin(maxSequenceLength: 10), modelDirectory: ModelTier.modelDir)
        #expect(await impossible.sanitize(text) == nil, "an unusable window must fail closed")
    }

    /// A single unbroken blob longer than one window cannot be split — no cut helps — so the pass
    /// fails closed. That shape is usually a key or a token, exactly what must never go out
    /// unmasked, so dropping beats skipping the word and masking around it.
    @Test func anUnsplittableWordFailsClosed() async {
        let blob = String(repeating: "A1b2C3d4", count: 400)   // 3,200 chars, no whitespace
        let masker = PrivacyFilter(
            config: MaskerConfig(maxInputTokens: 100_000),      // not the budget under test
            modelDirectory: ModelTier.modelDir)
        #expect(await masker.sanitize("here is the key \(blob)") == nil)
    }

    // MARK: - Reload on invalidation

    /// A masker that already loaded a model must not keep using it once the bytes are replaced. On
    /// the launch that reverts, the session in memory is precisely the one whose masking was judged
    /// unsafe — it has to be dropped, not reused. This is what `ModelStore.generation` is for.
    @Test func aMaskerReloadsAfterInvalidation() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: ModelTier.modelDir!, to: root.appendingPathComponent("good"))
        // A "model" whose files are all present and correctly named, but which is not a model.
        _ = try ModelStoreTests.completeDir(root.appendingPathComponent("dud"), marker: "not an onnx graph")
        try ModelInstaller.pointCurrent(at: "good", in: root)

        let saved = ModelStore.location
        defer { ModelStore.configure(saved); ModelStore.invalidate() }
        ModelStore.configure(installRoot: root)

        let masker = PrivacyFilter(logging: .silent)
        #expect(await masker.probe().passes, "the masker should start on a model that works")

        try ModelInstaller.pointCurrent(at: "dud", in: root)
        ModelStore.invalidate()
        let afterSwap = await masker.probe()
        #expect(!afterSwap.ran, "the same masker must reload, not serve the replaced model")

        try ModelInstaller.pointCurrent(at: "good", in: root)
        ModelStore.invalidate()
        #expect(await masker.probe().passes, "and pick the restored model back up")
        _ = trash
    }

    /// A refresh landing mid-load must not pair one version's weights with another's tokenizer:
    /// that throws nothing and silently masks wrongly, so no other signal would catch it. The
    /// guarantee is that a load resolves its directory once and reads every file from it — asserted
    /// by moving `current` out from under a masker pinned to a concrete directory.
    @Test func aPinnedLoadIgnoresALaterCurrentFlip() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: ModelTier.modelDir!, to: root.appendingPathComponent("good"))
        _ = try ModelStoreTests.completeDir(root.appendingPathComponent("dud"), marker: "not an onnx graph")
        try ModelInstaller.pointCurrent(at: "good", in: root)

        let saved = ModelStore.location
        defer { ModelStore.configure(saved); ModelStore.invalidate() }
        ModelStore.configure(installRoot: root)

        let pinned = PrivacyFilter(modelDirectory: root.appendingPathComponent("good"))
        try ModelInstaller.pointCurrent(at: "dud", in: root)
        ModelStore.invalidate()

        #expect(await pinned.probe().passes,
                "a load must read every file from the directory it resolved, not from `current`")
        // …and the superseded version is still on disk for a load already reading it.
        #expect(ModelInstaller.isCompleteModelDir(root.appendingPathComponent("good")))
        _ = trash
    }

    /// `generation` is `nonisolated(unsafe)` on the argument that a lost race costs one extra
    /// reload. That is only true if concurrent loads racing an invalidation stay well-formed — a
    /// torn read handing back a half-built session would be a silent privacy failure.
    @Test func concurrentLoadsRacingAnInvalidationStayCoherent() async throws {
        let root = TestPublish.scratch(); let trash = Trash(root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: ModelTier.modelDir!, to: root.appendingPathComponent("good"))
        _ = try ModelStoreTests.completeDir(root.appendingPathComponent("dud"), marker: "not an onnx graph")
        try ModelInstaller.pointCurrent(at: "good", in: root)

        let saved = ModelStore.location
        defer { ModelStore.configure(saved); ModelStore.invalidate() }
        ModelStore.configure(installRoot: root)

        let flips = Task {
            for v in ["dud", "good", "dud", "good"] {
                try? ModelInstaller.pointCurrent(at: v, in: root)
                ModelStore.invalidate()
                try? await Task.sleep(nanoseconds: 120_000_000)
            }
        }
        var coherent = true
        await withTaskGroup(of: PrivacyFilter.ProbeResult.self) { group in
            for _ in 0..<3 {
                group.addTask { await PrivacyFilter().probe() }
            }
            for await r in group {
                // Either it loaded a whole working model, or it loaded nothing. "Ran but found
                // nothing" is the shape a mixed or half-built session would take.
                if r.ran && r.found.isEmpty { coherent = false }
            }
        }
        await flips.value
        #expect(coherent)
        _ = trash
    }
}

/// Opt-in plumbing for the model tier.
enum ModelTier {
    /// A directory holding the three files the loader opens.
    static var modelDir: URL? {
        guard let path = ProcessInfo.processInfo.environment["PII_MASKER_MODEL_DIR"],
              !path.isEmpty
        else { return nil }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        return ModelInstaller.isCompleteModelDir(url) ? url : nil
    }

    static var isAvailable: Bool { modelDir != nil }

    /// Varied prose filler for the windowing tests. Deliberately several different sentences
    /// rather than one repeated: the model detects almost nothing in degenerate repetition, so
    /// repeated filler would make a windowing test fail for recall reasons and vice versa.
    static func prose(_ words: Int) -> String {
        let pool = [
            "The quarterly report needs review before Thursday's leadership sync.",
            "I think we should postpone the launch until the metrics stabilise.",
            "Could you take a look at the draft when you get a chance?",
            "There were a few concerns raised about timeline and staffing.",
            "Let me know if the revised numbers change your recommendation.",
            "We agreed to revisit the pricing model after the pilot ends.",
        ]
        var out: [String] = []
        var i = 0
        while out.count < words {
            out.append(contentsOf: pool[i % pool.count].split(separator: " ").map(String.init))
            i += 1
        }
        return out.prefix(words).joined(separator: " ")
    }

    static func masker(firstName: String = "", lastName: String = "") -> PrivacyFilter {
        PrivacyFilter(
            firstName: firstName, lastName: lastName,
            modelDirectory: modelDir)
    }

    struct DetectionCase: Sendable, CustomStringConvertible {
        let label: String
        let phrases: [String]
        var description: String { label }
    }

    static let detectionCases: [DetectionCase] = [
        DetectionCase(label: "PHONE_NUMBER", phrases: [
            "Please call me at 555-867-5309 tomorrow",
            "My number is +1 (212) 555-0198",
            "Reach me on 07700 900461",
        ]),
        DetectionCase(label: "EMAIL_ADDRESS", phrases: [
            "Send it to alice.jones@company.com please",
            "Contact support@example.org for help",
            "Email me at bob123@gmail.com",
        ]),
        DetectionCase(label: "CREDIT_CARD_NUMBER", phrases: [
            "My card is 4111-1111-1111-1111",
            "Use card number 5500 0000 0000 0004",
        ]),
        DetectionCase(label: "ADDRESS", phrases: [
            "I live at 742 Evergreen Terrace, Springfield",
            "Ship to 1600 Pennsylvania Avenue, Washington DC",
        ]),
        DetectionCase(label: "SOCIAL_SECURITY_NUMBER", phrases: [
            "My SSN is 123-45-6789",
            "Social security number 987-65-4321",
        ]),
        DetectionCase(label: "DATE_OF_BIRTH", phrases: [
            "My date of birth is March 15, 1990",
            "Date of birth: 01/15/1985",
            "DOB: 1990-03-15",
            "She was born on 12/25/1988",
        ]),
        DetectionCase(label: "BANK_ACCOUNT_NUMBER", phrases: [
            "My bank account number is 12345678901234",
            "Wire to account 9876543210 at Chase",
        ]),
        DetectionCase(label: "PASSWORD", phrases: [
            "My password is hunter2",
            "The login credentials are admin / P@ssw0rd123",
            "Password: xK9#mQ2!vL7",
        ]),
        DetectionCase(label: "PIN_CODE", phrases: [
            "My PIN code is 4829",
            "Enter PIN: 13334 to proceed",
            "The ATM pin is 9021",
        ]),
        DetectionCase(label: "IP_ADDRESS", phrases: [
            "The server IP is 192.168.1.100",
            "Connect to 10.0.0.1 on port 443",
            "Blocked IP address 203.0.113.42",
        ]),
        DetectionCase(label: "DOLLAR_AMOUNT", phrases: [
            "The total is $250,000",
            "I owe $15,000 in bills",
            "Paid $99.99 for the subscription",
        ]),
        DetectionCase(label: "PASSPORT_NUMBER", phrases: [
            "My passport number is AB1234567",
            "Passport: X12345678 issued in London",
        ]),
        DetectionCase(label: "DRIVER_LICENSE_NUMBER", phrases: [
            "Driver license number D123-4567-8901",
            "My driver license is S550-2400-1234",
        ]),
        DetectionCase(label: "TAX_ID", phrases: [
            "My tax id is 12-3456789",
            "EIN / tax id: 98-7654321",
        ]),
        DetectionCase(label: "API_KEY", phrases: [
            "The API key is sk-abc123def456ghi789",
            "Set your api key: AKIAIOSFODNN7EXAMPLE",
        ]),
        DetectionCase(label: "ACCESS_TOKEN", phrases: [
            "Bearer access token eyJhbGciOiJIUzI1NiJ9.abc.xyz",
            "Use this access token: ghp_xxxxxxxxxxxxxxxxxxxx",
        ]),
        DetectionCase(label: "SECRET_KEY", phrases: [
            "AWS secret key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "The secret key is 0x4a2b3c4d5e6f7890",
        ]),
    ]
}

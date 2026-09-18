# pii-masker

Find PII in text on-device and replace it with **reversible** placeholders, so text can be sent to
a remote model and the real values swapped back into the answer locally.

The masking is reversible, and that is the whole design. A detected span becomes
`[EMAIL_ADDRESS_1]`, not `*****`. So a false positive costs you a placeholder you restore, not a
redaction marker leaking into text a person reads — and the remote model still sees a labelled
token it can reason around instead of a hole in the sentence.

Detection is [GLiNER](https://github.com/urchade/GLiNER) (ONNX INT8) through ONNX Runtime. Nothing
leaves the machine except the model download.

macOS 14+. MIT.

## Quick start

Mask something on the command line first, before wiring it into anything.

**1. Build.**

```sh
git clone https://github.com/BlinkWrite/pii-masker
cd pii-masker
swift build
```

**2. Download the model** (~137 MB).

```sh
scripts/fetch-model.sh
```

It reads the URL and both hashes out of [`model.json`](../model.json), checks the archive, unpacks it
into `./model`, and checks the unpacked weights — the second check *after* unpacking, which is the
whole point of having two. See [Getting the model](#getting-the-model). Re-running it when the
model is already there does nothing. In an app you do none of this by hand: `ModelInstaller`
performs the same fetch, the same two checks, and an atomic swap.

**3. Mask something.**

```sh
swift run pii-mask --model ./model --show-map <<< "Card 4111 1111 1111 1111, SSN 123-45-6789, I live at 42 Elm Street."
```

```
Card [CREDIT_CARD_NUMBER_1], SSN [SOCIAL_SECURITY_NUMBER_1], I live at [ADDRESS_1].

  [ADDRESS_1] ← 42 Elm Street
  [CREDIT_CARD_NUMBER_1] ← 4111 1111 1111 1111
  [SOCIAL_SECURITY_NUMBER_1] ← 123-45-6789
```

**4. Run the tests.**

```sh
swift test                                # no model, no network
PII_MASKER_MODEL_DIR=./model swift test   # …plus the tier that needs weights
```

Then read [Example](#example) for the library, and [The CLI](#the-cli) for the rest of the flags.

## Add it to your package

```swift
.package(url: "https://github.com/BlinkWrite/pii-masker", from: "0.1.0")
```

## Getting the model

The model version is **pinned in source** — there is no manifest and no runtime version
negotiation. `ModelPin.current` names an immutable URL, the SHA-256 of the archive, and the SHA-256
of the unpacked `model.onnx`; the installer checks both, the second one *after* unpacking and
before the model becomes `current`.

That is on purpose. A hosted manifest's hash makes the weights host untrusted, but whoever serves
the manifest sets that hash — so a manifest relocates trust rather than removing it. A value
compiled into the library has no host. The cost is that a new model means a new library version and
a release of your app, which for a library is the normal contract.

The weights live at
[`blinkwrite-ai/gliner-small-pii-onnx-int8`](https://huggingface.co/blinkwrite-ai/gliner-small-pii-onnx-int8)
on Hugging Face.

**Pointing it at your own mirror.** You are not tied to that host. The hashes decide correctness,
not the URL, so a mirror, a corporate proxy or a `file://` path all give the same guarantee — the
bytes still have to be the ones this library was built against, wherever they came from:

```swift
let mirrored = ModelPin.current.withSourceURL(
    URL(string: "https://mirror.internal/gliner-pii-2026.08.1.tar.gz")!)

try await ModelInstaller(pin: mirrored, installRoot: root).ensureLatest()
```

`withSourceURL` carries both hashes over unchanged, which is the point of it. Overriding those too
means trusting your own host — the thing pinning exists to avoid — so it needs the full
`ModelPin(...)` initializer, where it reads as the deliberate act it is.

## Example

```swift
import PIIMasker

let root = URL.applicationSupportDirectory.appending(path: "MyApp/models")
ModelStore.configure(installRoot: root)

let outcome = try await ModelInstaller(installRoot: root).ensureLatest()
if outcome.installed {
    // New bytes: prove them before trusting them. See "Probe after an install".
    let result = await PrivacyFilter(modelDirectory: outcome.directory).probe()
    guard result.passes else { /* do not use this model */ return }
}

// `maskFields` does NOT mask the user's own name — see below. Apply it per field first, with the
// scope each field deserves.
let conversation = UserNameMask.mask(
    rawConversation, firstName: "Alex", lastName: "Rivera", scope: .givenNameToo)
let draft = UserNameMask.mask(
    rawDraft, firstName: "Alex", lastName: "Rivera", scope: .fullNameOnly)

let masker = PrivacyFilter()
guard let (masked, restore) = await masker.maskFields(
    [conversation, draft], timeout: PrivacyFilter.defaultMaskTimeout)
else { return }                              // ← the only correct response to nil is to send nothing

let answer = try await myRemoteModel.complete(masked[0], masked[1])
let final = PrivacyFilter.restore(answer, with: restore)
```

`maskFields` takes several strings and masks them in **one** inference pass, so they share a single
token map. Separate passes restart the counter, and a later field's `[PERSON_1]` would then collide
with an earlier one on restore.

**`maskFields` does not mask names, and the initializer's `firstName`/`lastName` do nothing for
it.** Once the fields are joined it cannot tell them apart, and the two scopes want different rules:
a wrong match in a draft lands in text the user reads back, while conversation text they don't own
wants the given name masked too. So the choice stays with the caller who knows which field is which
— call `UserNameMask` per field first, as above. Pass `firstName`/`lastName` to `PrivacyFilter` only
when you are calling `sanitize` on a single string, which runs the `[USER]` pass itself.

## How it works

1. Replace the authenticated user's own name with `[USER]` (regex, before the model). `sanitize`
   does this itself; `maskFields` leaves it to the caller, as above.
2. Split into whitespace words and tokenize, with the label set prefixed as GLiNER expects.
3. Split into overlapping windows if the text exceeds the model's 768-token limit, so no single
   pass runs past what the weights can actually read.
4. Run ONNX inference per window; sigmoid the logits and keep spans above the threshold. Every
   window reports character offsets into the same original text, so the windows never surface.
5. Trim each span's edges past punctuation, so a flagged `ip?` masks `ip` and leaves the `?`.
6. Merge overlaps, then splice right-to-left into numbered placeholders (`[IP_ADDRESS_1]`).
7. `restore` swaps the originals back, and strips any placeholder-shaped leftover the remote model
   garbled — so a redaction marker can never reach the user.

## Probe after an install

**The library ships a health check but never calls it. Running it is your job.**

`PrivacyFilter.probe()` masks a sentence carrying three unmistakable pieces of PII and reports how
many came back masked. Call it once whenever `ensureLatest()` reports `installed == true`, and
treat two-of-three (`result.passes`) as the pass mark — one miss is model jitter, two is a model
that is not doing its job.

This matters more than it looks. A bad model does not crash. Either it fails to load, and the
fail-closed gate holds every request while your app looks alive; or it loads and quietly detects
less, and the gate stops closing at all. Nothing else in this library would notice either.

`ModelRollback` builds on that: arm probation when you install over a working version, probe, and
revert the `current` symlink if the probe fails. Superseded version directories are deliberately
never deleted, which is what makes the revert a single atomic `rename()`.

## Testing

```sh
swift test                                          # no model, no network
PII_MASKER_MODEL_DIR=/path/to/gliner swift test     # …plus the tier that needs weights
```

The default run is deliberately model-free so a green result on a fresh checkout means something.
The model tier covers all 17 label categories, the probe, and the reload behaviour a rollback
depends on; it is skipped without `PII_MASKER_MODEL_DIR`.

## The CLI

`pii-mask` pipes text through the masker, so you can see what it catches without writing any Swift.
It reads stdin and writes the masked text to stdout.

```sh
swift run pii-mask --model /path/to/gliner --show-map <<< "email me at a@b.com"
```

| Option | |
|---|---|
| `--model DIR` | Directory holding `model.onnx`, `tokenizer.json` and `tokenizer_config.json`. Defaults to `$PII_MASKER_MODEL_DIR`. |
| `--show-map` | Also print each placeholder and the text it replaced, on stderr. |
| `--threshold N` | Score floor for keeping a span (default `0.1`). Lower catches more, and false-positives more. |
| `--timeout N` | Seconds the pass may take (default `30`). |

Exit status `0` masked, `1` bad usage or no model, `2` the masker produced nothing — the case where
a caller must send nothing. Text longer than the model's window is split into several passes rather
than dropped; only input over `maxInputTokens` exits `2`. See [Limitations](#limitations).

The restore map prints only on request. The point of the tool is to show what would leave the
machine, and dumping the originals beside it undermines reading that at a glance.

To put it on your `PATH` instead of going through `swift run`:

```sh
swift build -c release && cp .build/release/pii-mask /usr/local/bin/
```

One thing the CLI cannot do: mask your own name. The `[USER]` substitution needs the `firstName`
and `lastName` you hand `PrivacyFilter`, and there is no flag for them.

## Configuration

`MaskerConfig` carries the two inference knobs:

- **`labels`** — fed to GLiNER at inference time. **Order is the output index space**: the model
  returns an index into this exact array, so the array used for inference must be the one used to
  read the result back. And a label is not just a filter, it *names the placeholder* —
  `"email address"` becomes `[EMAIL_ADDRESS_1]`. Rename one and you change the token text the
  remote model sees, and invalidate any restore map built before the change.
- **`threshold`** — the score floor, default `0.1`. Raise it and PII slips through permanently;
  lower it and legitimate text becomes a placeholder, which `restore` puts back — so a false
  positive costs prompt noise, not data. The two errors are not equally bad, and the floor sits
  below the middle for that reason.

  The default came from measurement, not taste. Eight planted entities, ten PII-free control texts:

  | threshold | bare sentence | +300 words of prose | false positives |
  |---|---|---|---|
  | `0.2` | 8/8 | 5/8 | 1 span, short text only |
  | **`0.1`** | 8/8 | 6/8 | none at 300 words |
  | `0.05` | 8/8 | **8/8** | 5 short / 1 at 300 words |

  Recall depends far more on how much text surrounds the PII than on the threshold. Short text is
  saturated at any of these, so the whole difference is in long context. **If you mask long
  context, consider `0.05`** — it caught everything measured, at the cost of inventing spans in
  short text where there was nothing to gain. That corpus is small and synthetic: enough to reject
  `0.2`, not enough to call `0.1` optimal for your traffic.

- **`maxInputTokens`** — the cost ceiling, default `2000` tokens. Text longer than one 768-token
  window is masked in several passes rather than dropped; this bounds how many. Over it,
  `maskFields` returns nil. Counted in tokens rather than characters because 1,800 characters is
  379 tokens of English prose and 1,752 of CJK.

`maxWidth` and `maxSequenceLength` are deliberately **not** here: they must equal the model's
`config.max_width` and `config.max_len`, so they ride on `ModelPin`. That is what makes reverting to
an older pin safe. `maxSequenceLength` is the input cap described under [Limitations](#limitations)
— too low only costs dropped passes, too high lets over-length text reach a model that silently
stops detecting, so it has no default.

`MaskerLogging` takes the `os.Logger` subsystem, and optionally a `debug` closure. **That closure
receives raw user text.** It is nil by default and there is no runtime switch — a host that wants
it should gate it on its own development-build predicate.

## Limitations

- **Best-effort NER, not a guarantee.** Recall is not 100%. Some PII will get through. Do not build
  a compliance claim on this.
- **Long input is split into overlapping windows.** One inference pass is capped at
  the model's 768-token window, so longer text is masked in several overlapping passes and
  reassembled. Nothing is dropped for length until `MaskerConfig.maxInputTokens` (default 2,000
  tokens), which is a cost ceiling you can raise. But windowing only fixes the *catastrophic* case;
  it does not make long input mask as well as short input. Measured on the shipped weights: 8 of 8
  planted entities are caught in a bare sentence, and 6 of 8 once 300 words of surrounding prose are
  added. **If you mask long context, consider lowering the threshold** — see
  [Configuration](#configuration).
- **Languages without spaces are not supported, and fail closed.** GLiNER's spans are
  whitespace-delimited words, so Chinese, Japanese and Thai arrive as one enormous "word" that no
  window can hold, and the pass returns nil. Verified: the same CJK text masks correctly once
  spaces are inserted between characters. Nothing leaks — but users writing those languages get no
  masking, so a caller that gates on it holds every one of their requests. Segmenting them needs a
  word-splitter this library does not ship.
- **Tuned for English chat and email prose.** The threshold and the label set were chosen there.
  Other languages and other registers are not characterised.
- **Name masking covers one person** — the authenticated user whose given name and surname you pass
  in. Everyone else is masked only if the model happens to detect them.
- **Restored text is as sensitive as the input.** `restore` puts the real values back. The
  protection is over the wire, and only over the wire.
- **Text that already looks like a placeholder gets eaten.** The safety net that strips leftover
  `[LABEL_1]` tokens cannot tell them from user text of the same shape, and runs after the
  substitutions. A test documents this so a change to it is deliberate.
- **Rollback is tested, not battle-tested.** The arm → probe → revert chain is exercised by the
  test suite and has never run outside it, because it needs at least two published model versions
  and so far there is one.
- **The fetch path of a rollback is unreachable today** for the same reason. The flip path — the
  previous version's bytes are already on disk — is the one that runs.

## License and attribution

MIT — see [LICENSE](../LICENSE).

The model is a modified `vicgalle/gliner-small-pii` (Apache-2.0), itself fine-tuned from
`gliner-community/gliner_small-v2.5` (Apache-2.0). The modification is the ONNX + INT8 conversion
in [`scripts/export_gliner_v2.py`](../scripts/export_gliner_v2.py); no retraining was done. Full
attribution, including GLiNER itself and the Swift dependencies, is in [NOTICE](../NOTICE).

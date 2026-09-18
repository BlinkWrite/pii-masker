# pii-masker for .NET

Find PII in text on-device and replace it with **reversible** placeholders, so text can be sent to
a remote model and the real values swapped back into the answer locally.

The masking is reversible, and that is the whole design. A detected span becomes
`[EMAIL_ADDRESS_1]`, not `*****`. So a false positive costs you a placeholder you restore, not a
redaction marker leaking into text a person reads — and the remote model still sees a labelled
token it can reason around instead of a hole in the sentence.

Detection is [GLiNER](https://github.com/urchade/GLiNER) (ONNX INT8) through ONNX Runtime. Nothing
leaves the machine except the model download.

`net8.0` and above. MIT. The .NET target of [pii-masker](../README.md); the Swift target is
[over here](../swift/README.md), and the two are kept behaviour-for-behaviour.

## Quick start

Mask something on the command line first, before wiring it into anything.

**1. Build.**

```sh
git clone https://github.com/BlinkWrite/swift-pii-masker
cd swift-pii-masker/dotnet
dotnet build
```

**2. Download the model** (~137 MB).

```sh
cd ..
scripts/fetch-model.sh
```

It reads the URL and both hashes out of [`model.json`](../model.json), checks the archive, unpacks
it into `./model`, and checks the unpacked weights — the second check *after* unpacking, which is
the whole point of having two. See [Getting the model](#getting-the-model). Re-running it when the
model is already there does nothing. The script reads `model.json` with `python3`, so it needs one
on PATH. In an app you do none of this by hand: `ModelInstaller` performs the same fetch, the same
two checks, and an atomic promotion.

**3. Mask something.**

```sh
dotnet run --project dotnet/src/pii-mask -- --model ./model --show-map <<< "Card 4111 1111 1111 1111, SSN 123-45-6789, I live at 42 Elm Street."
```

```
Card [CREDIT_CARD_NUMBER_1], SSN [SOCIAL_SECURITY_NUMBER_1], I live at [ADDRESS_1].

  [ADDRESS_1] ← 42 Elm Street
  [CREDIT_CARD_NUMBER_1] ← 4111 1111 1111 1111
  [SOCIAL_SECURITY_NUMBER_1] ← 123-45-6789
```

**4. Run the tests.**

```sh
cd dotnet && dotnet test
```

## Add it to your project

```sh
dotnet add package PIIMasker
```

The package brings `Microsoft.ML.OnnxRuntime` and the managed `Tokenizers.DotNet`, but **not** a
native tokenizer runtime — that is a per-RID package, and forcing one would put a Windows-only
graph into a Linux consumer's dependency tree. Add the RIDs you ship:

```xml
<PackageReference Include="Tokenizers.DotNet.runtime.win-x64" Version="1.4.1" />
<PackageReference Include="Tokenizers.DotNet.runtime.linux-x64" Version="1.4.1" />
```

`win-arm64`, `linux-arm64`, `osx-x64` and `osx-arm64` exist under the same naming.

## Getting the model

The model is **pinned**: `ModelPin.Current` states the version, an immutable source URI, the
SHA-256 of the archive, the SHA-256 of the unpacked weights, the byte count, the two shape limits
the weights impose, and the SHA-256 of the two loose tokenizer files this target opens directly.
Those values are compiled in. [`model.json`](../model.json) states the same thing in a file a human
can diff, and `ModelPinTests` fails the build if the two disagree.

Both hashes are checked, and the order is the point rather than an accident: the archive is verified
before anything is unpacked, and the weights are verified again *after*, because an archive that
hashes correctly can still unpack to something else if the unpacking is interrupted.

`WithSourceUri` changes **where** the bytes come from and nothing about **which** bytes are
accepted. Point it at your own CDN or a `file://` fixture; the hashes are still the compiled ones,
so a compromised distribution host cannot substitute a model. A host should relocate it rather than
ship the public Hugging Face URL.

## Example

```csharp
using PIIMasker;

var root = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MyApp", "models");
ModelStore.Configure(root);

var installer = new ModelInstaller(root, ModelPin.Current);
var outcome = await installer.EnsureLatestAsync();
if (outcome.Installed)
{
    // New bytes: prove them before trusting them. See "Probe after an install".
    using var candidate = new PrivacyFilter(outcome.Directory);
    if (!(await candidate.ProbeAsync()).Passes) return;   // do not use this model
}

// MaskAsync does NOT mask the user's own name — see below. Apply it per field first, with the
// scope each field deserves.
var conversation = UserNameMask.MaskContext(
    rawConversation, "Alex", "Rivera", senderHeader: "SENDER|MESSAGE");
var draft = UserNameMask.Mask(rawDraft, "Alex", "Rivera", UserNameMask.Scope.FullNameOnly);

using var masker = new PrivacyFilter(ModelStore.ResolvedModelDirectory()!);
var result = await masker.MaskAsync([conversation, draft]);
if (result == null) return;          // ← the only correct response to null is to send nothing

var answer = await myRemoteModel.CompleteAsync(result.Fields[0], result.Fields[1]);
var final = PrivacyFilter.Restore(answer, result.Restore);
```

`MaskAsync` takes several strings and masks them in **one** inference pass, so they share a single
token map. Separate passes restart the counter, and a later field's `[PERSON_1]` would then collide
with an earlier one on restore.

**`MaskAsync` does not mask names.** Once the fields are joined it cannot tell them apart, and the
two scopes want different rules: a wrong match in a draft lands in text the user reads back, while
conversation text they do not own wants the given name masked too. So the choice stays with the
caller who knows which field is which — call `UserNameMask` per field first, as above.

`UserNameMask` takes the given name and the surname **separately**, and will not guess between
them: a display name carries no reliable order, and plenty of directories store it surname-first.
A host holding only a display name splits it itself, so that guess stays visible at the call site.

## How it works

1. Replace the authenticated user's own name with `[USER]` — a regex pass, before the model, and
   the caller's job (see above).
2. Join the fields with a guard and a separator built from ASCII RECORD SEPARATOR, and refuse the
   input outright if a field already contains one.
3. Split into whitespace words and tokenize, with the label set prefixed as GLiNER expects.
4. Split into overlapping windows if the text exceeds the model's 768-token limit, so no single
   pass runs past what the weights can actually read.
5. Run ONNX inference per window; sigmoid the logits and keep spans above the threshold. Every
   window reports character offsets into the same original text, so the windows never surface.
6. Split any span that crosses the framing around it, so the separator survives and the detected
   content on **both** sides is still masked.
7. Trim each span's edges past punctuation, so a flagged `ip?` masks `ip` and leaves the `?`.
8. Merge overlaps, then splice right-to-left into numbered placeholders (`[IP_ADDRESS_1]`).
9. Split the blob back into fields, and drop everything if the framing did not survive.
10. `Restore` swaps the originals back, and strips any placeholder-shaped leftover the remote model
    garbled — so a redaction marker can never reach the user.

## Probe after an install

**The library ships a health check but never calls it. Running it is your job.**

`PrivacyFilter.ProbeAsync()` masks a sentence carrying three unmistakable pieces of PII and reports
how many came back masked. Call it once whenever `EnsureLatestAsync()` reports `Installed == true`,
and treat two-of-three (`result.Passes`) as the pass mark — one miss is model jitter, two is a model
that is not doing its job.

This matters more than it looks. A bad model does not crash. Either it fails to load, and the
fail-closed gate holds every request while your app looks alive; or it loads and quietly detects
less, and the gate stops closing at all. Nothing else in this library would notice either.

`ModelRollback` builds on that: arm probation when you install over a working version, probe, and
revert the `current` pointer if the probe fails. Superseded version directories are deliberately
never deleted, which is what makes the revert a single atomic rename.

## Testing

```sh
cd dotnet && dotnet test
```

No network and no weights. The masking tests cover the pure logic — span trimming and merging,
placeholder minting, the restore round trip, the framing guard, the boundary splits, the name rules
— and the install tests build real `.tar.gz` fixtures and install them over `file://`, so the
download, both checksum gates, the unpack, the promotion and every rejection path are exercised
anywhere. `ModelPinTests` asserts the compiled pin equals the shared [`model.json`](../model.json).

Detection itself needs real weights, so it is a **model tier** that skips unless you opt in:

```sh
PII_MASKER_MODEL_DIR=/path/to/model dotnet test
```

That tier runs every label category through the model, plus the windowing, the window-boundary
overlap, the input budget, the timeout and the shipped probe. Without the variable those tests are
skipped rather than silently passing, so a green run on a machine with no model means what it says.

The library targets `net8.0`; the tests run on the current runtime, so the shipped floor is compiled
against rather than executed. A CI leg on net8.0 would close that gap.

## The CLI

`pii-mask` pipes text through the masker so you can see what it catches without writing any C#.

```sh
dotnet run --project dotnet/src/pii-mask -- [--model DIR] [--show-map] [--threshold N] [--timeout SECONDS]
```

`--model` defaults to `$PII_MASKER_MODEL_DIR`. `--show-map` prints each placeholder and the text it
replaced **on stderr**, so `pii-mask ... > masked.txt` still leaves you with only what would leave
the machine. Exit status is `0` masked, `1` bad usage or no model, `2` the masker produced nothing —
which a caller treats as "send nothing", and is not the same as "found no PII".

## Configuration

`MaskerConfig` carries the two inference knobs:

- **`Labels`** — fed to GLiNER at inference time. **Order is the output index space**: the model
  returns an index into this exact array, so the array used for inference must be the one used to
  read the result back. And a label is not just a filter, it *names the placeholder* —
  `"email address"` becomes `[EMAIL_ADDRESS_1]`. Rename one and you change the token text the
  remote model sees, and invalidate any restore map built before the change.
- **`Threshold`** — the score floor, default `0.1`. Raise it and PII slips through permanently;
  lower it and legitimate text becomes a placeholder, which `Restore` puts back — so a false
  positive costs prompt noise, not data. The two errors are not equally bad, and the floor sits
  below the middle for that reason. The default came from measurement on these weights, tabulated in
  the [Swift target's README](../swift/README.md#configuration); the short version is that recall
  depends far more on how much text surrounds the PII than on the threshold. **If you mask long
  context, consider `0.05`.**
- **`MaxInputTokens`** — the cost ceiling, default `2000` tokens. Text longer than one 768-token
  window is masked in several passes rather than dropped; this bounds how many. Over it, `MaskAsync`
  returns null. Counted in tokens rather than characters because 1,800 characters is 379 tokens of
  English prose and 1,752 of CJK.

`MaxWidth` and `MaxSequenceLength` are deliberately **not** here: they must equal the model's
`config.max_width` and `config.max_len`, so they ride on `ModelPin`. That is what makes reverting to
an older pin safe.

`MaskerLogging` takes an optional `Diagnostic` and an optional `Trace` callback. **`Trace` receives
raw user text.** Both are null by default and there is no runtime switch — a host that wants them
should gate them on its own development-build predicate.

## Where this deliberately differs from the Swift target

Same contract, different mechanism — each because the Swift approach does not work everywhere .NET
runs.

**Promotion is a pointer file, not a symlink.** Swift promotes a version by pointing a `current`
symlink at it. Creating a symlink on Windows needs elevation or Developer Mode, neither of which a
per-user install can assume — and a promotion that cannot happen means a verified model never
becomes the one that loads. This target writes a `current` plain-text file naming the version
directory, replaced with an atomic rename. `ModelStore.ResolvePromoted` reads **both** layouts, and
promoting replaces a symlink-shaped `current` as readily as a file — so a store either target can
read is one either target can write. Removing a symlink deletes the link and never its target, so
the version directory survives; a *real* directory named `current` is refused rather than deleted,
since that is a model directory in its own right. The name inside the pointer is checked against
`InstallSupport.IsSafePathComponent` before it is joined, because a file on disk is something a
caller can write.

**Unpacking is in-process.** Swift shells out to `/usr/bin/tar`. .NET has `System.Formats.Tar` over
`GZipStream` in the box, so there is no external process and no assumption about which tar is
installed. `TarFile.ExtractToDirectory` refuses entries that would escape the destination, which is
the traversal guard the shelled-out version got from bsdtar.

**Rollback state is injected.** Swift takes a `UserDefaults` and two key names. .NET has no
equivalent that is right on every platform a host might ship to, so `IRollbackStore` is an interface:
back it with the registry on Windows, a plist on macOS, or use the included `FileRollbackStore`.
`ModelRollback.VerifyAsync` also takes the health probe as a **required argument** rather than
defaulting to the filter's own — the rollback has no business constructing a masker, and a default
that quietly detected nothing would fail every model's probe and revert it.

**A cancelled pass throws rather than returning null.** `OperationCanceledException` is what a .NET
caller expects from a cancelled `Task`, and folding it into the same null the gate uses for "masked
nothing" would disguise a cancellation as a privacy drop.

Two additions with no Swift counterpart. `IsSafePathComponent` also rejects the Windows reserved
device names — `NUL`, `CON`, `COM1` and the rest are not filenames on Windows at any extension, so
`NUL.2026` resolves to the device rather than a directory, and an installer would write a model into
nothing and then fail to read it back with no indication why. And `ModelPin` carries the SHA-256 of
the loose tokenizer files, because this target opens them directly where Swift reaches the tokenizer
through swift-transformers.

One place the two produce different output today: the `(you)` marker some chat surfaces append to
the signed-in user's name is consumed with the name here, and left behind by the Swift target.
Swift's own comment says it should be consumed; its pattern puts the marker inside an alternation
ending in `)`, where the trailing word boundary cannot hold, so the engine backtracks to the bare
name. This target does what both intend.

## Limitations

- **Best-effort NER, not a guarantee.** Recall is not 100%. Some PII will get through. Do not build
  a compliance claim on this.
- **Long input is split into overlapping windows.** Nothing is dropped for length until
  `MaskerConfig.MaxInputTokens`. But windowing only fixes the *catastrophic* case; it does not make
  long input mask as well as short input.
- **Languages without spaces are not supported, and fail closed.** GLiNER's spans are
  whitespace-delimited words, so Chinese, Japanese and Thai arrive as one enormous "word" that no
  window can hold, and the pass returns null. Nothing leaks — but users writing those languages get
  no masking, so a caller that gates on it holds every one of their requests.
- **Tuned for English chat and email prose.** The threshold and the label set were chosen there.
- **Name masking covers one person** — the authenticated user whose given name and surname you pass
  in. Everyone else is masked only if the model happens to detect them.
- **Restored text is as sensitive as the input.** `Restore` puts the real values back. The
  protection is over the wire, and only over the wire.
- **Text that already looks like a placeholder gets eaten.** The safety net that strips leftover
  `[LABEL_1]` tokens cannot tell them from user text of the same shape. A test documents this so a
  change to it is deliberate.
- **Rollback is tested, not battle-tested.** The arm → probe → revert chain is exercised by the test
  suite and has never run outside it, because it needs at least two published model versions and so
  far there is one. The fetch path of a rollback is unreachable today for the same reason; the flip
  path — the previous version's bytes are already on disk — is the one that runs.

## License and attribution

MIT — see [LICENSE](../LICENSE).

The model is a modified `vicgalle/gliner-small-pii` (Apache-2.0), itself fine-tuned from
`gliner-community/gliner_small-v2.5` (Apache-2.0). The modification is the ONNX + INT8 conversion
in [`scripts/export_gliner_v2.py`](../scripts/export_gliner_v2.py); no retraining was done. Full
attribution, including GLiNER itself, is in [NOTICE](../NOTICE).

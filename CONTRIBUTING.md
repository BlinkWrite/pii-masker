# Contributing

## Building and testing

```sh
swift build
swift test
```

`swift test` needs no model and no network — that is deliberate, so a green run on a fresh
checkout means something. The tests that need real weights are a separate tier, skipped unless you
point them at a model:

```sh
PII_MASKER_MODEL_DIR=/path/to/gliner swift test
```

That directory needs `model.onnx`, `tokenizer.json` and `tokenizer_config.json`. To get one:

```sh
scripts/fetch-model.sh          # downloads, verifies both hashes, unpacks into ./model
PII_MASKER_MODEL_DIR=./model swift test
```

`fetch-model.sh` reads `model.json` with `python3`, so it needs one on PATH; it is used for nothing
else and the script says so if it is missing.

### The .NET target

Same shape, same contract:

```sh
cd dotnet
dotnet build
dotnet test
```

`dotnet test` likewise needs no model and no network, and the weights-only tier is opt-in the same
way:

```sh
PII_MASKER_MODEL_DIR=/path/to/gliner dotnet test
```

The library ships `net8.0`; the suite runs on the current runtime by default and on that floor as
well with `-p:TestFloor=true`, which is what CI does. Running it yourself needs the .NET 8 runtime
installed, which is why it is not the default.

There is also a `pii-mask` command-line tool, the counterpart of the Swift one, for looking at what
the masker catches without writing any code:

```sh
dotnet run --project dotnet/src/pii-mask -- --model ./model --show-map <<< "email me at a@b.com"
```

## What a good change looks like

- **A bug becomes a test first.** Write the failing test, confirm it is red for the right reason,
  then fix it.
- **Comments explain why, not what.** The code says what it does. A comment earns its place by
  recording the failure that made the code look like that.
- **Fail-closed stays fail-closed.** Any change that could make an entry point return text on a
  path where it used to return nil needs to say so explicitly in the pull request.

## Adding a model version

Models are pinned in source; there is no manifest and no runtime version negotiation. To ship a
new one:

1. Export it: `pip install -r scripts/requirements.txt`, then
   `python3 scripts/export_gliner_v2.py --out /some/dir`.
2. Package it: `scripts/package-model.sh --source /some/dir --version YYYY.MM.N --out dist`.
   It prints the four numbers a pin needs.
3. Upload the archive somewhere immutable. For Hugging Face that means a `resolve/<commit-sha>/`
   URL — never `resolve/main`, or the bytes under a published pin could change.
4. **Append** a pin in BOTH targets — `ModelPin.known` in Swift and `ModelPin.Known` in .NET.
   Never edit or remove an existing entry: rollback walks that list, and an older entry has to stay
   fetchable.
5. Copy the same seven fields into `model.json`, plus the `files` block: the SHA-256 of
   `tokenizer.json` and `tokenizer_config.json`. The .NET target opens those two directly and
   verifies them on every load, so they are part of the identity it accepts; the Swift target
   reaches the tokenizer through swift-transformers and ignores the block. `package-model.sh`
   prints the numbers a pin needs.

   Each target has a test asserting its own pin equals `model.json`, so skipping either is a red
   build rather than a silent drift — but note that means updating one target and not the other
   fails only that target's suite.

`maxWidth` must equal the model's `config.max_width`, and `maxSequenceLength` its `config.max_len`.
Both ride on the pin rather than in `MaskerConfig` because they are properties of the weights —
which is what makes reverting to an older entry safe. `package-model.sh` reads both out of
`gliner_config.json` so neither is typed by hand; a `maxSequenceLength` set too high is a silent
privacy failure, not just a wrong number. Labels and the score threshold are not properties of the
weights: GLiNER takes labels at inference time, so any label set works with any model.

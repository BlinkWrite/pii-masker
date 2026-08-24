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
4. **Append** a `ModelPin` to `ModelPin.known`. Never edit or remove an existing entry: rollback
   walks that list, and an older entry has to stay fetchable.
5. Copy the same seven fields into `model.json`. A test compares the two, so skipping this is a red
   build rather than a silent drift.

`maxWidth` must equal the model's `config.max_width`, and `maxSequenceLength` its `config.max_len`.
Both ride on the pin rather than in `MaskerConfig` because they are properties of the weights —
which is what makes reverting to an older entry safe. `package-model.sh` reads both out of
`gliner_config.json` so neither is typed by hand; a `maxSequenceLength` set too high is a silent
privacy failure, not just a wrong number. Labels and the score threshold are not properties of the
weights: GLiNER takes labels at inference time, so any label set works with any model.

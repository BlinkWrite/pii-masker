# pii-masker

Find PII in text on-device and replace it with **reversible** placeholders, so text can be sent to
a remote model and the real values swapped back into the answer locally.

The masking is reversible, and that is the whole design. A detected span becomes
`[EMAIL_ADDRESS_1]`, not `*****`. So a false positive costs you a placeholder you restore, not a
redaction marker leaking into text a person reads — and the remote model still sees a labelled
token it can reason around instead of a hole in the sentence.

Detection is [GLiNER](https://github.com/urchade/GLiNER) (ONNX INT8) through ONNX Runtime. Nothing
leaves the machine except the model download. MIT.

## Pick your language

| | Status | Docs |
|---|---|---|
| **Swift** — macOS 14+ | Masking, model install, rollback, `pii-mask` CLI | [swift/README.md](swift/README.md) |
| **.NET** — net8.0+ | Masking, model install, rollback, `pii-mask` CLI | [dotnet/README.md](dotnet/README.md) |

Both targets do the same job, and each ships the same `pii-mask` command-line tool so the two can
be compared on the same input. The Swift one is the only one with a published release history so
far. The handful of places they deliberately diverge are listed in the .NET target's README.

## What both targets share, and why they live together

One repository, because the two targets have to agree on things that are cheap to state twice and
expensive to get wrong twice.

**[`model.json`](model.json)** is the canonical statement of which model release is trusted: the
version, the immutable source URL, the archive and weights SHA-256, the size, and the two shape
limits the weights impose. It is the one file at the root that both targets read. Nothing fetches
it at runtime — it exists so a human can diff it without building anything, and **each target has a
test asserting its own compiled pin equals this document**. Two languages stating one fact will
drift; this makes the drift a red build in whichever one moved.

The `files` block in it is read by the .NET target only. The Swift target reaches the tokenizer
through swift-transformers and never opens those files, so it ignores the block; the .NET target
opens them directly, which makes their bytes part of the identity it accepts.

**[`scripts/`](scripts)** is the model tooling, shared the same way: `fetch-model.sh` reads
`model.json` and performs the same two checks an installed model gets, and `export_gliner_v2.py`
is how the weights were produced.

## Layout

```
model.json          the trusted model release — read by BOTH targets
scripts/            model tooling: fetch, package, export
Package.swift       the Swift manifest; must stay at the root (see below)
swift/              the Swift target — Sources, Tests, docs
dotnet/             the .NET target — src, tests, docs
```

`Package.swift` sits at the repository root rather than in `swift/`, and cannot move. SwiftPM
resolves a git dependency by reading the manifest at the repository root, with no way to point a
repository URL at a subdirectory — so relocating it would break every consumer pinning this
repository. Its targets carry explicit `path:` values into `swift/` instead.

## Security

Both targets fail closed: a failure to mask drops the text rather than sending it. See
[SECURITY.md](SECURITY.md) for how to report an issue, and
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

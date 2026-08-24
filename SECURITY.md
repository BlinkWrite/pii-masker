# Security

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/BlinkWrite/swift-pii-masker/security/advisories/new) rather than
opening a public issue. Please include a reproduction and the version or commit you tested.

We aim to acknowledge within a few working days.

## What this library does and does not promise

It is worth being precise, because "PII masker" invites assumptions this code does not meet.

**It promises:**

- **Fail-closed.** `maskFields` returns nil when the model is missing, the load fails, inference
  throws, the pass exceeds its timeout, the input is longer than the model's window, or the masked
  text does not round-trip. It never returns the input unchanged as a fallback. A caller that sends
  on nil has defeated the library, not found a bug in it.
- **No silent under-masking on long input.** One inference pass is limited to
  `ModelPin.maxSequenceLength` tokens (768 for the shipped weights), because GLiNER's position
  embeddings are relative: over-length input throws nothing, it quietly detects less, reaching zero
  detections around 1,250 tokens — and an empty result is indistinguishable from clean text. Longer
  text is therefore split into overlapping windows, each inside that limit, and the spans pooled.
  Windows overlap by `maxWidth - 1` words so an entity cannot hide in a cut. Beyond
  `MaskerConfig.maxInputTokens` (default 2,000) the pass is dropped rather than run unbounded.
  Without this the library fails *open*, which is the only way it can.
- **Reversible masking.** A detected span becomes `[EMAIL_ADDRESS_1]`, not `*****`, so a false
  positive costs a placeholder the caller restores rather than a redaction marker leaking into
  text a person reads.
- **On-device inference.** The only network access in this package is downloading the model, from
  the URL the pin names. Nothing sends text anywhere.
- **A verifiable model.** `ModelPin` carries the SHA-256 of the archive and of the unpacked
  weights; both are checked before the model becomes `current`. The same numbers are in
  `model.json` and on the model card, and a test asserts the two copies agree.
- **No user text in the system log on a release build.** The verbose channel is nil unless a host
  explicitly passes a closure (`MaskerLogging.debug`). There is no runtime switch for it.

**It does not promise:**

- **Complete recall.** This is best-effort NER, not a guarantee. Some PII will get through. Do not
  build a compliance claim on it. Recall is markedly worse with context around the PII than without
  — measured at the default 0.1 threshold, 8 of 8 planted entities in a bare sentence but 6 of 8
  with 300 words of prose around them. Recall depends far more on surrounding context than on the
  threshold. Lower `MaskerConfig.threshold` if you mask long context; the README's Configuration
  section has the numbers.
- **Anything about text after restore.** `restore` puts the real values back. The protection is
  over the wire only; the restored text is as sensitive as what went in.
- **Protection for anyone but the one authenticated user you name.** `UserNameMask` masks the name
  you pass it. Other people's names are only masked if the model happens to detect them.
- **Uniform quality across languages.** The threshold and label set are tuned for English chat and
  email prose. Non-English text also tokenizes far denser, so it reaches the window limit sooner.
- **Integrity of a model you point it at yourself.** Change `sourceURL` to your own mirror and the
  hashes still gate what installs — but if you change the hashes too, you are trusting your own
  host, which is the thing pinning exists to avoid.

Do not read "fail-closed" as "catches everything". It means the library never *pretends* to have
masked when it could not run — not that it finds every entity when it did.

## Threat model

The adversary is the remote service the masked text is sent to, plus anyone who can read that
traffic or those logs. It is not an attacker with code execution on the user's machine: everything
here runs in-process with the host application's privileges, and the restore map holds the
originals in memory for the duration of a request.

A tampered model is in scope, and is what the two hashes are for. A tampered *pin* is not — it is
compiled into the binary, so an attacker who can change it can change anything else too.

#!/usr/bin/env bash
# Downloads the pinned model, verifies it, and unpacks it into ./model.
#
# Every value comes out of model.json — the same seven fields as ModelPin.current,
# which a test keeps in step — so there is nothing to copy by hand here, and
# nothing in this script to update when the pin moves.
#
# Both hashes are checked, and the second one AFTER unpacking. That ordering is
# the point rather than an accident: see "Getting the model" in the README.
#
# Usage: scripts/fetch-model.sh [--out DIR] [--keep-archive]
set -euo pipefail

OUT="model"
KEEP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --keep-archive) KEEP=1; shift ;;
        -h|--help) grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

err() { echo "error: $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$ROOT/model.json"
[[ -f "$PIN" ]] || err "model.json not found at $PIN"

field() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$PIN" "$1"; }

VERSION="$(field version)"
URL="$(field sourceURL)"
ARCHIVE_SHA="$(field archiveSHA256)"
WEIGHTS_SHA="$(field weightsSHA256)"
BYTES="$(field bytes)"

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

# Re-running costs nothing: if the weights on disk are already the pinned ones,
# there is no download to repeat and no check left to make.
if [[ -f "$OUT/model.onnx" && "$(sha256_of "$OUT/model.onnx")" == "$WEIGHTS_SHA" ]]; then
    echo "$OUT already holds model $VERSION — nothing to do."
    exit 0
fi

ARCHIVE="$OUT.tar.gz"
mkdir -p "$OUT"

echo "Downloading model $VERSION ($((BYTES / 1048576)) MB)"
curl -fL --retry 3 --progress-bar -o "$ARCHIVE" "$URL"

echo "Checking the archive"
GOT="$(sha256_of "$ARCHIVE")"
[[ "$GOT" == "$ARCHIVE_SHA" ]] || err "archive hash mismatch — expected $ARCHIVE_SHA, got $GOT"

echo "Unpacking into $OUT"
tar -xzf "$ARCHIVE" -C "$OUT"

# The archive hash says the download arrived intact. This one says the weights
# that came out of it are the pinned ones, which is the claim that matters.
echo "Checking the weights"
[[ -f "$OUT/model.onnx" ]] || err "$OUT/model.onnx missing — the archive did not unpack flat"
GOT="$(sha256_of "$OUT/model.onnx")"
[[ "$GOT" == "$WEIGHTS_SHA" ]] || err "weights hash mismatch — expected $WEIGHTS_SHA, got $GOT"

for required in tokenizer.json tokenizer_config.json; do
    [[ -f "$OUT/$required" ]] || err "$OUT/$required missing — the masker needs it to load"
done

[[ $KEEP -eq 1 ]] || rm -f "$ARCHIVE"

echo
echo "Model $VERSION is in $OUT. Try it:"
echo "  swift run pii-mask --model $OUT --show-map <<< \"email me at a@b.com\""

#!/usr/bin/env bash
# Packages an exported model directory into the versioned .tar.gz that
# ModelInstaller downloads, and prints the four numbers a ModelPin needs.
#
# This is the second half of the published recipe. export_gliner_v2.py produces
# the model directory; this turns it into the archive, and between them a reader
# can reproduce what ModelPin names instead of taking it on faith.
#
# The archive is REPRODUCIBLE: `tar -czf` would embed a gzip timestamp and each
# file's checkout-time mtime, so every run would differ. mtime and owner are
# pinned and gzip's timestamp stripped, leaving bytes that depend only on the
# model content. Note that bsdtar and GNU tar still disagree with each other —
# which is exactly why ModelPin carries weightsSHA256 as well: the hash of the
# unpacked model.onnx is the same number under either packer.
#
# Usage: scripts/package-model.sh --source DIR --version V [--out DIR]
set -euo pipefail

SRC=""
OUT="dist"
VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SRC="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        -h|--help) grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

err() { echo "error: $*" >&2; exit 1; }

[[ -n "$SRC" ]] || err "--source is required (the directory export_gliner_v2.py wrote)"
[[ -n "$VERSION" ]] || err "--version is required (becomes the install directory name)"
[[ -d "$SRC" ]] || err "model source dir not found: $SRC"
[[ -f "$SRC/model.onnx" ]] || err "missing $SRC/model.onnx"

# The version becomes an on-disk path component and the `current` symlink target,
# so it has to pass InstallSupport.isSafePathComponent — same rule, stated twice
# because a rejection here is far cheaper than one after a 137 MB download.
[[ "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || err "unsafe version '$VERSION': allowed characters are A-Z a-z 0-9 . _ -"
[[ "$VERSION" != "." && "$VERSION" != ".." ]] || err "unsafe version '$VERSION'"

# A checkout without `git lfs pull` leaves model.onnx as a ~130-byte pointer;
# packaging that would ship a broken model that still passes every size check.
if head -c 100 "$SRC/model.onnx" | grep -q 'git-lfs'; then
    err "$SRC/model.onnx is an unmaterialized Git LFS pointer — run: git lfs pull"
fi

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

ARCHIVE_NAME="gliner-pii-$VERSION.tar.gz"
mkdir -p "$OUT"
ARCHIVE="$OUT/$ARCHIVE_NAME"

# Tar the model files by explicit name (skips .DS_Store and other junk). The
# first three are ModelInstaller.requiredModelFiles; gliner_config.json rides
# along because it carries max_width, which ModelPin has to match.
# These filenames never contain spaces, so unquoted splitting is safe.
FILE_LIST=""
for f in model.onnx tokenizer.json tokenizer_config.json gliner_config.json \
         special_tokens_map.json config.json added_tokens.json; do
    [[ -f "$SRC/$f" ]] && FILE_LIST="$FILE_LIST $f"
done
for required in model.onnx tokenizer.json tokenizer_config.json; do
    [[ -f "$SRC/$required" ]] || err "missing $SRC/$required — the installer requires it"
done

echo "Packaging $SRC → $ARCHIVE (version $VERSION)"
# The archive unpacks FLAT — no top-level directory — which is what
# isCompleteModelDir expects of the staging dir.
if tar --version 2>/dev/null | grep -q 'GNU tar'; then
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -C "$SRC" -cf - $FILE_LIST | gzip -n > "$ARCHIVE"
else
    # bsdtar (macOS): no --sort/--mtime, but the fixed FILE_LIST order + gzip -n
    # keep repeat runs on one checkout stable.
    tar --uid 0 --gid 0 --numeric-owner --no-mac-metadata -C "$SRC" -cf - $FILE_LIST | gzip -n > "$ARCHIVE"
fi

ARCHIVE_SHA="$(sha256_of "$ARCHIVE")"
WEIGHTS_SHA="$(sha256_of "$SRC/model.onnx")"
BYTES="$(wc -c < "$ARCHIVE" | tr -d '[:space:]')"

# Both are properties of the WEIGHTS, so they are read out of the exported config rather than
# typed by hand. max_len is the safety-critical one: too high and over-length input reaches a model
# that silently stops detecting, which is the one way the masker can fail open.
config_field() {
    python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
        "$SRC/gliner_config.json" "$1" 2>/dev/null || echo unknown
}
MAX_WIDTH="unknown"
MAX_SEQ="unknown"
if [[ -f "$SRC/gliner_config.json" ]] && command -v python3 >/dev/null 2>&1; then
    MAX_WIDTH="$(config_field max_width)"
    MAX_SEQ="$(config_field max_len)"
fi
[[ "$MAX_WIDTH" != "unknown" ]] || echo "warning: could not read max_width — fill it in by hand" >&2
[[ "$MAX_SEQ" != "unknown" ]] || echo "warning: could not read max_len — fill it in by hand" >&2

cat > "$OUT/model.json" <<EOF
{
  "version": "$VERSION",
  "sourceURL": "<publish the archive, then put its immutable URL here>",
  "archiveSHA256": "$ARCHIVE_SHA",
  "weightsSHA256": "$WEIGHTS_SHA",
  "bytes": $BYTES,
  "maxWidth": $MAX_WIDTH,
  "maxSequenceLength": $MAX_SEQ
}
EOF

echo
echo "Wrote $OUT/model.json:"
cat "$OUT/model.json"
echo
echo "Next: upload $ARCHIVE, then add a ModelPin entry with the fields above and"
echo "the archive's immutable URL (a commit SHA, never a branch). Copy the same"
echo "seven fields into model.json at the repository root — a test compares them."

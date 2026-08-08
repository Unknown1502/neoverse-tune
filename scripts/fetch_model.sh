#!/usr/bin/env bash
#
# fetch_model.sh — download and VERIFY the benchmark model.
#
# WHY THE DEFAULT IS PINNED BY HASH
# ---------------------------------
# Every number published by this repository was produced by one specific file:
# Qwen2.5-1.5B-Instruct at Q4_K_M, 1,117,320,736 bytes. An earlier version of
# this script defaulted to the 0.5B build, which meant a CI run on Arm would
# have measured a different model than the x86 baseline it was meant to be
# compared against — two numbers that cannot go in the same table. Comparability
# is the whole reason to collect an Arm number, so the model is now pinned by
# content hash rather than by name.
#
# The hash also closes a supply-chain gap: a substituted or truncated artifact
# would previously have been benchmarked silently.
#
# The pinned digest was verified against HuggingFace's X-Linked-ETag, which for
# LFS objects is the SHA-256 of the file itself.
#
# NOTE ON QUANTIZATION
# --------------------
# Q4_K_M is a k-quant, NOT the Q4_0 format that KleidiAI's int4 matmul
# microkernels target. That is irrelevant to the slot-selection finding, which
# is scheduler behaviour. It matters a great deal to any future per-core prefill
# comparison, where the quantization format decides which microkernel runs.
# Sweep formats deliberately there; do not change this default to chase one.
#
# Usage:
#   scripts/fetch_model.sh                    # pinned default, verified
#   scripts/fetch_model.sh <url>              # override, verification skipped
#   SPECARM_MODEL_URL=... SPECARM_MODEL_SHA256=... scripts/fetch_model.sh
#
# Writes: $MODELS_DIR/<name>.gguf, results/model.path, results/model.json

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# --- The pinned artifact ------------------------------------------------------
PINNED_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PINNED_SHA256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
PINNED_BYTES=1117320736

URL="${1:-${SPECARM_MODEL_URL:-$PINNED_URL}}"

# A hash applies only to the artifact it was computed from. If the caller points
# at a different URL, the pinned digest is meaningless and must not be used to
# "verify" it.
if [ "$URL" = "$PINNED_URL" ]; then
  EXPECT_SHA256="${SPECARM_MODEL_SHA256:-$PINNED_SHA256}"
else
  EXPECT_SHA256="${SPECARM_MODEL_SHA256:-}"
fi

NAME="$(basename "${URL%%\?*}")"
mkdir -p "$MODELS_DIR"
DEST="$MODELS_DIR/$NAME"

# --- Hashing, with fallbacks --------------------------------------------------
# GNU coreutils, BSD/macOS, then Python. Returns non-zero if none are available,
# which is reported as "unverified" rather than treated as a hash mismatch.
sha256_of() {
  if have sha256sum; then
    sha256sum "$1" | cut -d' ' -f1
  elif have shasum; then
    shasum -a 256 "$1" | cut -d' ' -f1
  elif have python3; then
    python3 -c 'import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],"rb") as f:
    for c in iter(lambda: f.read(1<<20), b""): h.update(c)
print(h.hexdigest())' "$1"
  else
    return 1
  fi
}

# Verify DEST against EXPECT_SHA256. Echoes the actual digest on stdout.
#   0 = matched, 1 = MISMATCH, 2 = could not hash, 3 = nothing to check against
verify() {
  local actual
  actual="$(sha256_of "$DEST" 2>/dev/null)" || return 2
  echo "$actual"
  [ -n "$EXPECT_SHA256" ] || return 3
  [ "$actual" = "$EXPECT_SHA256" ] || return 1
  return 0
}

# --- Fetch --------------------------------------------------------------------
if [ -s "$DEST" ]; then
  log "model already present: $DEST ($(du -h "$DEST" | cut -f1))"
else
  log "downloading $NAME"
  [ -n "$EXPECT_SHA256" ] && log "expecting sha256 $EXPECT_SHA256"
  if have curl; then
    curl -fL --retry 3 --retry-delay 2 -o "$DEST.part" "$URL" || die "download failed: $URL"
  elif have wget; then
    wget -q --tries=3 -O "$DEST.part" "$URL" || die "download failed: $URL"
  else
    die "need curl or wget to fetch the model"
  fi
  # Move into place only after the transfer completes, so an interrupted
  # download is never mistaken for a finished one. Verification happens next,
  # against the final path, so a cached file gets checked on every run too.
  mv "$DEST.part" "$DEST"
  log "saved $DEST ($(du -h "$DEST" | cut -f1))"
fi

# --- Verify -------------------------------------------------------------------
ACTUAL_SHA256=""
VERIFIED=false
set +e
ACTUAL_SHA256="$(verify)"
rc=$?
set -e

case "$rc" in
  0)
    VERIFIED=true
    log "sha256 OK: $ACTUAL_SHA256"
    ;;
  1)
    # Refuse rather than warn. A wrong model silently produces numbers that
    # look fine and mean nothing.
    rm -f "$DEST"
    die "SHA-256 MISMATCH — removed $DEST
       expected $EXPECT_SHA256
       actual   $ACTUAL_SHA256
     The artifact does not match the pinned model. Re-run to download again;
     if it persists, the upstream file changed and the pin needs updating with
     a deliberate commit, not a silent override."
    ;;
  2)
    warn "no sha256sum, shasum or python3 available — model NOT verified"
    ;;
  3)
    warn "custom model URL with no SPECARM_MODEL_SHA256 — model NOT verified."
    warn "Numbers from an unpinned model are not comparable to the published ones."
    ;;
esac

BYTES="$(wc -c < "$DEST" | tr -d '[:space:]')"
if [ "$URL" = "$PINNED_URL" ] && [ "$BYTES" != "$PINNED_BYTES" ]; then
  warn "size $BYTES != pinned $PINNED_BYTES (hash check above is authoritative)"
fi

# --- Record what was actually used --------------------------------------------
# run_matrix.py records only the basename. This is where the identity lives, so
# a result can be traced to an exact artifact rather than a filename.
cat > "$RESULTS_DIR/model.json" <<EOF
{
  "schema": "specarm.model/1",
  "path": $(json_escape "$DEST"),
  "name": $(json_escape "$NAME"),
  "url": $(json_escape "$URL"),
  "sha256": $(json_escape "$ACTUAL_SHA256"),
  "sha256_expected": $(json_escape "$EXPECT_SHA256"),
  "verified": $VERIFIED,
  "bytes": $BYTES,
  "is_pinned_default": $([ "$URL" = "$PINNED_URL" ] && echo true || echo false)
}
EOF

# Consumed by the CI workflow and the sweep script.
echo "$DEST" > "$RESULTS_DIR/model.path"
echo "$DEST"

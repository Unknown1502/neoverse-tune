#!/usr/bin/env bash
#
# fetch_model.sh — download the benchmark model.
#
# Default is Qwen2.5-0.5B-Instruct Q4_0 (Apache-2.0, official Qwen GGUF repo).
# Q4_0 is the quantization KleidiAI's int4 matmul microkernels target, and 0.5B
# keeps a 4-vCPU CI runner honest on time. For headline numbers use a larger
# target model on a bigger instance — override with SPECARM_MODEL_URL.
#
# Usage: scripts/fetch_model.sh [url]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DEFAULT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
URL="${1:-${SPECARM_MODEL_URL:-$DEFAULT_URL}}"
NAME="$(basename "${URL%%\?*}")"

mkdir -p "$MODELS_DIR"
DEST="$MODELS_DIR/$NAME"

if [ -s "$DEST" ]; then
  log "model already present: $DEST ($(du -h "$DEST" | cut -f1))"
else
  log "downloading $NAME"
  if have curl; then
    curl -fL --retry 3 --retry-delay 2 -o "$DEST.part" "$URL" || die "download failed: $URL"
  elif have wget; then
    wget -q --tries=3 -O "$DEST.part" "$URL" || die "download failed: $URL"
  else
    die "need curl or wget to fetch the model"
  fi
  mv "$DEST.part" "$DEST"
  log "saved $DEST ($(du -h "$DEST" | cut -f1))"
fi

# Consumed by the sweep script.
echo "$DEST" > "$RESULTS_DIR/model.path"
echo "$DEST"

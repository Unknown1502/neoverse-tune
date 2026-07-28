#!/usr/bin/env bash
#
# 05_kai_probe.sh — build and run the direct KleidiAI microkernel probe.
#
# This is the measurement with nothing in between. 02_batch_sweep.sh infers
# kernel behaviour through llama.cpp; this calls the microkernels themselves and
# sweeps the row count M, which in a forward pass is the number of tokens being
# processed together.
#
# Pipeline: vendor KleidiAI -> generate the variant table from that checkout ->
# build -> run -> JSON.
#
# Usage: scripts/05_kai_probe.sh [--n 4096] [--k 4096] [--reps 20]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

N="${SPECARM_PROBE_N:-4096}"
K="${SPECARM_PROBE_K:-4096}"
REPS="${SPECARM_PROBE_REPS:-20}"
KAI_REF="${SPECARM_KAI_REF:-main}"
KAI_REPO="${SPECARM_KAI_REPO:-https://github.com/ARM-software/kleidiai.git}"

while [ $# -gt 0 ]; do
  case "$1" in
    --n)    N="${2:?}";    shift 2 ;;
    --k)    K="${2:?}";    shift 2 ;;
    --reps) REPS="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

require_aarch64
for tool in git cmake python3; do
  have "$tool" || die "missing '$tool'"
done

# ---------------------------------------------------------------- vendor KleidiAI
mkdir -p "$VENDOR_DIR"
KAI_ROOT="$VENDOR_DIR/kleidiai"

if [ ! -d "$KAI_ROOT/.git" ]; then
  log "cloning KleidiAI ($KAI_REF)"
  git clone --filter=blob:none "$KAI_REPO" "$KAI_ROOT"
  git -C "$KAI_ROOT" checkout "$KAI_REF"
else
  log "reusing KleidiAI checkout at $KAI_ROOT"
fi
KAI_SHA="$(git -C "$KAI_ROOT" rev-parse HEAD)"
log "KleidiAI resolved to $KAI_SHA"

# ---------------------------------------------------------------- variant table
BUILD_DIR="$SPECARM_ROOT/src/kai_probe/build"
GEN_DIR="$BUILD_DIR/generated"
mkdir -p "$GEN_DIR"

log "discovering microkernel variants in the checkout"
python3 "$SPECARM_ROOT/tools/gen_variants.py" "$KAI_ROOT" "$GEN_DIR/kai_variants.inc" \
  || die "variant discovery failed — inspect $KAI_ROOT/kai/ukernels/matmul"

# Surface the real list. If a kernel name has moved between KleidiAI releases,
# this is where you see it rather than in a linker error.
log "variants compiled in:"
grep -o 'KAI_V([^,]*' "$GEN_DIR/kai_variants.inc" | sed 's/KAI_V(/    /' || true

# ---------------------------------------------------------------- build
log "configuring probe"
cmake -S "$SPECARM_ROOT/src/kai_probe" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DKAI_ROOT="$KAI_ROOT" \
  > "$RESULTS_DIR/kai_probe_configure.log" 2>&1 \
  || { tail -n 40 "$RESULTS_DIR/kai_probe_configure.log" >&2
       die "cmake configure failed (log: $RESULTS_DIR/kai_probe_configure.log)"; }

log "building probe"
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || echo 2)" \
  > "$RESULTS_DIR/kai_probe_build.log" 2>&1 \
  || { tail -n 60 "$RESULTS_DIR/kai_probe_build.log" >&2
       die "build failed (log: $RESULTS_DIR/kai_probe_build.log)"; }

BIN="$(find "$BUILD_DIR" -name kai_probe -type f -perm -u+x 2>/dev/null | head -n1)"
[ -n "$BIN" ] || die "built, but cannot locate the kai_probe binary under $BUILD_DIR"

# ---------------------------------------------------------------- run
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RESULTS_DIR/kai_probe_${STAMP}.json"

log "running probe (n=$N k=$K reps=$REPS)"
set +e
"$BIN" --n "$N" --k "$K" --reps "$REPS" --out "$OUT"
RC=$?
set -e

# Exit 2 means variants disagreed numerically: the packing is wrong, so the
# timings are meaningless. Keep the file for diagnosis but fail the step.
if [ "$RC" -eq 2 ]; then
  warn "variant outputs disagreed — packing is wrong, timings are not usable"
  warn "kept $OUT for diagnosis"
  exit 2
elif [ "$RC" -ne 0 ]; then
  die "probe exited $RC"
fi

# Record which KleidiAI produced this.
python3 - "$OUT" "$KAI_SHA" "$KAI_REF" <<'PY'
import json, sys
path, sha, ref = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as fh:
    data = json.load(fh)
data["kleidiai_sha"] = sha
data["kleidiai_ref"] = ref
with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
PY

cp "$OUT" "$RESULTS_DIR/kai_probe.latest.json"
log "wrote $OUT"
log "next: python3 tools/analyze_probe.py"

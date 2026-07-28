#!/usr/bin/env bash
#
# 02_batch_sweep.sh — THE CRUX EXPERIMENT.
#
# Thesis under test: single-token decode (batch=1) runs as a GEMV and cannot use
# KleidiAI's i8mm SMMLA matmul microkernels, which need multiple rows. Verifying
# N speculative draft tokens is one forward pass over N tokens — the same matmul
# shape as prompt processing of N tokens. So `llama-bench -p N` for small N is a
# precise proxy for "what does speculative verification of N tokens cost?"
#
# If the thesis holds we expect a *knee* in tokens/sec somewhere at small N,
# where the kernel selection flips from dotprod-GEMV to i8mm-GEMM. The knee is
# necessary but not sufficient evidence — 03_kernel_attrib.sh supplies the
# mechanism proof by naming the symbol that actually ran.
#
# Usage: scripts/02_batch_sweep.sh [--reps N] [--threads N] [--sweep 1,2,4,...]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

REPS="${SPECARM_REPS:-10}"
THREADS="${SPECARM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
# Dense at the low end: that is where a GEMV->GEMM switch would live.
SWEEP="${SPECARM_SWEEP:-1,2,3,4,6,8,12,16,24,32}"
TG_TOKENS="${SPECARM_TG:-64}"

while [ $# -gt 0 ]; do
  case "$1" in
    --reps)    REPS="${2:?}";    shift 2 ;;
    --threads) THREADS="${2:?}"; shift 2 ;;
    --sweep)   SWEEP="${2:?}";   shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -f "$RESULTS_DIR/build.latest.json" ] || die "no build found — run scripts/01_build_llama.sh first"

MODEL="$(cat "$RESULTS_DIR/model.path" 2>/dev/null || true)"
[ -s "${MODEL:-/nonexistent}" ] || MODEL="$("$SPECARM_ROOT/scripts/fetch_model.sh")"

# Pull the two binaries back out of the build manifest without needing jq.
BIN_KLEIDI="$(sed -n 's/.*"bin_kleidi": "\(.*\)",/\1/p' "$RESULTS_DIR/build.latest.json")"
BIN_BASE="$(sed  -n 's/.*"bin_base": "\(.*\)",/\1/p'   "$RESULTS_DIR/build.latest.json")"
[ -x "$BIN_KLEIDI" ] || die "kleidi binary not executable: $BIN_KLEIDI"
[ -x "$BIN_BASE"   ] || die "base binary not executable: $BIN_BASE"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# --- environment integrity, sampled around the run -------------------------
# A benchmark taken on a throttling burstable instance is not a benchmark. We
# record enough for the Skeptic to reject the run rather than silently trust it.
snapshot_env() {
  local tag="$1"
  {
    echo "  \"loadavg_$tag\": $(json_escape "$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"),"
    local f="/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    if [ -r "$f" ]; then
      echo "  \"cpu0_khz_$tag\": $(cat "$f"),"
    else
      echo "  \"cpu0_khz_$tag\": null,"
    fi
  }
}

run_sweep() {
  local variant="$1" bin="$2"
  local out_pp="$RESULTS_DIR/sweep_${variant}_pp_${STAMP}.json"
  local out_tg="$RESULTS_DIR/sweep_${variant}_tg_${STAMP}.json"

  log "[$variant] prompt-batch sweep p={$SWEEP}, r=$REPS, t=$THREADS"
  "$bin" -m "$MODEL" -p "$SWEEP" -n 0 -r "$REPS" -t "$THREADS" -o json > "$out_pp" \
    || die "[$variant] batch sweep failed — see output above"

  log "[$variant] single-token decode baseline (tg$TG_TOKENS)"
  "$bin" -m "$MODEL" -p 0 -n "$TG_TOKENS" -r "$REPS" -t "$THREADS" -o json > "$out_tg" \
    || die "[$variant] decode baseline failed"

  log "[$variant] wrote $out_pp / $out_tg"
}

{
  echo "{"
  echo "  \"schema\": \"specarm.run/1\","
  echo "  \"stamp\": $(json_escape "$STAMP"),"
  echo "  \"model\": $(json_escape "$(basename "$MODEL")"),"
  echo "  \"threads\": $THREADS,"
  echo "  \"reps\": $REPS,"
  echo "  \"sweep\": $(json_escape "$SWEEP"),"
  snapshot_env before
} > "$RESULTS_DIR/run_${STAMP}.json"

run_sweep kleidi "$BIN_KLEIDI"
run_sweep base   "$BIN_BASE"

{
  snapshot_env after
  echo "  \"completed_utc\": $(json_escape "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  echo "}"
} >> "$RESULTS_DIR/run_${STAMP}.json"

echo "$STAMP" > "$RESULTS_DIR/run.latest"
log "sweep complete — stamp $STAMP"
log "next: python3 tools/analyze.py --stamp $STAMP"

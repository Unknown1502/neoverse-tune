#!/usr/bin/env bash
#
# 04_serve_bench.sh — serving-level benchmark: the Cloud AI regime.
#
# 02_batch_sweep.sh measures a matmul in isolation. This measures a *service*,
# and it is where the kernel question stops being academic: llama-server batches
# concurrent requests together, so raising client concurrency raises the actual
# matmul batch size. If the i8mm thesis holds, throughput should climb faster
# than concurrency alone would explain — and it should do so on a core that has
# i8mm, but not on one that does not.
#
# Workload is agentic tool-calling (strict JSON output), which is what the Cloud
# AI track names and what speculative drafting later exploits.
#
# Usage: scripts/04_serve_bench.sh [--conc 1,2,4,8,16] [--requests 48]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONC="${SPECARM_CONC:-1,2,4,8,16}"
REQUESTS="${SPECARM_SERVE_REQUESTS:-48}"
MAX_TOKENS="${SPECARM_MAX_TOKENS:-96}"
PORT="${SPECARM_PORT:-8080}"
THREADS="${SPECARM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
CTX="${SPECARM_CTX:-4096}"

while [ $# -gt 0 ]; do
  case "$1" in
    --conc)     CONC="${2:?}";      shift 2 ;;
    --requests) REQUESTS="${2:?}";  shift 2 ;;
    --port)     PORT="${2:?}";      shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -f "$RESULTS_DIR/build.latest.json" ] || die "no build found — run scripts/01_build_llama.sh first"
have python3 || die "python3 is required for the load generator"

MODEL="$(cat "$RESULTS_DIR/model.path" 2>/dev/null || true)"
[ -s "${MODEL:-/nonexistent}" ] || MODEL="$("$SPECARM_ROOT/scripts/fetch_model.sh")"

SRV_KLEIDI="$(sed -n 's/.*"srv_kleidi": "\(.*\)",/\1/p' "$RESULTS_DIR/build.latest.json")"
SRV_BASE="$(sed   -n 's/.*"srv_base": "\(.*\)",/\1/p'   "$RESULTS_DIR/build.latest.json")"
[ -x "${SRV_KLEIDI:-/nonexistent}" ] || die "kleidi server not executable: ${SRV_KLEIDI:-<unset>} (rebuild with 01)"
[ -x "${SRV_BASE:-/nonexistent}"   ] || die "base server not executable: ${SRV_BASE:-<unset>} (rebuild with 01)"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MAX_CONC="$(echo "$CONC" | tr ',' '\n' | sort -n | tail -n1)"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    # Give it a moment to release the port before the next variant binds it.
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

run_variant() {
  local variant="$1" srv="$2"
  local log="$RESULTS_DIR/serve_${variant}_${STAMP}.server.log"

  log "[$variant] starting llama-server (threads=$THREADS, slots=$MAX_CONC, ctx=$CTX)"
  # -np must be at least the peak client concurrency, otherwise requests queue
  # behind each other and we would be measuring a queue, not batched inference.
  "$srv" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
         -t "$THREADS" -c "$CTX" -np "$MAX_CONC" \
         > "$log" 2>&1 &
  SERVER_PID=$!

  local conc
  IFS=',' read -ra CONC_ARR <<< "$CONC"
  for conc in "${CONC_ARR[@]}"; do
    local out="$RESULTS_DIR/serve_${variant}_c${conc}_${STAMP}.json"
    log "[$variant] concurrency=$conc, requests=$REQUESTS"
    if ! python3 "$SPECARM_ROOT/tools/loadgen.py" \
           --url "http://127.0.0.1:$PORT" \
           --concurrency "$conc" --requests "$REQUESTS" \
           --max-tokens "$MAX_TOKENS" \
           --label "${variant}_c${conc}" --out "$out" > /dev/null; then
      warn "[$variant] concurrency=$conc had failed requests — see $out"
      warn "A serving result containing failures is not a throughput number."
    fi
  done

  stop_server
  log "[$variant] server stopped"
}

run_variant kleidi "$SRV_KLEIDI"
run_variant base   "$SRV_BASE"

echo "$STAMP" > "$RESULTS_DIR/serve.latest"
log "serving benchmark complete — stamp $STAMP"
log "next: python3 tools/analyze_serve.py --stamp $STAMP"

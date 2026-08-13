#!/usr/bin/env bash
#
# await_and_report.sh — block until the sweep finishes, then produce the audit.
#
# WHY THIS EXISTS
# ---------------
# The threshold sweep takes hours and must not share the CPU with anything else,
# or its timings measure contention rather than configuration. So the steps that
# follow it cannot simply be launched in parallel — they have to wait.
#
# This waits for the sweep to release the machine, then runs the remaining
# pipeline in the only order that is valid:
#
#   1. correctness   needs an idle CPU and a free port, same as the sweep
#   2. analysis      needs correctness, because a FAILED threshold is excluded
#                    from optimum selection
#   3. audit tables  recomputed from the raw JSONL, not from the CSV
#
# It reports whatever it finds. If the sweep died without writing its output,
# that is the report.
#
# Usage: bash tools/await_and_report.sh [--skip-correctness]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS="${SPECARM_RESULTS:-results}"
TARGET="$RESULTS/tune_rag.json"
SERVER="./llama/llama-server.exe"
MODEL="./models/qwen1.5b.gguf"
SKIP_CORRECTNESS=0
[ "${1:-}" = "--skip-correctness" ] && SKIP_CORRECTNESS=1

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- 1. wait
say "waiting for the sweep to finish"
for i in $(seq 1 400); do          # 400 x 30s ~ 3.3 h ceiling
  if [ -f "$TARGET" ]; then
    say "sweep complete: $TARGET written"
    break
  fi
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python; then
    say "python exited WITHOUT writing $TARGET"
    echo "  logs present: $(ls "$RESULTS"/sweep_rag_sim*.log 2>/dev/null | wc -l)/21"
    echo "  tail of sweep output:"
    tail -20 "$RESULTS/sweep_rag_run.log" 2>/dev/null | sed 's/^/    /'
    echo
    echo "  Mechanism data survives in the logs; TTFT does not."
    echo "  Analysis will run on what exists and mark the rest missing."
    break
  fi
  [ $((i % 10)) -eq 1 ] && \
    echo "  $(date -u +%H:%M) · $(ls "$RESULTS"/sweep_rag_sim*.log 2>/dev/null | wc -l)/21 logs"
  sleep 30
done

# ------------------------------------------------------- 2. correctness
if [ "$SKIP_CORRECTNESS" = 0 ] && [ -f "$SERVER" ] && [ -f "$MODEL" ]; then
  say "correctness across thresholds (CPU is now free)"
  python tools/correctness.py --server "$SERVER" --model "$MODEL" \
      --thresholds 0.1,0.5,0.8,0.9 2>&1 | sed 's/^/  /'
else
  say "skipping correctness"
fi

# ---------------------------------------------------------- 3. analysis
say "threshold sweep analysis"
python tools/analyze_threshold_sweep.py 2>&1 | sed 's/^/  /'

# ------------------------------------------------------- 4. audit tables
say "independent audit — recomputed from raw JSONL, not from the CSV"
python - <<'PY'
import json, csv, statistics, os, collections
RAW="results/raw/threshold_sweep.jsonl"
if not os.path.exists(RAW):
    print("  no raw data"); raise SystemExit

rows=[json.loads(l) for l in open(RAW,encoding="utf-8")]
shapes=sorted({r["workload_shape"] for r in rows})

print("\n  == INTEGRITY ==")
print("  Shape                Thr   Act  Valid  TTFT  Logs")
for s in shapes:
    for thr in sorted({r["threshold"] for r in rows if r["workload_shape"]==s}):
        g=[r for r in rows if r["workload_shape"]==s and r["threshold"]==thr]
        ok=[r for r in g if r["status"]=="OK"]
        tt=[r for r in g if r["ttft_ms"] is not None]
        ex=sum(1 for r in g if os.path.exists(r["log"]))
        print("  %-19s %-5s %4d %6d %5d %5s" % (s,thr,len(g),len(ok),len(tt),f"{ex}/{len(g)}"))
d=[k for k,v in collections.Counter(r["experiment_id"] for r in rows).items() if v>1]
print("  duplicate ids:", d or "none")

print("\n  == JOIN ==")
srcs=collections.Counter(r["ttft_source"] for r in rows)
for k,v in srcs.items(): print(f"  {v:>3} rows: {k}")

def pctl(v,q):
    s=sorted(v)
    if len(s)==1: return s[0]
    i=q*(len(s)-1); lo,hi=int(i),min(int(i)+1,len(s)-1)
    return s[lo]+(s[hi]-s[lo])*(i-lo)

for s in shapes:
    print(f"\n  == {s} · TTFT (ms) ==")
    print("  %6s %3s %10s %10s %8s %10s %10s %10s" % ("thr","n","median","mean","sd","p95","min","max"))
    for thr in sorted({r["threshold"] for r in rows if r["workload_shape"]==s}):
        v=[r["ttft_ms"] for r in rows if r["workload_shape"]==s and r["threshold"]==thr and r["ttft_ms"] is not None]
        if not v: print("  %6s %3d %10s" % (thr,0,"NO TTFT DATA")); continue
        sd=statistics.stdev(v) if len(v)>1 else 0.0
        print("  %6s %3d %10.1f %10.1f %8.1f %10.1f %10.1f %10.1f" %
              (thr,len(v),statistics.median(v),statistics.fmean(v),sd,pctl(v,.95),min(v),max(v)))

    print(f"\n  == {s} · MECHANISM ==")
    print("  %6s %3s %12s %10s %12s %10s" % ("thr","n","recomp med","recomp max","chosen sim","requests"))
    for thr in sorted({r["threshold"] for r in rows if r["workload_shape"]==s}):
        g=[r for r in rows if r["workload_shape"]==s and r["threshold"]==thr and r["status"]=="OK"]
        if not g: continue
        rc=[r["recomputed_prompt_tokens"] for r in g if r["recomputed_prompt_tokens"] is not None]
        mx=[r["max_recomputed_tokens"] for r in g if r["max_recomputed_tokens"] is not None]
        si=[r["chosen_slot_similarity"] for r in g if r["chosen_slot_similarity"] is not None]
        rq=[r["requests"] for r in g if r["requests"] is not None]
        print("  %6s %3d %12s %10s %12s %10s" % (thr,len(g),
              int(statistics.median(rc)) if rc else "-",
              max(mx) if mx else "-",
              f"{statistics.median(si):.3f}" if si else "-",
              int(statistics.median(rq)) if rq else "-"))

    v={thr:[r["ttft_ms"] for r in rows if r["workload_shape"]==s and r["threshold"]==thr and r["ttft_ms"] is not None]
       for thr in sorted({r["threshold"] for r in rows if r["workload_shape"]==s})}
    med={k:statistics.median(x) for k,x in v.items() if x}
    if len(med)>1:
        b=min(med,key=med.get); w=max(med,key=med.get)
        band=[k for k,x in med.items() if x<=med[b]*1.10]
        print(f"\n  == {s} · SENSITIVITY ==")
        print(f"  best {b} = {med[b]:.1f} ms · worst {w} = {med[w]:.1f} ms")
        print(f"  worst/best = {med[w]/med[b]:.2f}x · spread = {(med[w]-med[b])/med[b]*100:.0f}%")
        print(f"  within 10% band: {sorted(band)}  -> "
              f"{'UNIQUE optimum' if len(band)==1 else 'PLATEAU/TIE, not a unique optimum'}")
    else:
        print(f"\n  == {s} · SENSITIVITY ==\n  insufficient TTFT data")
PY

say "artifacts"
for f in results/raw/threshold_sweep.jsonl results/aggregated/threshold_summary.csv \
         results/aggregated/summary.json results/correctness.json \
         docs/fig_ttft_vs_threshold.svg docs/fig_tokens_vs_threshold.svg; do
  if [ -f "$f" ]; then echo "  OK   $f ($(wc -c < "$f") bytes)"; else echo "  MISS $f"; fi
done
say "done"

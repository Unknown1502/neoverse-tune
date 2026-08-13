#!/usr/bin/env bash
#
# run_prefix_experiment.sh — where does threshold sensitivity emerge?
#
# THE QUESTION
# ------------
# The two-shape experiment established that one workload is highly sensitive to
# --slot-prompt-similarity (5.91x, recomputation 220 -> 45) and another is
# completely insensitive (recomputation flat at 25 across the whole range). It
# did NOT establish why, because the two shapes differed in two things at once:
# shared-prefix fraction AND total prompt length. Length independently changes
# prefill cost, so it is a confounder.
#
# This family holds total length constant at ~2650 characters and varies only
# the shared fraction: 0.19, 0.34, 0.49, 0.63. Those fill the gap between the
# two measured anchors at ~0.12 and ~0.72.
#
# PREDICTION, RECORDED BEFORE THE RUN
# -----------------------------------
# If inter-tenant similarity really is shared_preamble / total_prompt, then
# sensitivity should rise with the shared fraction, and each shape's cliff
# should sit near its own shared fraction rather than at a fixed value.
#
# FALSIFIER
# ---------
# If sensitivity does not vary across this family, the two-shape result was
# driven by something other than prefix sharing -- most likely the length
# confounder -- and the mechanism story in this repository is wrong.
#
# n=5 per point, which meets the project's own MIN_REPS, so these results can
# go through skeptic.py rather than being exploration only.
#
# Runtime ~3.3 h. Run it with nothing else on the machine.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SERVER="./llama/llama-server.exe"
MODEL="./models/qwen1.5b.gguf"
THRESHOLDS="0.1,0.3,0.5,0.7,0.9"
REPS=5
SHAPES="prefix20 prefix35 prefix50 prefix65"

say() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

for f in "$SERVER" "$MODEL"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

say "intermediate-prefix experiment"
echo "  shapes:     $SHAPES"
echo "  thresholds: $THRESHOLDS"
echo "  reps:       $REPS   (meets MIN_REPS, so skeptic.py can adjudicate)"
echo "  total:      $(( $(echo $SHAPES | wc -w) * 5 * REPS )) runs"

for shape in $SHAPES; do
  say "sweeping $shape"
  python tools/tune_similarity.py --sweep \
      --workload "$shape" \
      --server "$SERVER" --model "$MODEL" \
      --agents 4 --turns 4 --repeats "$REPS" \
      --thresholds "$THRESHOLDS" \
      --out "results/tune_${shape}.json" 2>&1 | sed 's/^/    /'
  # A failed shape must not silently look like a flat one. Say so and continue,
  # so the shapes that did work are still usable.
  if [ ! -f "results/tune_${shape}.json" ]; then
    say "WARNING: $shape produced no output — later analysis will mark it missing"
  fi
done

say "all sweeps finished — analysing"
python tools/analyze_threshold_sweep.py 2>&1 | sed 's/^/  /'

say "sensitivity vs shared fraction"
python - <<'PY'
import json, glob, os, re, statistics
rows=[]
for f in sorted(glob.glob("results/tune_prefix*.json")):
    shape=re.search(r"tune_(prefix\d+)\.json", f).group(1)
    d=json.load(open(f,encoding="utf-8"))
    med={r["threshold"]: r["warm_median_ms"] for r in d.get("sweep",[]) if r.get("warm_median_ms")}
    if len(med)<2: continue
    b=min(med,key=med.get); w=max(med,key=med.get)
    # RSD per threshold, so an unstable shape cannot masquerade as a sensitive one
    noisy=[r["threshold"] for r in d["sweep"]
           if len(r.get("samples",[]))>1
           and statistics.stdev(r["samples"])/statistics.fmean(r["samples"])*100 > 10]
    rows.append((shape, med[b], b, med[w], w, med[w]/med[b], noisy))

if not rows:
    print("  no completed shapes"); raise SystemExit
print("  %-10s %10s %6s %10s %6s %9s  %s" %
      ("shape","best ms","at","worst ms","at","worst/best","thresholds failing 10% noise gate"))
print("  " + "-"*94)
for s,bm,b,wm,w,r,noisy in rows:
    print("  %-10s %10.1f %6s %10.1f %6s %8.2fx  %s" % (s,bm,b,wm,w,r,noisy or "none"))
print()
print("  Anchors from the completed two-shape experiment:")
print("    low_prefix_reuse  (~0.12 shared)  worst/best 2.26x  but recomputation FLAT at 25")
print("    high_prefix_reuse (~0.72 shared)  worst/best 5.91x  recomputation 220 -> 45")
print()
print("  Read sensitivity from RECOMPUTATION, not from latency alone. A latency")
print("  spread on a shape whose recomputation never moves is machine noise, which")
print("  is exactly what the RAG shape turned out to be.")
PY

say "done"

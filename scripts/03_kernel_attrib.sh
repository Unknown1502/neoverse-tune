#!/usr/bin/env bash
#
# 03_kernel_attrib.sh — MECHANISM PROOF.
#
# A knee in the throughput curve is correlation. This script supplies causation:
# KleidiAI microkernel symbols encode the ISA path they were compiled for, e.g.
#
#   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod
#   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm
#
# so whichever kai_* symbol is hot literally names the kernel that executed. We
# sample at a low batch size and a high one; if the thesis holds, the hot symbol
# changes from a *dotprod* kernel to an *i8mm* kernel between them.
#
# perf is frequently unavailable in virtualized CI. That is expected, not fatal:
# we degrade to a capability inventory (which kernels are *present*) and say so
# plainly rather than inventing occupancy numbers.
#
# Usage: scripts/03_kernel_attrib.sh [--batches 1,16]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BATCHES="${SPECARM_ATTRIB_BATCHES:-1,16}"
[ "${1:-}" = "--batches" ] && BATCHES="${2:?}"

[ -f "$RESULTS_DIR/build.latest.json" ] || die "no build found — run scripts/01_build_llama.sh first"
BIN="$(sed -n 's/.*"bin_kleidi": "\(.*\)",/\1/p' "$RESULTS_DIR/build.latest.json")"
[ -x "$BIN" ] || die "kleidi binary not executable: $BIN"

MODEL="$(cat "$RESULTS_DIR/model.path" 2>/dev/null || true)"
[ -s "${MODEL:-/nonexistent}" ] || MODEL="$("$SPECARM_ROOT/scripts/fetch_model.sh")"

THREADS="${SPECARM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RESULTS_DIR/kernels_${STAMP}.json"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------- capability inventory
# Always available, no privileges needed: which kai_* kernels were even linked in?
INVENTORY="$WORK/inventory.txt"
: > "$INVENTORY"
if have nm; then
  nm "$BIN" 2>/dev/null | awk '{print $NF}' | grep '^kai_' | sort -u > "$INVENTORY" || true
fi
log "kai_* kernels linked into binary: $(wc -l < "$INVENTORY")"

# ---------------------------------------------------------------- perf viability
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unavailable)"
PERF_OK=0
if have perf && [ "$PARANOID" != unavailable ] && [ "$PARANOID" -le 2 ] 2>/dev/null; then
  # Prove it can actually record here, not just that the binary exists.
  if perf record -q -o "$WORK/probe.data" -- true >/dev/null 2>&1; then
    PERF_OK=1
  fi
fi

if [ "$PERF_OK" -eq 0 ]; then
  warn "perf unusable (paranoid=$PARANOID). Falling back to capability inventory."
  warn "This records which kernels EXIST, not which RAN — it is not mechanism proof."
fi

# ---------------------------------------------------------------- per-batch capture
profile_batch() {
  local n="$1"
  local data="$WORK/perf_$n.data"
  local rpt="$WORK/report_$n.txt"

  if [ "$PERF_OK" -eq 0 ]; then
    echo "" > "$rpt"
    echo "$rpt"
    return
  fi

  log "profiling batch n=$n"
  perf record -q -F 999 -g -o "$data" -- \
    "$BIN" -m "$MODEL" -p "$n" -n 0 -r 3 -t "$THREADS" -o json \
    >/dev/null 2>&1 || warn "perf record returned nonzero for n=$n (continuing)"

  if [ -s "$data" ]; then
    perf report -i "$data" --stdio --no-children --sort symbol --percent-limit 0.01 \
      > "$rpt" 2>/dev/null || : > "$rpt"
  else
    : > "$rpt"
  fi
  echo "$rpt"
}

declare -A REPORTS
IFS=',' read -ra BATCH_ARR <<< "$BATCHES"
for n in "${BATCH_ARR[@]}"; do
  REPORTS["$n"]="$(profile_batch "$n")"
done

# ---------------------------------------------------------------- classify + emit
# Python does the parsing: tolerant of perf's column layout drifting between
# versions, and it keeps the ISA classification rules in one readable place.
PY_BATCHES="$BATCHES" PY_OUT="$OUT" PY_INV="$INVENTORY" PY_PERF_OK="$PERF_OK" \
PY_WORK="$WORK" PY_PARANOID="$PARANOID" python3 <<'PYEOF'
import json, os, re

work      = os.environ["PY_WORK"]
batches   = [b.strip() for b in os.environ["PY_BATCHES"].split(",") if b.strip()]
perf_ok   = os.environ["PY_PERF_OK"] == "1"
inventory = os.environ["PY_INV"]

def isa_of(sym: str) -> str:
    """Classify a KleidiAI microkernel by the ISA encoded in its name.
    Order matters: i8mm/sme names also contain 'neon'."""
    s = sym.lower()
    if "sme2" in s:    return "sme2"
    if "sme" in s:     return "sme"
    if "i8mm" in s:    return "i8mm"
    if "dotprod" in s: return "dotprod"
    if "neon" in s:    return "neon"
    return "other"

# perf --stdio lines look like:  "    12.34%  [.] kai_run_matmul_..."
LINE = re.compile(r"^\s*([0-9]+\.[0-9]+)%\s+(?:\[[.k]\]\s+)?(\S+)")

def parse(path):
    rows = []
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                m = LINE.match(line)
                if m:
                    rows.append((float(m.group(1)), m.group(2)))
    except FileNotFoundError:
        pass
    return rows

per_batch = {}
for n in batches:
    rows = parse(os.path.join(work, f"report_{n}.txt"))
    kai  = [(p, s) for p, s in rows if s.startswith("kai_")]
    buckets = {}
    for pct, sym in kai:
        buckets[isa_of(sym)] = round(buckets.get(isa_of(sym), 0.0) + pct, 4)
    kai.sort(reverse=True)
    per_batch[n] = {
        "measured":            perf_ok and bool(rows),
        "kai_cycles_pct":      round(sum(p for p, _ in kai), 4),
        "isa_breakdown_pct":   dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
        "hottest_kai_symbol":  kai[0][1] if kai else None,
        "hottest_kai_isa":     isa_of(kai[0][1]) if kai else None,
        "top_symbols_overall": [{"pct": p, "symbol": s} for p, s in rows[:10]],
    }

linked = []
try:
    with open(inventory) as fh:
        linked = [l.strip() for l in fh if l.strip()]
except FileNotFoundError:
    pass

linked_by_isa = {}
for s in linked:
    linked_by_isa.setdefault(isa_of(s), []).append(s)

# The headline: did the hot ISA actually change across batch sizes?
isas = [per_batch[n]["hottest_kai_isa"] for n in batches if per_batch[n]["measured"]]
if not isas:
    switch = None          # unknown — we could not measure
elif len(set(isas)) > 1:
    switch = True          # kernel selection changed with batch size
else:
    switch = False         # same kernel throughout

out = {
    "schema": "specarm.kernels/1",
    "perf_available": perf_ok,
    "perf_event_paranoid": os.environ["PY_PARANOID"],
    "evidence_class": "execution" if perf_ok and isas else "capability_only",
    "kernel_switch_observed": switch,
    "batches": per_batch,
    "linked_kernels_by_isa": {k: sorted(v) for k, v in sorted(linked_by_isa.items())},
    "linked_kernel_count": len(linked),
}
with open(os.environ["PY_OUT"], "w") as fh:
    json.dump(out, fh, indent=2)

print()
print(f"  evidence class ...... {out['evidence_class']}")
print(f"  kernels linked ...... {len(linked)}  ({', '.join(f'{k}:{len(v)}' for k, v in sorted(linked_by_isa.items()))})")
for n in batches:
    b = per_batch[n]
    if b["measured"]:
        print(f"  batch n={n:<4} ....... hot={b['hottest_kai_isa'] or 'none'}  "
              f"kai={b['kai_cycles_pct']}%  sym={(b['hottest_kai_symbol'] or '-')[:64]}")
    else:
        print(f"  batch n={n:<4} ....... not measured (perf unavailable)")
print()
if switch is True:
    print("  >>> KERNEL SWITCH OBSERVED across batch sizes.")
elif switch is False:
    print("  >>> No kernel switch: the same ISA path served every batch size.")
else:
    print("  >>> Inconclusive: no execution evidence captured on this host.")
PYEOF

cp "$OUT" "$RESULTS_DIR/kernels.latest.json"
log "wrote $OUT"

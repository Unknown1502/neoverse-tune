#!/usr/bin/env python3
"""
analyze.py — turn raw llama-bench output into an adjudicated result table.

Two jobs:

  1. Find the knee. For each prompt-batch size N, report throughput. If decode
     (N=1) is stuck on a GEMV/dotprod path and larger N reaches an i8mm GEMM
     path, there should be a disproportionate jump at some small N.

  2. Adjudicate. Every KleidiAI-vs-baseline comparison gets a verdict of
     VERIFIED / UNCERTAIN / REJECTED under thresholds fixed in advance (see
     docs/methodology.md). Rejected rows are printed, not hidden — a harness
     that only reports its wins is a harness nobody should believe.

Standard library only, so it runs anywhere an Arm box has python3.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))

# --- Pre-registered thresholds. Changing these after seeing results is cheating.
MIN_EFFECT_PCT = 5.0   # below this, a difference is not worth claiming
MIN_REPS = 5           # fewer samples than this cannot support a VERIFIED
MAX_RSD_PCT = 10.0     # relative stdev above this means the host was too noisy

# Two-sided t critical values at 95% for small samples, by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131,
        20: 2.086, 25: 2.060, 30: 2.042}


def t_crit(n: int) -> float:
    """95% two-sided critical value for n samples; approaches 1.96 as n grows."""
    df = max(n - 1, 1)
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if df < k:
            return _T95[k]
    return 1.96


class Stat:
    """Summary of one measurement cell."""

    def __init__(self, samples: list[float]):
        self.samples = [s for s in samples if s and s > 0]
        self.n = len(self.samples)
        self.mean = statistics.fmean(self.samples) if self.n else 0.0
        self.median = statistics.median(self.samples) if self.n else 0.0
        self.sd = statistics.stdev(self.samples) if self.n > 1 else 0.0
        self.hw = (t_crit(self.n) * self.sd / math.sqrt(self.n)) if self.n > 1 else 0.0

    @property
    def lo(self) -> float:
        return self.mean - self.hw

    @property
    def hi(self) -> float:
        return self.mean + self.hw

    @property
    def rsd(self) -> float:
        """Relative standard deviation, as a percentage."""
        return (self.sd / self.mean * 100.0) if self.mean else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "mean": round(self.mean, 4), "median": round(self.median, 4),
                "stdev": round(self.sd, 4), "ci95_lo": round(self.lo, 4),
                "ci95_hi": round(self.hi, 4), "rsd_pct": round(self.rsd, 3)}


def verdict(base: Stat, cand: Stat) -> tuple[str, str]:
    """Adjudicate candidate against baseline. Higher throughput is better."""
    # Distinguish "we measured nothing" from "we measured no gain". Conflating
    # them would let a failed run masquerade as a negative result.
    if not base.n or not cand.n:
        return "UNCERTAIN", "no valid samples — measurement failed, not a null result"
    if base.n < MIN_REPS or cand.n < MIN_REPS:
        return "UNCERTAIN", f"only {min(base.n, cand.n)} reps (need {MIN_REPS})"
    if base.rsd > MAX_RSD_PCT or cand.rsd > MAX_RSD_PCT:
        return "UNCERTAIN", f"host too noisy (rsd {max(base.rsd, cand.rsd):.1f}% > {MAX_RSD_PCT}%)"
    if not base.mean:
        return "UNCERTAIN", "baseline measured zero — measurement failed"

    effect = (cand.mean - base.mean) / base.mean * 100.0
    if effect <= 0:
        return "REJECTED", f"no improvement ({effect:+.1f}%)"
    if cand.lo <= base.hi:
        return "UNCERTAIN", f"{effect:+.1f}% but 95% CIs overlap"
    if effect < MIN_EFFECT_PCT:
        return "UNCERTAIN", f"{effect:+.1f}% is below the {MIN_EFFECT_PCT}% floor"
    return "VERIFIED", f"{effect:+.1f}%, CIs disjoint"


def load_bench(path: str) -> list[dict[str, Any]]:
    """Read a llama-bench -o json file, tolerating wrapper shapes."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! could not read {os.path.basename(path)}: {exc}", file=sys.stderr)
        return []
    if isinstance(data, dict):
        for key in ("results", "data", "benchmarks"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data if isinstance(data, list) else []


def samples_of(row: dict[str, Any]) -> list[float]:
    """Per-repetition throughputs; fall back to the average if not exposed."""
    for key in ("samples_ts", "samples", "ts_samples"):
        v = row.get(key)
        if isinstance(v, list) and v:
            return [float(x) for x in v]
    for key in ("avg_ts", "t/s", "ts", "avg_ns"):
        if row.get(key):
            return [float(row[key])]
    return []


def cells(rows: list[dict[str, Any]]) -> dict[tuple[int, int], Stat]:
    """Index measurements by (n_prompt, n_gen)."""
    out: dict[tuple[int, int], Stat] = {}
    for row in rows:
        try:
            key = (int(row.get("n_prompt", 0) or 0), int(row.get("n_gen", 0) or 0))
        except (TypeError, ValueError):
            continue
        s = samples_of(row)
        if s:
            out[key] = Stat(s)
    return out


def newest(pattern: str, stamp: str | None) -> str | None:
    pat = pattern.replace("*", stamp) if stamp else pattern
    hits = sorted(glob.glob(os.path.join(RESULTS, pat)))
    return hits[-1] if hits else None


def find_knee(points: list[tuple[int, Stat]]) -> dict[str, Any] | None:
    """Largest step-up in throughput between adjacent batch sizes.

    A GEMV->GEMM kernel switch should show up as a disproportionate jump. This
    locates the candidate; it does NOT prove a switch happened. Only the symbol
    evidence from 03_kernel_attrib.sh can do that.
    """
    if len(points) < 3:
        return None
    best = None
    for (n0, s0), (n1, s1) in zip(points, points[1:]):
        if not s0.mean:
            continue
        gain = (s1.mean - s0.mean) / s0.mean * 100.0
        if best is None or gain > best["gain_pct"]:
            best = {"from_batch": n0, "to_batch": n1, "gain_pct": round(gain, 2),
                    "from_ts": round(s0.mean, 2), "to_ts": round(s1.mean, 2)}
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", help="run stamp; defaults to the latest run")
    ap.add_argument("--out", help="markdown report path")
    args = ap.parse_args()

    stamp = args.stamp
    if not stamp:
        latest = os.path.join(RESULTS, "run.latest")
        if os.path.exists(latest):
            stamp = open(latest).read().strip()

    pp_k = newest("sweep_kleidi_pp_*.json", stamp)
    pp_b = newest("sweep_base_pp_*.json", stamp)
    tg_k = newest("sweep_kleidi_tg_*.json", stamp)
    tg_b = newest("sweep_base_tg_*.json", stamp)

    if not pp_k or not pp_b:
        print("No sweep results found. Run scripts/02_batch_sweep.sh first.", file=sys.stderr)
        return 1

    ck, cb = cells(load_bench(pp_k)), cells(load_bench(pp_b))
    tk, tb = cells(load_bench(tg_k)) if tg_k else {}, cells(load_bench(tg_b)) if tg_b else {}

    env = {}
    env_path = os.path.join(RESULTS, "env.latest.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))

    kern = {}
    kern_path = os.path.join(RESULTS, "kernels.latest.json")
    if os.path.exists(kern_path):
        kern = json.load(open(kern_path))

    batches = sorted({p for (p, g) in ck if g == 0} & {p for (p, g) in cb if g == 0})
    points = [(n, ck[(n, 0)]) for n in batches]

    lines: list[str] = []
    core = env.get("core", "unknown")
    lines.append(f"# SpecArm crux result — {core}")
    lines.append("")
    lines.append(f"- **Core**: `{core}` ({env.get('nproc', '?')} threads, virt=`{env.get('virtualization','?')}`)")
    lines.append(f"- **i8mm**: `{env.get('features', {}).get('i8mm')}` · "
                 f"**sve2**: `{env.get('features', {}).get('sve2')}` "
                 f"(vector bits: `{env.get('sve_vector_bits')}`) · "
                 f"**bf16**: `{env.get('features', {}).get('bf16')}`")
    lines.append(f"- **Crux role**: `{env.get('crux_role','?')}`")
    lines.append(f"- **llama.cpp**: `{(env.get('llama_cpp_sha') or 'unknown')[:12]}`")
    lines.append("")
    if env.get("crux_role") == "control":
        lines.append("> ⚠️ This core has **no i8mm**. There is no SMMLA kernel to wake up here, "
                     "so a null result is expected and proves nothing about the thesis. "
                     "These numbers are the control row only.")
        lines.append("")

    lines.append("## Batch sweep — KleidiAI vs baseline")
    lines.append("")
    lines.append("`-p N` = one forward pass over N tokens = the matmul shape of verifying "
                 "N speculative draft tokens. Throughput in tokens/sec, mean ± 95% CI.")
    lines.append("")
    lines.append("| batch N | baseline t/s | kleidi t/s | Δ | verdict | note |")
    lines.append("|--:|--:|--:|--:|:--|:--|")

    analysis: dict[str, Any] = {"schema": "specarm.analysis/1", "stamp": stamp,
                                "core": core, "batches": {}, "verdicts": {}}

    for n in batches:
        b, k = cb[(n, 0)], ck[(n, 0)]
        v, why = verdict(b, k)
        delta = ((k.mean - b.mean) / b.mean * 100.0) if b.mean else 0.0
        icon = {"VERIFIED": "✅", "UNCERTAIN": "⚠️", "REJECTED": "❌"}[v]
        lines.append(f"| {n} | {b.mean:.1f} ± {b.hw:.1f} | {k.mean:.1f} ± {k.hw:.1f} "
                     f"| {delta:+.1f}% | {icon} {v} | {why} |")
        analysis["batches"][str(n)] = {"base": b.as_dict(), "kleidi": k.as_dict(),
                                       "delta_pct": round(delta, 3)}
        analysis["verdicts"][str(n)] = {"verdict": v, "reason": why}

    # Single-token decode baseline: the regime the whole thesis is about.
    if tk and tb:
        gk = next((s for (p, g), s in tk.items() if g > 0), None)
        gb = next((s for (p, g), s in tb.items() if g > 0), None)
        if gk and gb:
            v, why = verdict(gb, gk)
            d = ((gk.mean - gb.mean) / gb.mean * 100.0) if gb.mean else 0.0
            icon = {"VERIFIED": "✅", "UNCERTAIN": "⚠️", "REJECTED": "❌"}[v]
            lines.append(f"| decode (tg) | {gb.mean:.1f} ± {gb.hw:.1f} | {gk.mean:.1f} ± {gk.hw:.1f} "
                         f"| {d:+.1f}% | {icon} {v} | {why} |")
            analysis["decode"] = {"base": gb.as_dict(), "kleidi": gk.as_dict(),
                                  "delta_pct": round(d, 3), "verdict": v, "reason": why}

    lines.append("")

    knee = find_knee(points)
    analysis["knee"] = knee
    lines.append("## Knee")
    lines.append("")
    if knee:
        lines.append(f"Largest step-up: **N={knee['from_batch']} → N={knee['to_batch']}**, "
                     f"{knee['from_ts']:.1f} → {knee['to_ts']:.1f} t/s "
                     f"(**{knee['gain_pct']:+.1f}%**).")
        lines.append("")
        lines.append("A knee is *consistent with* a GEMV→GEMM kernel switch but does not "
                     "establish one — throughput also rises simply from amortizing weight "
                     "streaming. See the mechanism section.")
    else:
        lines.append("Not enough batch points to locate a knee.")
    lines.append("")

    lines.append("## Mechanism")
    lines.append("")
    if kern:
        ev = kern.get("evidence_class")
        lines.append(f"- Evidence class: **{ev}**")
        if ev == "capability_only":
            lines.append("- `perf` was unavailable, so we can report which kernels were "
                         "**linked**, not which **ran**. This is not mechanism proof.")
        by_isa = kern.get("linked_kernels_by_isa", {})
        if by_isa:
            lines.append(f"- Kernels linked by ISA: " +
                         ", ".join(f"`{k}`×{len(v)}" for k, v in sorted(by_isa.items())))
        for n, b in sorted(kern.get("batches", {}).items(), key=lambda kv: int(kv[0])):
            if b.get("measured"):
                lines.append(f"- batch **N={n}** → hot ISA `{b.get('hottest_kai_isa')}`, "
                             f"{b.get('kai_cycles_pct')}% of cycles in kai_* "
                             f"(`{(b.get('hottest_kai_symbol') or '-')[:72]}`)")
            else:
                lines.append(f"- batch **N={n}** → not measured")
        sw = kern.get("kernel_switch_observed")
        lines.append("")
        if sw is True:
            lines.append("**KERNEL SWITCH OBSERVED.** The hot KleidiAI ISA path changes with "
                         "batch size — decode and verification do not run the same kernel.")
        elif sw is False:
            lines.append("**No kernel switch observed.** The same ISA path served every batch "
                         "size. If this holds up, the finding inverts: KleidiAI leaves i8mm on "
                         "the table for small-batch work, and *that* is the upstream gap.")
        else:
            lines.append("**Inconclusive** — no execution evidence on this host. Re-run "
                         "kernel attribution somewhere `perf` is permitted.")
    else:
        lines.append("No kernel attribution data. Run `scripts/03_kernel_attrib.sh`.")
    lines.append("")

    counts: dict[str, int] = {}
    for item in analysis["verdicts"].values():
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    lines.append("## Skeptic summary")
    lines.append("")
    lines.append("| verdict | count |")
    lines.append("|:--|--:|")
    for v in ("VERIFIED", "UNCERTAIN", "REJECTED"):
        lines.append(f"| {v} | {counts.get(v, 0)} |")
    lines.append("")
    lines.append(f"Thresholds (pre-registered): min effect {MIN_EFFECT_PCT}%, "
                 f"min reps {MIN_REPS}, max RSD {MAX_RSD_PCT}%.")
    lines.append("")

    md = "\n".join(lines)
    out_md = args.out or os.path.join(RESULTS, f"report_{stamp or 'latest'}.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(RESULTS, "analysis.latest.json"), "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    print(md)
    print(f"\n[analyze] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

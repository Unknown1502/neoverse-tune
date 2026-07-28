#!/usr/bin/env python3
"""
analyze_serve.py — adjudicate the serving benchmark.

Answers the Cloud AI form of the question: as client concurrency rises, a server
batches more requests into each forward pass. If the i8mm thesis holds, the
KleidiAI advantage should *widen* with concurrency, because higher concurrency
reaches the batch regime where i8mm matmul kernels become selectable.

Two distinct kinds of number are reported, and they are not mixed:

  Latency (TTFT, TPOT)   — one sample per request, so real confidence intervals
                           and full Skeptic adjudication.
  Throughput (tok/s)     — ONE observation per run. A single observation has no
                           confidence interval, so it is reported as an
                           observation and never carries a VERIFIED verdict.

Conflating those two is the most common way serving benchmarks overclaim.

Run:  python3 tools/analyze_serve.py [--stamp STAMP]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))

# Reuse the adjudicator rather than reimplementing thresholds: one definition of
# VERIFIED in the codebase, tested in one place.
_spec = importlib.util.spec_from_file_location("analyze", os.path.join(HERE, "analyze.py"))
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)
Stat, verdict = analyze.Stat, analyze.verdict

FNAME = re.compile(r"serve_(?P<variant>kleidi|base)_c(?P<conc>\d+)_(?P<stamp>[0-9TZ]+)\.json$")


def load_runs(stamp: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "serve_*_c*_*.json"))):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        if stamp and m.group("stamp") != stamp:
            continue
        try:
            with open(path) as fh:
                out[(m.group("variant"), int(m.group("conc")))] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping {os.path.basename(path)}: {exc}", file=sys.stderr)
    return out


def samples(run: dict[str, Any], metric: str) -> list[float]:
    raw = (run.get("raw") or {}).get(metric)
    if isinstance(raw, list) and raw:
        return [float(x) for x in raw]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", help="serving run stamp; defaults to latest")
    ap.add_argument("--out", help="markdown report path")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    stamp = args.stamp
    if not stamp:
        latest = os.path.join(RESULTS, "serve.latest")
        if os.path.exists(latest):
            stamp = open(latest).read().strip()

    runs = load_runs(stamp)
    if not runs:
        print("No serving results found. Run scripts/04_serve_bench.sh first.",
              file=sys.stderr)
        return 1

    concurrencies = sorted({c for (_v, c) in runs})
    env: dict[str, Any] = {}
    env_path = os.path.join(RESULTS, "env.latest.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))

    lines: list[str] = []
    core = env.get("core", "unknown")
    lines.append(f"# SpecArm serving result — {core}")
    lines.append("")
    lines.append(f"- **Core**: `{core}` · i8mm `{env.get('features', {}).get('i8mm')}` "
                 f"· role `{env.get('crux_role', '?')}`")
    lines.append(f"- **Workload**: agentic tool-call, strict JSON output")
    lines.append("")

    analysis: dict[str, Any] = {"schema": "specarm.serve_analysis/1",
                                "stamp": stamp, "core": core,
                                "concurrency": {}, "integrity": {}}

    # --- integrity gate ----------------------------------------------------
    bad = [(v, c) for (v, c), r in runs.items() if r.get("requests_failed", 0)]
    if bad:
        lines.append("> ⚠️ **Runs with failed requests are excluded.** A throughput "
                     "number measured while requests were erroring is not throughput.")
        lines.append("")
        for v, c in sorted(bad):
            lines.append(f">   - `{v}` @ concurrency {c}: "
                         f"{runs[(v, c)]['requests_failed']} failed")
        lines.append("")
    analysis["integrity"]["excluded_runs"] = [f"{v}_c{c}" for v, c in sorted(bad)]

    ok_runs = {k: r for k, r in runs.items() if not r.get("requests_failed", 0)}

    # --- throughput: single observations -----------------------------------
    lines.append("## Aggregate throughput (single observation per cell)")
    lines.append("")
    lines.append("One run per cell, so **no confidence interval and no verdict**. "
                 "Read the trend, not the third digit.")
    lines.append("")
    lines.append("| concurrency | base tok/s | kleidi tok/s | Δ |")
    lines.append("|--:|--:|--:|--:|")
    for c in concurrencies:
        b, k = ok_runs.get(("base", c)), ok_runs.get(("kleidi", c))
        if not b or not k:
            continue
        bt = b.get("output_tokens_per_sec", 0.0)
        kt = k.get("output_tokens_per_sec", 0.0)
        d = ((kt - bt) / bt * 100.0) if bt else 0.0
        lines.append(f"| {c} | {bt:.1f} | {kt:.1f} | {d:+.1f}% |")
        analysis["concurrency"].setdefault(str(c), {})["throughput"] = {
            "base": bt, "kleidi": kt, "delta_pct": round(d, 3),
            "note": "single observation, no CI",
        }
    lines.append("")

    # --- latency: adjudicated ----------------------------------------------
    # Lower latency is better, so the comparison is inverted before it reaches
    # verdict(), which is written for higher-is-better.
    for metric, label in (("ttft_ms", "Time to first token (prefill / GEMM regime)"),
                          ("tpot_ms", "Time per output token (decode / GEMV regime)")):
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| concurrency | base p50 | kleidi p50 | Δ | verdict | note |")
        lines.append("|--:|--:|--:|--:|:--|:--|")
        for c in concurrencies:
            b, k = ok_runs.get(("base", c)), ok_runs.get(("kleidi", c))
            if not b or not k:
                continue
            bs, ks = samples(b, metric), samples(k, metric)
            if not bs or not ks:
                lines.append(f"| {c} | – | – | – | ⚠️ UNCERTAIN | no raw samples |")
                continue
            # Invert: speed = 1/latency, so higher is better.
            b_stat = Stat([1000.0 / x for x in bs if x > 0])
            k_stat = Stat([1000.0 / x for x in ks if x > 0])
            v, why = verdict(b_stat, k_stat)
            bp = b[metric]["p50"]
            kp = k[metric]["p50"]
            d = ((bp - kp) / bp * 100.0) if bp else 0.0
            icon = {"VERIFIED": "✅", "UNCERTAIN": "⚠️", "REJECTED": "❌"}[v]
            lines.append(f"| {c} | {bp:.1f} ms | {kp:.1f} ms | {d:+.1f}% faster "
                         f"| {icon} {v} | {why} |")
            analysis["concurrency"].setdefault(str(c), {})[metric] = {
                "base_p50_ms": bp, "kleidi_p50_ms": kp,
                "delta_pct_faster": round(d, 3), "verdict": v, "reason": why,
            }
        lines.append("")

    # --- the thesis test ---------------------------------------------------
    lines.append("## Does the advantage widen with concurrency?")
    lines.append("")
    deltas = []
    for c in concurrencies:
        cell = analysis["concurrency"].get(str(c), {}).get("throughput")
        if cell:
            deltas.append((c, cell["delta_pct"]))
    if len(deltas) >= 2:
        lo_c, lo_d = deltas[0]
        hi_c, hi_d = deltas[-1]
        widened = hi_d - lo_d
        analysis["widening_pct_points"] = round(widened, 3)
        lines.append(f"KleidiAI advantage at concurrency {lo_c}: **{lo_d:+.1f}%** → "
                     f"at concurrency {hi_c}: **{hi_d:+.1f}%** "
                     f"(change: **{widened:+.1f}** percentage points).")
        lines.append("")
        if widened > 5:
            lines.append("Consistent with reaching a batch regime that unlocks better "
                         "kernels at higher concurrency. **Not proof** — confirm against "
                         "the kernel attribution and the direct probe "
                         "(`tools/analyze_probe.py`).")
        elif widened < -5:
            lines.append("The advantage *narrows* with concurrency — the opposite of the "
                         "thesis. Report it as such.")
        else:
            lines.append("Flat. The KleidiAI advantage does not depend meaningfully on "
                         "concurrency in this range.")
    else:
        lines.append("Not enough concurrency levels to assess a trend.")
    lines.append("")
    lines.append("> Throughput deltas here are single observations. Treat this section "
                 "as a direction to investigate, not a measured effect.")
    lines.append("")

    md = "\n".join(lines)
    out_md = args.out or os.path.join(RESULTS, f"serve_report_{stamp or 'latest'}.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(RESULTS, "serve_analysis.latest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    print(md)
    print(f"\n[analyze_serve] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

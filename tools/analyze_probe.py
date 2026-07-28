#!/usr/bin/env python3
"""
analyze_probe.py — turn the direct microkernel probe into the crossover result.

This is the measurement with nothing in between. It answers two questions:

  1. STRUCTURAL (exact, no statistics involved). Each variant reports its row
     granularity `mr` via a plain API call. A kernel with mr=8 computes 8 rows
     per macro-tile, so at M=1 it does 8 rows of work to produce 1. That is not
     an estimate — it is the kernel's declared geometry.

  2. EMPIRICAL. At which M does an i8mm variant overtake the best dotprod
     variant? Below that crossover, decode cannot benefit from i8mm no matter
     how the surrounding code is tuned. Above it, speculative decoding or
     serving concurrency can reach it.

Run:  python3 tools/analyze_probe.py [--input FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))


def best_by_isa(rows: list[dict[str, Any]], m: int, isa: str) -> dict[str, Any] | None:
    cands = [r for r in rows
             if r.get("m") == m and r.get("isa") == isa and r.get("ok")
             and r.get("gflops", 0) > 0]
    return max(cands, key=lambda r: r["gflops"]) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="probe JSON; defaults to results/kai_probe.latest.json")
    ap.add_argument("--out", help="markdown report path")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    path = args.input or os.path.join(RESULTS, "kai_probe.latest.json")
    if not os.path.exists(path):
        print(f"No probe results at {path}. Run scripts/05_kai_probe.sh first.",
              file=sys.stderr)
        return 1

    with open(path) as fh:
        data = json.load(fh)

    variants = data.get("variants", [])
    rows = data.get("measurements", [])
    if not rows:
        print("Probe file contains no measurements.", file=sys.stderr)
        return 1

    ms = sorted({r["m"] for r in rows if "m" in r})
    out: dict[str, Any] = {"schema": "specarm.probe_analysis/1",
                           "n": data.get("n"), "k": data.get("k"),
                           "kleidiai_sha": data.get("kleidiai_sha")}

    lines: list[str] = []
    lines.append("# SpecArm — direct microkernel probe")
    lines.append("")
    lines.append(f"Matmul shape: **n={data.get('n')}, k={data.get('k')}**, "
                 f"{data.get('reps')} reps, median reported. "
                 f"KleidiAI `{(data.get('kleidiai_sha') or 'unknown')[:12]}`.")
    lines.append("")
    if data.get("mismatches"):
        lines.append(f"> ❌ **{data['mismatches']} variant mismatches.** Outputs "
                     "disagreed, so packing is wrong and these timings mean nothing.")
        lines.append("")

    # --- 1. structural ------------------------------------------------------
    lines.append("## 1. Kernel geometry — the structural finding")
    lines.append("")
    lines.append("`mr` is the kernel's row granularity, read from its own API. No "
                 "statistics, no measurement error: a kernel with `mr=8` computes 8 "
                 "rows per macro-tile.")
    lines.append("")
    lines.append("| variant | ISA | mr | nr | rows used at M=1 |")
    lines.append("|:--|:--|--:|--:|--:|")
    for v in sorted(variants, key=lambda x: (x.get("isa", ""), x.get("name", ""))):
        mr = v.get("mr", 0) or 1
        util = 100.0 / mr
        flag = " ⬅" if mr > 1 and v.get("isa") == "i8mm" else ""
        lines.append(f"| `{v.get('name','?')[:56]}` | {v.get('isa')} | {mr} | "
                     f"{v.get('nr','?')} | {util:.1f}%{flag} |")
    lines.append("")

    i8mm_mrs = [v.get("mr", 0) for v in variants if v.get("isa") == "i8mm" and v.get("mr")]
    dot_mrs = [v.get("mr", 0) for v in variants if v.get("isa") == "dotprod" and v.get("mr")]
    out["min_i8mm_mr"] = min(i8mm_mrs) if i8mm_mrs else None
    out["min_dotprod_mr"] = min(dot_mrs) if dot_mrs else None

    if i8mm_mrs:
        smallest = min(i8mm_mrs)
        if smallest > 1:
            waste = (1.0 - 1.0 / smallest) * 100.0
            lines.append(f"**The smallest i8mm row granularity is {smallest}.** At "
                         f"decode (M=1) such a kernel produces one useful row per "
                         f"{smallest}-row tile — **{waste:.0f}% of the tile's row "
                         f"capacity is unused**. This is why decode does not reach "
                         f"i8mm: not a tuning oversight, a shape mismatch.")
        else:
            lines.append("An i8mm variant with `mr=1` exists, so the structural "
                         "argument does **not** hold on this KleidiAI revision. "
                         "Report that plainly.")
    else:
        lines.append("No i8mm variants present — nothing to compare.")
    lines.append("")

    # --- 2. empirical -------------------------------------------------------
    lines.append("## 2. Measured throughput by row count")
    lines.append("")
    lines.append("| M | best dotprod | best i8mm | winner | i8mm advantage |")
    lines.append("|--:|--:|--:|:--|--:|")

    crossover = None
    per_m: dict[str, Any] = {}
    for m in ms:
        d = best_by_isa(rows, m, "dotprod")
        i = best_by_isa(rows, m, "i8mm")
        dg = d["gflops"] if d else 0.0
        ig = i["gflops"] if i else 0.0
        if dg and ig:
            adv = (ig - dg) / dg * 100.0
            winner = "i8mm" if ig > dg else "dotprod"
            if winner == "i8mm" and crossover is None:
                crossover = m
            lines.append(f"| {m} | {dg:.1f} | {ig:.1f} | **{winner}** | {adv:+.1f}% |")
            per_m[str(m)] = {"dotprod_gflops": round(dg, 3), "i8mm_gflops": round(ig, 3),
                             "winner": winner, "i8mm_advantage_pct": round(adv, 3)}
        else:
            lines.append(f"| {m} | {dg:.1f} | {ig:.1f} | – | – |")
            per_m[str(m)] = {"dotprod_gflops": round(dg, 3), "i8mm_gflops": round(ig, 3),
                             "winner": None}
    lines.append("")
    out["per_m"] = per_m
    out["crossover_m"] = crossover

    # --- 3. what it means ---------------------------------------------------
    lines.append("## 3. Crossover")
    lines.append("")
    if crossover is None:
        lines.append("**No crossover in the swept range.** i8mm never overtook dotprod. "
                     "If that holds up, the practical conclusion is that i8mm matmul is "
                     "not reachable at any batch size a decoder realistically produces — "
                     "a more interesting result than confirming the thesis, and worth "
                     "reporting upstream.")
    elif crossover <= 1:
        lines.append("**i8mm wins even at M=1.** The premise is false on this hardware "
                     "and KleidiAI revision. Report it plainly and drop the "
                     "'kernels asleep during decode' framing.")
    else:
        lines.append(f"**i8mm overtakes dotprod at M={crossover}.**")
        lines.append("")
        lines.append(f"- Decode at M=1 sits **below** the crossover, so single-token "
                     f"decode cannot benefit from i8mm however the surrounding code is "
                     f"tuned.")
        lines.append(f"- Any mechanism that raises the effective row count to "
                     f"**≥ {crossover}** reaches it. Speculative decoding with a draft "
                     f"length of at least {crossover} qualifies; so does serving "
                     f"concurrency of at least {crossover} once requests batch.")
        lines.append(f"- This is the number `tools/advisor.py` uses to recommend a draft "
                     f"length.")
    lines.append("")
    lines.append("> Measured with nothing between the timer and the kernel — no "
                 "llama.cpp, no threading policy, no KV cache. That makes it a clean "
                 "statement about kernel selection, and **not** a prediction of "
                 "end-to-end speedup.")
    lines.append("")

    md = "\n".join(lines)
    out_md = args.out or os.path.join(RESULTS, "probe_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(RESULTS, "probe_analysis.latest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(md)
    print(f"\n[analyze_probe] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

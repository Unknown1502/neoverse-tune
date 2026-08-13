#!/usr/bin/env python3
"""
analyze_threshold_sweep.py — does the optimal routing threshold depend on
prompt shape?

THE QUESTION, STATED SO IT CAN COME BACK "NO"
---------------------------------------------
Inter-tenant similarity is approximately shared_preamble / total_prompt, which
is a property of the PROMPTS, not of llama.cpp. If that model is right, the
threshold that separates tenants must move when the prompt shape moves. If the
two shapes produce the same optimum, the model is wrong and this script should
say so.

SELECTION RULE — FIXED BEFORE LOOKING AT RESULTS
------------------------------------------------
For each shape, the chosen threshold is the one with the lowest median warm
TTFT, subject to:

  * at least MIN_REPS repetitions completed
  * correctness not FAILED for that threshold, where correctness was measured
  * ties broken toward the LOWER threshold

The tie rule matters. Thresholds this close are not distinguishable at n=3, and
preferring the lower one is the conservative choice: it is closer to llama.cpp's
default, so it makes the "you must change this" claim harder to support, not
easier.

TIE DEFINITION
--------------
Two thresholds are tied when their medians are within TIE_PCT of the better one.
At n=3 with no confidence intervals, a 10% gap is not a result. Reporting a
"winner" inside that band would be reading noise.

OUTPUTS
-------
    results/raw/threshold_sweep.jsonl        one row per repetition
    results/aggregated/threshold_summary.csv aggregates, reproducible from raw
    results/aggregated/summary.json          machine-readable verdict
    docs/fig_ttft_vs_threshold.svg           required figure 1
    docs/fig_tokens_vs_threshold.svg         required figure 2
    docs/fig_optimal_by_shape.svg            required figure 3

Usage:
    python3 tools/analyze_threshold_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, HERE)

MIN_REPS = 3          # sweep repetitions; below this a threshold is not reported
TIE_PCT = 10.0        # medians within this of the best are called tied

# Where each shape's artifacts live. The names on disk are historical; the
# shape names here follow the experiment's vocabulary.
SHAPES = {
    "high_prefix_reuse": {
        "aka": "agent",
        "logs": [os.path.join(RESULTS, "sweep", "sweep_sim*_r*.log"),
                 os.path.join(RESULTS, "sweep_sim*_r*.log")],
        "tune": os.path.join(RESULTS, "tune.latest.json"),
        "desc": "system prompt + tool schemas shared by every tenant",
    },
    "low_prefix_reuse": {
        "aka": "rag",
        "logs": [os.path.join(RESULTS, "sweep_rag_sim*_r*.log")],
        "tune": os.path.join(RESULTS, "tune_rag.json"),
        "desc": "short shared instruction, large per-tenant retrieved document",
    },
}

RE_SIM = re.compile(r"f_sim_best\s*=\s*([0-9.]+)")
RE_PROMPT = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_NAME = re.compile(r"sim([0-9.]+)_r(\d+)\.log$")


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def parse_log(path: str) -> dict:
    """Mechanism metrics for one run, from llama-server's own log."""
    sims, toks, rates = [], [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = RE_SIM.search(line)
                if m:
                    sims.append(float(m.group(1)))
                m = RE_PROMPT.search(line)
                if m:
                    toks.append(int(m.group(2)))
    except OSError as exc:
        return {"status": "FAILED", "reason": str(exc)}
    if not toks:
        return {"status": "FAILED", "reason": "no prompt-eval lines in log"}
    return {
        "status": "OK",
        "median_similarity": round(statistics.median(sims), 4) if sims else None,
        "median_recomputed_tokens": int(statistics.median(toks)),
        "max_recomputed_tokens": max(toks),
        "requests": len(toks),
    }


def collect(shape: str, spec: dict) -> list[dict]:
    """One raw row per (shape, threshold, repetition)."""
    # TTFT samples live in the tune JSON, keyed by threshold, in repetition order.
    ttft_by_thr: dict[float, list[float]] = {}
    if os.path.exists(spec["tune"]):
        try:
            with open(spec["tune"], encoding="utf-8") as fh:
                for r in json.load(fh).get("sweep", []):
                    ttft_by_thr[float(r["threshold"])] = list(r.get("samples", []))
        except (OSError, ValueError, KeyError):
            pass

    paths: list[str] = []
    for pattern in spec["logs"]:
        paths.extend(glob.glob(pattern))
    # The RAG logs live beside the agent ones; keep them apart by name.
    if shape == "high_prefix_reuse":
        paths = [p for p in paths if "_rag_" not in os.path.basename(p)]

    # ttft_ms is client-side timing and is NOT in the server log, so it is
    # joined to each log by POSITION: samples[repetition]. sweep() appends a
    # sample only for a repetition that produced one, so if any repetition had
    # failed the list would be short and every later row would silently shift
    # onto the wrong measurement. Count logs per threshold first and refuse the
    # join when the counts disagree, rather than emitting a plausible lie.
    logs_per_thr: dict[float, int] = {}
    for p in paths:
        m = RE_NAME.search(os.path.basename(p))
        if m:
            logs_per_thr[float(m.group(1))] = logs_per_thr.get(float(m.group(1)), 0) + 1

    rows = []
    for p in sorted(paths):
        m = RE_NAME.search(os.path.basename(p))
        if not m:
            continue
        thr, rep = float(m.group(1)), int(m.group(2))
        mech = parse_log(p)
        samples = ttft_by_thr.get(thr, [])
        aligned = len(samples) == logs_per_thr.get(thr, 0)
        rows.append({
            "experiment_id": f"{shape}-{str(thr).replace('.', '')}-r{rep:02d}",
            "workload_shape": shape,
            "threshold": thr,
            "repetition": rep,
            "ttft_ms": (samples[rep] if aligned and rep < len(samples) else None),
            "ttft_source": ("tune json, positional join, counts agree" if aligned
                            else f"WITHHELD: {len(samples)} samples vs "
                                 f"{logs_per_thr.get(thr,0)} logs at this threshold"),
            "recomputed_prompt_tokens": mech.get("median_recomputed_tokens"),
            "max_recomputed_tokens": mech.get("max_recomputed_tokens"),
            "chosen_slot_similarity": mech.get("median_similarity"),
            "requests": mech.get("requests"),
            "status": mech.get("status"),
            "reason": mech.get("reason"),
            "log": os.path.relpath(p, ROOT).replace("\\", "/"),
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["workload_shape"], r["threshold"]) for r in rows})
    for shape, thr in keys:
        grp = [r for r in rows if r["workload_shape"] == shape and r["threshold"] == thr]
        ok = [r for r in grp if r["status"] == "OK"]
        ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
        toks = [r["recomputed_prompt_tokens"] for r in ok
                if r["recomputed_prompt_tokens"] is not None]
        sims = [r["chosen_slot_similarity"] for r in ok
                if r["chosen_slot_similarity"] is not None]
        out.append({
            "workload_shape": shape,
            "threshold": thr,
            "n": len(ttft),
            "n_failed": len(grp) - len(ok),
            "ttft_median": round(statistics.median(ttft), 1) if ttft else None,
            "ttft_mean": round(statistics.fmean(ttft), 1) if ttft else None,
            "ttft_sd": round(statistics.stdev(ttft), 1) if len(ttft) > 1 else None,
            "ttft_p95": round(pct(ttft, 0.95), 1) if ttft else None,
            "ttft_min": round(min(ttft), 1) if ttft else None,
            "ttft_max": round(max(ttft), 1) if ttft else None,
            "recomputed_tokens_median": int(statistics.median(toks)) if toks else None,
            "chosen_similarity_median": round(statistics.median(sims), 4) if sims else None,
        })
    return out


def select(agg: list[dict], shape: str, correctness: dict) -> dict:
    """Apply the pre-registered rule. Returns the decision AND its reasoning."""
    cand = [a for a in agg
            if a["workload_shape"] == shape
            and a["n"] >= MIN_REPS
            and a["ttft_median"] is not None]
    excluded = []

    bad = {t for t, v in correctness.items() if v == "FAILED"}
    if bad:
        cand2 = [a for a in cand if a["threshold"] not in bad]
        excluded = [a["threshold"] for a in cand if a["threshold"] in bad]
        cand = cand2

    if not cand:
        return {"shape": shape, "chosen": None,
                "reason": f"no threshold reached {MIN_REPS} repetitions",
                "excluded_for_correctness": excluded}

    best = min(cand, key=lambda a: a["ttft_median"])
    tied = [a for a in cand
            if a["ttft_median"] <= best["ttft_median"] * (1 + TIE_PCT / 100.0)]
    chosen = min(tied, key=lambda a: a["threshold"])   # conservative tie-break

    worst = max(cand, key=lambda a: a["ttft_median"])
    return {
        "shape": shape,
        "chosen": chosen["threshold"],
        "chosen_ttft_median": chosen["ttft_median"],
        "lowest_median_threshold": best["threshold"],
        "lowest_median": best["ttft_median"],
        "tied_within_%.0f%%" % TIE_PCT: sorted(a["threshold"] for a in tied),
        "worst_threshold": worst["threshold"],
        "worst_ttft_median": worst["ttft_median"],
        "spread_x": (round(worst["ttft_median"] / best["ttft_median"], 2)
                     if best["ttft_median"] else None),
        "recomputed_at_chosen": chosen["recomputed_tokens_median"],
        "recomputed_at_worst": worst["recomputed_tokens_median"],
        "excluded_for_correctness": excluded,
        "reason": ("lowest median TTFT; tie-broken to the lower threshold"
                   if chosen["threshold"] != best["threshold"]
                   else "lowest median TTFT, no tie"),
    }


# ------------------------------------------------------------------ figures

def _line_svg(path, title, sub, series, ylab, ylog=False):
    """Minimal multi-series line chart. No dependencies, theme-aware."""
    W, H, L, R, T, B = 860, 430, 78, 26, 78, 74
    pw, ph = W - L - R, H - T - B
    xs = sorted({x for s in series for x, _ in s["pts"]})
    ys = [y for s in series for _, y in s["pts"] if y is not None]
    if not xs or not ys:
        return False
    ymax = max(ys) * 1.15
    ymin = 0.0
    xmin, xmax = min(xs), max(xs)
    sx = lambda x: L + (0 if xmax == xmin else (x - xmin) / (xmax - xmin)) * pw
    sy = lambda y: T + ph - (y - ymin) / (ymax - ymin) * ph

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
         '<style>.t{font-size:17px;font-weight:700;fill:#c9d1d9}'
         '.s{font-size:11.5px;fill:#8b949e}.a{font-size:10.5px;fill:#8b949e}'
         '.g{stroke:#8b949e;stroke-opacity:.2}'
         '@media(prefers-color-scheme:light){.t{fill:#1f2328}.s,.a{fill:#59636e}.g{stroke:#59636e}}'
         '</style>',
         f'<text x="{L}" y="30" class="t">{title}</text>',
         f'<text x="{L}" y="50" class="s">{sub}</text>']
    for i in range(5):
        gy = T + ph - ph * i / 4
        o.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{L+pw}" y2="{gy:.1f}" class="g"/>')
        o.append(f'<text x="{L-8}" y="{gy+4:.1f}" class="a" text-anchor="end">'
                 f'{ymin + (ymax-ymin)*i/4:,.0f}</text>')
    for x in xs:
        o.append(f'<text x="{sx(x):.1f}" y="{T+ph+18:.1f}" class="a" '
                 f'text-anchor="middle">{x}</text>')
    o.append(f'<text x="{L+pw/2:.0f}" y="{H-26}" class="s" text-anchor="middle">'
             f'--slot-prompt-similarity</text>')
    o.append(f'<text x="16" y="{T+ph/2:.0f}" class="s" transform="rotate(-90 16 {T+ph/2:.0f})" '
             f'text-anchor="middle">{ylab}</text>')
    for s in series:
        pts = [(sx(x), sy(y)) for x, y in s["pts"] if y is not None]
        if not pts:
            continue
        d = " ".join(f'{"M" if i==0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2.6"/>')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{s["color"]}"/>')
    lx = L
    for s in series:
        o.append(f'<rect x="{lx}" y="{H-14}" width="18" height="3" fill="{s["color"]}"/>')
        o.append(f'<text x="{lx+24}" y="{H-9}" class="a">{s["name"]}</text>')
        lx += 24 + len(s["name"]) * 6.6 + 26
    o.append('</svg>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    correctness: dict[float, str] = {}
    cpath = os.path.join(RESULTS, "correctness.json")
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as fh:
                for r in json.load(fh).get("thresholds", []):
                    correctness[float(r["threshold"])] = r["status"]
        except (OSError, ValueError, KeyError):
            pass

    raw: list[dict] = []
    for shape, spec in SHAPES.items():
        raw.extend(collect(shape, spec))
    if not raw:
        print("no sweep logs found — run tools/tune_similarity.py --sweep first",
              file=sys.stderr)
        return 1

    agg = aggregate(raw)
    shapes_present = sorted({r["workload_shape"] for r in raw})
    decisions = {s: select(agg, s, correctness) for s in shapes_present}

    # ---------------------------------------------------------------- write
    rawdir = os.path.join(args.out_dir, "raw")
    aggdir = os.path.join(args.out_dir, "aggregated")
    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(aggdir, exist_ok=True)

    with open(os.path.join(rawdir, "threshold_sweep.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")

    if agg:
        with open(os.path.join(aggdir, "threshold_summary.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)

    # ---------------------------------------------------------------- print
    print()
    for shape in shapes_present:
        spec = SHAPES[shape]
        print(f"  {shape}   ({spec['desc']})")
        print(f"  {'thr':>6} {'n':>3} {'TTFT med':>10} {'mean':>9} {'sd':>8} "
              f"{'p95':>9} {'min':>9} {'max':>9} {'recomp':>8} {'sim':>7}")
        print("  " + "-" * 92)
        for a in [x for x in agg if x["workload_shape"] == shape]:
            f = lambda v, d=1: ("-" if v is None else f"{v:,.{d}f}")
            print(f"  {a['threshold']:>6} {a['n']:>3} {f(a['ttft_median']):>10} "
                  f"{f(a['ttft_mean']):>9} {f(a['ttft_sd']):>8} {f(a['ttft_p95']):>9} "
                  f"{f(a['ttft_min']):>9} {f(a['ttft_max']):>9} "
                  f"{('-' if a['recomputed_tokens_median'] is None else a['recomputed_tokens_median']):>8} "
                  f"{f(a['chosen_similarity_median'], 3):>7}")
        d = decisions[shape]
        print(f"\n    chosen: {d['chosen']}   ({d.get('reason')})")
        if d.get("tied_within_10%"):
            print(f"    tied within {TIE_PCT:.0f}%: {d['tied_within_10%']}")
        if d.get("spread_x"):
            print(f"    spread across tested thresholds: {d['spread_x']}x")
        print()

    # -------------------------------------------------------------- figures
    COL = {"high_prefix_reuse": "#f0555f", "low_prefix_reuse": "#3fcf7f"}
    figs = os.path.join(ROOT, "docs")
    s_ttft, s_tok = [], []
    for shape in shapes_present:
        rows = [a for a in agg if a["workload_shape"] == shape and a["n"] >= MIN_REPS]
        s_ttft.append({"name": shape, "color": COL.get(shape, "#5aa9ff"),
                       "pts": [(a["threshold"], a["ttft_median"]) for a in rows]})
        s_tok.append({"name": shape, "color": COL.get(shape, "#5aa9ff"),
                      "pts": [(a["threshold"], a["recomputed_tokens_median"]) for a in rows]})

    made = []
    if _line_svg(os.path.join(figs, "fig_ttft_vs_threshold.svg"),
                 "Time to first token vs routing threshold",
                 "median of repetitions · lower is better", s_ttft, "median TTFT (ms)"):
        made.append("fig_ttft_vs_threshold.svg")
    if _line_svg(os.path.join(figs, "fig_tokens_vs_threshold.svg"),
                 "Tokens recomputed vs routing threshold",
                 "from llama-server's own log · the mechanism behind the latency",
                 s_tok, "recomputed tokens"):
        made.append("fig_tokens_vs_threshold.svg")

    summary = {
        "schema": "specarm.threshold_sweep/1",
        "min_reps": MIN_REPS, "tie_pct": TIE_PCT,
        "selection_rule": "lowest median warm TTFT; ties within "
                          f"{TIE_PCT:.0f}% broken toward the LOWER threshold; "
                          "correctness-FAILED thresholds excluded",
        "shapes": {s: SHAPES[s]["desc"] for s in shapes_present},
        "decisions": decisions,
        "aggregated": agg,
        "correctness_measured": bool(correctness),
        "figures": made,
    }
    with open(os.path.join(aggdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    chosen = {s: decisions[s]["chosen"] for s in shapes_present}
    print("  " + "=" * 72)
    if len(shapes_present) < 2:
        print("  Only one shape measured — the central question needs two.")
    elif len(set(v for v in chosen.values() if v is not None)) > 1:
        print(f"  Optimal threshold DIFFERED across shapes: {chosen}")
        print("  -> evidence that prompt shape influences the optimum.")
    else:
        print(f"  Optimal threshold was the SAME for both shapes: {chosen}")
        print("  -> the evaluated shapes did not shift the optimum. Null result.")
    print("  " + "=" * 72)
    print(f"\n  raw: {os.path.join(rawdir,'threshold_sweep.jsonl')}  ({len(raw)} rows)")
    print(f"  agg: {os.path.join(aggdir,'threshold_summary.csv')}")
    print(f"  fig: {', '.join(made) if made else 'none'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

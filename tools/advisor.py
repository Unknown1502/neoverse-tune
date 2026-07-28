#!/usr/bin/env python3
"""
advisor.py — turn SpecArm's measurements into a configuration recommendation.

This is the reusable artifact. Everything else produces evidence; this consumes
it and answers the question a developer actually has:

    "I want to serve an LLM on Arm. What do I set, and does this box even
     benefit from the thing you measured?"

It degrades honestly. With a full evidence chain it gives a specific draft
length backed by a measured crossover. With only a CPU feature report it says
what can be known from CPU features alone and labels its own confidence
accordingly. It never invents a number it did not read.

Run:  python3 tools/advisor.py [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))

# Confidence reflects what evidence was actually available, not how good the
# numbers looked.
CONFIDENCE = {
    "measured": "measured on this machine",
    "inferred": "inferred from CPU features only",
    "unknown": "no evidence available",
}


def load(name: str) -> dict[str, Any]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_advice() -> dict[str, Any]:
    env = load("env.latest.json")
    probe = load("probe_analysis.latest.json")
    serve = load("serve_analysis.latest.json")

    features = env.get("features", {}) or {}
    has_i8mm = bool(features.get("i8mm"))
    core = env.get("core", "unknown")
    nproc = env.get("nproc") or 0

    advice: dict[str, Any] = {
        "schema": "specarm.advice/1",
        "core": core,
        "has_i8mm": has_i8mm,
        "evidence": {
            "env": bool(env),
            "probe": bool(probe),
            "serve": bool(serve),
        },
        "recommendations": [],
        "warnings": [],
    }

    # --- can this box benefit at all? --------------------------------------
    if not env:
        advice["verdict"] = "unknown"
        advice["headline"] = ("No environment report found. Run "
                              "scripts/00_env_report.sh on the target machine.")
        return advice

    if not has_i8mm:
        advice["verdict"] = "no_benefit"
        advice["headline"] = (
            f"{core} has no i8mm. There is no SMMLA matmul path to reach, so "
            "raising batch size cannot unlock better kernels here. Batching still "
            "helps by amortizing weight streaming — just not for this reason.")
        advice["recommendations"].append({
            "setting": "instance",
            "value": "move to an i8mm-capable core",
            "why": "Neoverse N2/V2 (Azure Cobalt, Graviton3/4) expose i8mm; "
                   "Neoverse N1 (Graviton2) does not",
            "confidence": CONFIDENCE["measured"],
        })
        return advice

    # --- crossover ----------------------------------------------------------
    crossover = probe.get("crossover_m")
    min_mr = probe.get("min_i8mm_mr")

    if crossover is None and probe:
        advice["verdict"] = "no_crossover"
        advice["headline"] = (
            f"{core} has i8mm, but the probe found no batch size where i8mm beat "
            "dotprod. Raising draft length will not buy a better kernel here.")
        advice["warnings"].append(
            "Probe ran but found no crossover — re-check the swept M range before "
            "concluding, and confirm the probe reported zero mismatches.")
        return advice

    if crossover is None:
        # Only CPU features known. Say what follows from geometry alone.
        advice["verdict"] = "likely_benefit"
        advice["headline"] = (
            f"{core} has i8mm, so an SMMLA matmul path exists. Without probe data "
            "the crossover batch size is unknown.")
        advice["recommendations"].append({
            "setting": "next step",
            "value": "bash scripts/05_kai_probe.sh",
            "why": "measures the actual crossover on this machine rather than guessing",
            "confidence": CONFIDENCE["inferred"],
        })
        if min_mr:
            advice["recommendations"].append({
                "setting": "draft length (provisional)",
                "value": f">= {min_mr}",
                "why": f"smallest i8mm kernel row granularity is {min_mr}; "
                       f"below it the tile is partly unused",
                "confidence": CONFIDENCE["inferred"],
            })
        return advice

    # --- full evidence ------------------------------------------------------
    advice["verdict"] = "benefit"
    advice["crossover_m"] = crossover
    advice["headline"] = (
        f"{core}: i8mm overtakes dotprod at M={crossover}. Single-token decode "
        f"(M=1) sits below that, so it cannot reach i8mm. Any mechanism raising "
        f"the effective row count to {crossover} or more can.")

    advice["recommendations"].append({
        "setting": "speculative draft length",
        "value": f">= {crossover}",
        "why": f"verifying D draft tokens is one forward pass over D rows; "
               f"D >= {crossover} lands past the measured crossover",
        "confidence": CONFIDENCE["measured"],
    })
    advice["recommendations"].append({
        "setting": "server concurrency",
        "value": f">= {crossover} concurrent requests",
        "why": "a server batches concurrent requests into one forward pass, "
               "reaching the same regime from the serving side",
        "confidence": CONFIDENCE["measured"],
    })
    if nproc:
        advice["recommendations"].append({
            "setting": "threads",
            "value": str(nproc),
            "why": f"one thread per available core ({nproc} detected); "
                   f"oversubscription adds scheduling noise without throughput",
            "confidence": CONFIDENCE["inferred"],
        })

    # --- corroboration ------------------------------------------------------
    widening = serve.get("widening_pct_points")
    if widening is not None:
        if widening > 5:
            advice["recommendations"].append({
                "setting": "corroboration",
                "value": f"serving advantage widened {widening:+.1f} pts with concurrency",
                "why": "independent evidence consistent with the probe's crossover",
                "confidence": CONFIDENCE["measured"],
            })
        elif widening < -5:
            advice["warnings"].append(
                f"Serving advantage NARROWED {widening:+.1f} points with concurrency, "
                "which contradicts the probe. Reconcile before relying on either.")

    if env.get("virtualization") not in (None, "none", "unknown"):
        advice["warnings"].append(
            f"Virtualized host ({env.get('virtualization')}): burstable instance types "
            "throttle mid-benchmark. Confirm these numbers on a dedicated instance "
            "before acting on them.")

    return advice


def render(advice: dict[str, Any]) -> str:
    icon = {"benefit": "✅", "likely_benefit": "🔎", "no_benefit": "⛔",
            "no_crossover": "⚠️", "unknown": "❔"}.get(advice.get("verdict"), "❔")
    out: list[str] = []
    out.append("")
    out.append("  ┌─ SpecArm advisor " + "─" * 52)
    out.append(f"  │ {icon}  {advice.get('core', 'unknown')}")
    out.append("  │")
    for line in _wrap(advice.get("headline", ""), 66):
        out.append(f"  │ {line}")
    if advice.get("recommendations"):
        out.append("  │")
        out.append("  ├─ recommendations " + "─" * 51)
        for r in advice["recommendations"]:
            out.append(f"  │ • {r['setting']}: {r['value']}")
            for line in _wrap(r["why"], 62):
                out.append(f"  │     {line}")
            out.append(f"  │     [{r['confidence']}]")
    if advice.get("warnings"):
        out.append("  │")
        out.append("  ├─ warnings " + "─" * 58)
        for w in advice["warnings"]:
            for line in _wrap(w, 66):
                out.append(f"  │ ! {line}")
    ev = advice.get("evidence", {})
    out.append("  │")
    out.append(f"  └─ evidence: env={_yn(ev.get('env'))} probe={_yn(ev.get('probe'))} "
               f"serve={_yn(ev.get('serve'))} " + "─" * 22)
    out.append("")
    return "\n".join(out)


def _yn(v: Any) -> str:
    return "yes" if v else "no"


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    advice = build_advice()

    if args.json:
        print(json.dumps(advice, indent=2))
    else:
        print(render(advice))

    out_path = os.path.join(RESULTS, "advice.latest.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(advice, fh, indent=2)

    # Exit code carries the verdict so the advisor can gate a deployment script.
    return {"benefit": 0, "likely_benefit": 0, "no_crossover": 3,
            "no_benefit": 4, "unknown": 5}.get(advice.get("verdict"), 5)


if __name__ == "__main__":
    sys.exit(main())

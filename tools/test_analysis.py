#!/usr/bin/env python3
"""
test_analysis.py — validate analyze_serve, analyze_probe and advisor.

Each is fed synthetic evidence with a known correct conclusion, including the
cases where the honest answer is "no" or "I cannot tell". A tool that only
behaves correctly on the happy path will mislead precisely when it matters.

ALL FIXTURES ARE SYNTHETIC. Nothing here measures hardware.

Run:  python3 tools/test_analysis.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  [' + detail + ']') if detail else ''}")
    if not ok:
        FAILURES.append(name)


def run(script: str, results: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, SPECARM_RESULTS=results, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, os.path.join(HERE, script), *args],
                          capture_output=True, text=True, encoding="utf-8", env=env)


# --------------------------------------------------------------------------- serve

def serve_run(variant: str, conc: int, tok_s: float, ttft: float, tpot: float,
              failed: int = 0, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = 24
    ttfts = [rng.gauss(ttft, ttft * 0.05) for _ in range(n)]
    tpots = [rng.gauss(tpot, tpot * 0.05) for _ in range(n)]
    totals = [a + b * 40 for a, b in zip(ttfts, tpots)]
    summ = lambda vals: {"metric": "x", "n": len(vals),
                         "mean": sum(vals) / len(vals),
                         "p50": sorted(vals)[len(vals) // 2],
                         "p90": sorted(vals)[int(len(vals) * 0.9)],
                         "p95": sorted(vals)[int(len(vals) * 0.95)],
                         "p99": sorted(vals)[-1], "max": max(vals)}
    return {"schema": "specarm.serve/1", "label": f"{variant}_c{conc}",
            "concurrency": conc, "requests_attempted": n, "requests_ok": n - failed,
            "requests_failed": failed, "wall_seconds": 10.0,
            "output_tokens_per_sec": tok_s, "completed_requests_per_sec": 2.4,
            "total_output_tokens": n * 40,
            "ttft_ms": summ(ttfts), "tpot_ms": summ(tpots), "total_ms": summ(totals),
            "raw": {"ttft_ms": ttfts, "tpot_ms": tpots, "total_ms": totals},
            "errors": []}


def test_serve() -> None:
    print("\nanalyze_serve\n")
    tmp = tempfile.mkdtemp(prefix="specarm-serve-")
    try:
        S = "20260101T000000Z"
        # KleidiAI advantage widens with concurrency: 2% -> 30%.
        plan = {1: (100.0, 102.0), 2: (110.0, 118.0), 4: (120.0, 140.0), 8: (130.0, 169.0)}
        for c, (base_t, kle_t) in plan.items():
            json.dump(serve_run("base", c, base_t, 200.0, 25.0, seed=c),
                      open(os.path.join(tmp, f"serve_base_c{c}_{S}.json"), "w"))
            json.dump(serve_run("kleidi", c, kle_t, 190.0, 25.0 * base_t / kle_t,
                                seed=100 + c),
                      open(os.path.join(tmp, f"serve_kleidi_c{c}_{S}.json"), "w"))
        # One poisoned cell that must be excluded.
        json.dump(serve_run("kleidi", 16, 999.0, 190.0, 5.0, failed=7, seed=9),
                  open(os.path.join(tmp, f"serve_kleidi_c16_{S}.json"), "w"))
        json.dump(serve_run("base", 16, 140.0, 200.0, 25.0, seed=10),
                  open(os.path.join(tmp, f"serve_base_c16_{S}.json"), "w"))
        open(os.path.join(tmp, "serve.latest"), "w").write(S)
        json.dump({"core": "Neoverse-N2", "crux_role": "subject", "nproc": 8,
                   "features": {"i8mm": True}},
                  open(os.path.join(tmp, "env.latest.json"), "w"))

        p = run("analyze_serve.py", tmp)
        check("analyze_serve exits 0", p.returncode == 0, p.stderr.strip()[-160:])
        md = p.stdout
        a = json.load(open(os.path.join(tmp, "serve_analysis.latest.json")))

        check("failed-request run excluded",
              "kleidi_c16" in a["integrity"]["excluded_runs"],
              str(a["integrity"]["excluded_runs"]))
        check("poisoned throughput not reported", "999.0" not in md)
        check("widening detected",
              a.get("widening_pct_points", 0) > 5,
              f"{a.get('widening_pct_points')}")
        check("widening section present", "Does the advantage widen" in md)
        check("throughput labelled as single observation",
              "single observation" in md.lower())
        check("latency adjudicated", any(
            "verdict" in v for k, v in a["concurrency"].items()
            for v in [a["concurrency"][k].get("tpot_ms", {})]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- probe

def probe_json(crossover: int | None, i8mm_mr: int = 8) -> dict:
    variants = [
        {"name": "clamp_f32_a_1x4x32_neon_dotprod", "isa": "dotprod",
         "mr": 1, "nr": 4, "kr": 8, "sr": 1},
        {"name": "clamp_f32_b_8x8x32_neon_i8mm", "isa": "i8mm",
         "mr": i8mm_mr, "nr": 8, "kr": 8, "sr": 1},
    ]
    meas = []
    for m in [1, 2, 4, 8, 16, 32]:
        dot = 10.0 + m * 0.5
        if crossover is None:
            i8 = dot * 0.6                       # never wins
        else:
            i8 = dot * (1.6 if m >= crossover else 0.5)
        for name, isa, g in (("clamp_f32_a_1x4x32_neon_dotprod", "dotprod", dot),
                             ("clamp_f32_b_8x8x32_neon_i8mm", "i8mm", i8)):
            mr = 1 if isa == "dotprod" else i8mm_mr
            meas.append({"m": m, "variant": name, "isa": isa, "ok": True, "mr": mr,
                         "row_utilization": m / (((m + mr - 1) // mr) * mr),
                         "median_ns": 1000.0, "gflops": g})
    return {"schema": "specarm.kai_probe/1", "n": 4096, "k": 4096, "reps": 20,
            "variants": variants, "measurements": meas, "mismatches": 0,
            "kleidiai_sha": "a" * 40}


def test_probe() -> None:
    print("\nanalyze_probe\n")
    for label, crossover, expect in [("crossover at 8", 8, 8),
                                     ("no crossover", None, None)]:
        tmp = tempfile.mkdtemp(prefix="specarm-probe-")
        try:
            json.dump(probe_json(crossover),
                      open(os.path.join(tmp, "kai_probe.latest.json"), "w"))
            p = run("analyze_probe.py", tmp)
            check(f"{label}: exits 0", p.returncode == 0, p.stderr.strip()[-160:])
            a = json.load(open(os.path.join(tmp, "probe_analysis.latest.json")))
            check(f"{label}: crossover = {expect}", a.get("crossover_m") == expect,
                  f"got {a.get('crossover_m')}")
            if expect is None:
                check(f"{label}: reports absence honestly",
                      "No crossover" in p.stdout)
            else:
                check(f"{label}: structural waste stated",
                      "unused" in p.stdout and "88%" in p.stdout,
                      "expected 1/8 -> 88% unused")
                check(f"{label}: advisor linkage mentioned", "advisor.py" in p.stdout)
            check(f"{label}: disclaims end-to-end prediction",
                  "not" in p.stdout.lower() and "end-to-end" in p.stdout.lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------------------- advisor

def test_advisor() -> None:
    print("\nadvisor\n")
    cases = [
        ("no evidence at all", {}, "unknown", 5),
        ("N1 control, no i8mm",
         {"env": {"core": "Neoverse-N1", "nproc": 2, "crux_role": "control",
                  "features": {"i8mm": False}}}, "no_benefit", 4),
        ("i8mm but no probe yet",
         {"env": {"core": "Neoverse-N2", "nproc": 4, "crux_role": "subject",
                  "features": {"i8mm": True}}}, "likely_benefit", 0),
        ("full evidence",
         {"env": {"core": "Neoverse-N2", "nproc": 4, "crux_role": "subject",
                  "virtualization": "none", "features": {"i8mm": True}},
          "probe": {"crossover_m": 8, "min_i8mm_mr": 8}}, "benefit", 0),
        ("probe found no crossover",
         {"env": {"core": "Neoverse-N2", "nproc": 4, "features": {"i8mm": True}},
          "probe": {"crossover_m": None, "min_i8mm_mr": 8}}, "no_crossover", 3),
    ]
    for label, ev, expect_verdict, expect_rc in cases:
        tmp = tempfile.mkdtemp(prefix="specarm-adv-")
        try:
            if "env" in ev:
                json.dump(ev["env"], open(os.path.join(tmp, "env.latest.json"), "w"))
            if "probe" in ev:
                json.dump(ev["probe"],
                          open(os.path.join(tmp, "probe_analysis.latest.json"), "w"))
            p = run("advisor.py", tmp, "--json")
            a = json.loads(p.stdout) if p.stdout.strip() else {}
            check(f"{label}: verdict={expect_verdict}",
                  a.get("verdict") == expect_verdict, f"got {a.get('verdict')}")
            check(f"{label}: exit={expect_rc}", p.returncode == expect_rc,
                  f"got {p.returncode}")
            if expect_verdict == "benefit":
                recs = " ".join(str(r) for r in a.get("recommendations", []))
                check(f"{label}: recommends draft length >= crossover", ">= 8" in recs)
                check(f"{label}: every rec states confidence",
                      all("confidence" in r for r in a["recommendations"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Contradictory evidence must produce a warning, not a confident answer.
    tmp = tempfile.mkdtemp(prefix="specarm-adv-contra-")
    try:
        json.dump({"core": "Neoverse-N2", "nproc": 4, "features": {"i8mm": True}},
                  open(os.path.join(tmp, "env.latest.json"), "w"))
        json.dump({"crossover_m": 8, "min_i8mm_mr": 8},
                  open(os.path.join(tmp, "probe_analysis.latest.json"), "w"))
        json.dump({"widening_pct_points": -20.0},
                  open(os.path.join(tmp, "serve_analysis.latest.json"), "w"))
        p = run("advisor.py", tmp, "--json")
        a = json.loads(p.stdout)
        check("contradictory serve evidence warns",
              any("NARROWED" in w for w in a.get("warnings", [])),
              str(a.get("warnings")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Human-readable render must not crash on the sparsest input.
    tmp = tempfile.mkdtemp(prefix="specarm-adv-render-")
    try:
        p = run("advisor.py", tmp)
        check("text render works with no evidence", "SpecArm advisor" in p.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("\nAnalysis tools — synthetic evidence with known conclusions")
    test_serve()
    test_probe()
    test_advisor()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All analysis checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

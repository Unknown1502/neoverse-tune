#!/usr/bin/env python3
"""
test_skeptic.py — validate the adjudicator against data whose answer we know.

The Skeptic is the credibility layer of this project. If it rubber-stamps noise
as VERIFIED, every number downstream is worthless. So before it is ever pointed
at a real Arm core, it gets fed synthetic distributions with known ground truth.

These inputs are SYNTHETIC and exist only to test the statistics. No number here
is a measurement of anything, and none of it goes into results/.

Run:  python3 tools/test_skeptic.py
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("analyze", os.path.join(HERE, "analyze.py"))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

Stat, verdict = analyze.Stat, analyze.verdict

FAILURES: list[str] = []


def synth(mean: float, rsd_pct: float, n: int, seed: int) -> list[float]:
    """n samples around `mean` with the given relative standard deviation."""
    rng = random.Random(seed)
    sigma = mean * rsd_pct / 100.0
    return [rng.gauss(mean, sigma) for _ in range(n)]


def check(name: str, base: list[float], cand: list[float], expected: str) -> None:
    got, why = verdict(Stat(base), Stat(cand))
    ok = got == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} -> {got:<10} ({why})")
    if not ok:
        FAILURES.append(f"{name}: expected {expected}, got {got} ({why})")


def main() -> int:
    print("\nSkeptic adjudicator — synthetic ground-truth suite\n")

    # A large, clean improvement is the only thing that should ever pass.
    check("clean +20% win, low noise",
          synth(100, 1.0, 10, 1), synth(120, 1.0, 10, 2), "VERIFIED")

    # Identical distributions must never be sold as a win.
    check("no real difference",
          synth(100, 1.0, 10, 3), synth(100, 1.0, 10, 4), "REJECTED")

    # A regression must be called a regression.
    check("regression -15%",
          synth(100, 1.0, 10, 5), synth(85, 1.0, 10, 6), "REJECTED")

    # Real but trivial: below the pre-registered effect floor.
    check("tiny +2% (below effect floor)",
          synth(100, 0.3, 12, 7), synth(102, 0.3, 12, 8), "UNCERTAIN")

    # The dangerous case: a big-looking mean delta drowned in variance.
    check("+15% but host is very noisy",
          synth(100, 25.0, 10, 9), synth(115, 25.0, 10, 10), "UNCERTAIN")

    # Too few repetitions cannot support a claim regardless of effect size.
    check("+30% but only 3 reps",
          synth(100, 1.0, 3, 11), synth(130, 1.0, 3, 12), "UNCERTAIN")

    # Overlapping CIs must be caught even when noise is INSIDE the tolerance,
    # so this case exercises the CI branch rather than the noise gate.
    got, why = verdict(Stat(synth(100, 6.0, 6, 13)), Stat(synth(106, 6.0, 6, 14)))
    ok = got == "UNCERTAIN" and "CIs overlap" in why
    print(f"  {'PASS' if ok else 'FAIL'}  {'+6% with overlapping CIs':<44} -> {got:<10} ({why})")
    if not ok:
        FAILURES.append(f"CI-overlap branch not exercised: got {got} ({why})")

    # A failed measurement must NOT be reported as a negative result. "We
    # measured nothing" and "we measured no gain" are different claims.
    check("baseline measured zero (failed run)",
          [0.0] * 10, synth(100, 1.0, 10, 15), "UNCERTAIN")

    # --- Stat sanity ------------------------------------------------------
    print()
    s = Stat(synth(100, 5.0, 30, 42))
    print(f"  Stat check: n={s.n} mean={s.mean:.2f} rsd={s.rsd:.2f}% "
          f"ci95=[{s.lo:.2f}, {s.hi:.2f}]")
    if not (s.lo < s.mean < s.hi):
        FAILURES.append("CI does not bracket the mean")
    if not (3.0 < s.rsd < 7.0):
        FAILURES.append(f"rsd {s.rsd:.2f}% far from the 5% it was generated with")

    # A wider sample must not produce a narrower confidence interval.
    tight, wide = Stat(synth(100, 1.0, 10, 50)), Stat(synth(100, 10.0, 10, 51))
    if not wide.hw > tight.hw:
        FAILURES.append("noisier sample produced a narrower CI")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All adjudicator checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

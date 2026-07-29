#!/usr/bin/env python3
"""
skeptic.py — the adjudicator. One definition of VERIFIED for the whole project.

Every performance comparison in this repo passes through verdict(). The
thresholds below were committed before any measurement existed (see
docs/methodology.md and `git log` on this file's predecessor); changing them
after seeing results would make them decoration rather than thresholds.

Rejected and uncertain results are published alongside verified ones. A harness
that reports only its wins is a harness nobody should believe, including its
author.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

# --- Pre-registered thresholds. Do not tune these to fit a result. ----------
MIN_EFFECT_PCT = 5.0   # below this, a difference is not worth claiming
MIN_REPS = 5           # fewer samples cannot support a confidence interval
MAX_RSD_PCT = 10.0     # above this the host was too noisy to conclude anything

# Two-sided t critical values at 95% for small samples, by degrees of freedom.
# At n=5 the normal approximation understates the interval by roughly 30%,
# which would manufacture VERIFIED verdicts out of thin data.
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
        return {"n": self.n, "mean": round(self.mean, 4),
                "median": round(self.median, 4), "stdev": round(self.sd, 4),
                "ci95_lo": round(self.lo, 4), "ci95_hi": round(self.hi, 4),
                "rsd_pct": round(self.rsd, 3)}


def verdict(base: Stat, cand: Stat) -> tuple[str, str]:
    """Adjudicate candidate against baseline. Higher is better, so invert
    latency to a rate before calling this."""
    # "We measured nothing" and "we measured no gain" are different claims.
    # Checking them in this order stops a failed run being published as a
    # negative result.
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

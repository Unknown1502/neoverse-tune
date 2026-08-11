#!/usr/bin/env python3
"""
test_parse_slot_log.py — the mechanism claim's regression test. (Defect D1.)

WHY THIS EXISTS
---------------
Every causal claim in this repository — "220 tokens recomputed at the default,
36 at 0.9" — comes out of two regular expressions matched against llama-server's
unstructured log text. llama.cpp changes that formatting between releases.

The dangerous failure is silent: if the patterns stop matching, `parse()` returns
zero requests and zero similarities, and a reader cannot distinguish "the server
made no bad slot decisions" from "the parser is broken". One is a result and the
other is an outage, and they look identical in the output.

So: a fixture of real log lines with hand-checked expected values, and an
explicit test that format drift is *detected* rather than silently absorbed.

Run:  python3 tools/test_parse_slot_log.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from parse_slot_log import parse  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "slot_log_sample.log")

# Hand-computed from the fixture. Four requests:
#   task 0  LRU fallback   610 tokens @ 76.19 tok/s   n_tokens 657
#   task 1  LCP 0.949       35 tokens @ 92.39 tok/s   n_tokens 739
#   task 2  LCP 0.716      220 tokens @ 51.90 tok/s   n_tokens 836
#   task 3  LCP 0.716      220 tokens @ 51.90 tok/s   n_tokens 836
#
#   tokens sorted [35, 220, 220, 610]        -> median 220
#   heavy (>100)  [610, 220, 220]            -> median 220, 3 of 4 = 75%
#   sims          [0.949, 0.716, 0.716]      -> median 0.716
#   heavy rates   [76.19, 51.90, 51.90]      -> median 51.90
#   modelled ms   220 / 51.90 * 1000         -> 4238.9
#   slots touched {3, 3, 0, 1}               -> 3 distinct
EXPECTED = {
    "requests": 4,
    "lcp_selections": 3,
    "lru_fallbacks": 1,
    "threshold": 0.100,
    "similarity_median": 0.716,
    "similarity_min": 0.716,
    "similarity_max": 0.949,
    "prefill_median": 220,
    "prefill_min": 35,
    "prefill_max": 610,
    "over_100_count": 3,
    "over_100_pct": 75.0,
    "rate_median": 51.90,
    "modelled_ms": 4238.9,
    "heavy_tokens": 220,
    "distinct_slots": 3,
}

failures: list[str] = []


def check(name: str, got, want, tol: float = 0.0) -> None:
    ok = (abs(got - want) <= tol) if (tol and got is not None) else (got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} got {got!r:>12}  want {want!r}")
    if not ok:
        failures.append(name)


def test_fixture_parses_to_known_values() -> None:
    print("\nfixture parses to hand-checked values")
    r = parse(FIXTURE)

    check("requests", r["requests"], EXPECTED["requests"])
    check("lcp_selections", r["lcp_selections"], EXPECTED["lcp_selections"])
    check("lru_fallbacks", r["lru_fallbacks"], EXPECTED["lru_fallbacks"])
    check("threshold", r["threshold"], EXPECTED["threshold"], 1e-9)

    s = r["similarity"]
    check("similarity.median", s["median"], EXPECTED["similarity_median"], 1e-9)
    check("similarity.min", s["min"], EXPECTED["similarity_min"], 1e-9)
    check("similarity.max", s["max"], EXPECTED["similarity_max"], 1e-9)

    p = r["prefill_tokens"]
    check("prefill.median", p["median"], EXPECTED["prefill_median"], 1e-9)
    check("prefill.min", p["min"], EXPECTED["prefill_min"])
    check("prefill.max", p["max"], EXPECTED["prefill_max"])
    check("prefill.over_100_count", p["over_100_count"], EXPECTED["over_100_count"])
    check("prefill.over_100_pct", p["over_100_pct"], EXPECTED["over_100_pct"], 1e-9)

    q = r["prefill_rate"]
    check("rate.median_tok_per_s", q["median_tok_per_s"], EXPECTED["rate_median"], 0.01)
    check("rate.median_heavy_tokens", q["median_heavy_tokens"], EXPECTED["heavy_tokens"])
    check("rate.modelled_ms", q["modelled_ms_heavy_prefill"], EXPECTED["modelled_ms"], 0.2)

    check("distinct_slots_used", r["distinct_slots_used"], EXPECTED["distinct_slots"])


def test_noise_lines_are_ignored() -> None:
    """Prose, headers and non-slot log lines must not become data."""
    print("\nnon-slot lines are ignored")
    r = parse(FIXTURE)
    # The fixture contains a prose header and model-loading lines. If any of
    # those leaked into the counts, requests would exceed 4.
    check("requests unaffected by prose", r["requests"], 4)
    check("no phantom similarities", r["similarity"]["n"], 3)


def test_format_drift_is_visible(tmp: str) -> None:
    """THE point of this file.

    A log whose format has drifted must produce an obviously-empty result, not a
    plausible-looking one. Zero requests against a non-empty file is the signal
    that the parser needs updating for a new llama.cpp release.
    """
    print("\nformat drift produces an empty result, not a plausible one")
    drifted = (
        "slot get_available: id 0 | task -1 | picked slot via lcp_sim=0.716 thr=0.100\n"
        "slot print_timings: id 0 | task 2 | prompt_eval = 4238.90ms tokens=220\n"
    ) * 20
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(drifted)
    r = parse(tmp)
    check("drifted log -> 0 requests", r["requests"], 0)
    check("drifted log -> 0 lcp", r["lcp_selections"], 0)
    check("drifted log -> no median", r["prefill_tokens"]["median"], None)
    print("       (a non-empty file yielding 0 requests is the drift signal;")
    print("        callers must treat that as 'parser broken', not 'no evidence')")


def test_missing_file_reports_error() -> None:
    print("\nmissing file reports an error rather than empty data")
    r = parse(os.path.join(HERE, "fixtures", "does_not_exist.log"))
    check("error key present", "error" in r, True)
    check("no fabricated requests", r.get("requests"), None)


def main() -> int:
    if not os.path.exists(FIXTURE):
        print(f"fixture missing: {FIXTURE}", file=sys.stderr)
        return 2

    tmp = os.path.join(HERE, "fixtures", "_drifted.tmp.log")
    try:
        test_fixture_parses_to_known_values()
        test_noise_lines_are_ignored()
        test_format_drift_is_visible(tmp)
        test_missing_file_reports_error()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print()
    if failures:
        print(f"  {len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("  all parse_slot_log checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

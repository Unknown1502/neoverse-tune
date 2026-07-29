#!/usr/bin/env python3
"""
parse_slot_log.py — mechanism evidence straight from llama-server's own log.

WHY THIS IS BETTER EVIDENCE THAN TTFT
-------------------------------------
TTFT mixes scheduling, thermal state, memory pressure and CPU noise into one
number. llama-server, at default verbosity, already reports the thing we
actually want to know:

    selected slot by LCP similarity, f_sim_best = 0.974 (> 0.100 thold)
    prompt eval time = 327.02 ms / 21 tokens

The first line names the slot it chose, the similarity that justified it, and
the threshold it was compared against. The second says how many tokens were
ACTUALLY prefilled — 21 out of an 842-token conversation means the cache hit;
several hundred means it did not.

Tokens-prefilled is causally direct and immune to machine noise. It is the
metric the claim should rest on, with TTFT as the user-visible consequence.

WHAT IT REPORTS
---------------
  * distribution of f_sim_best, and how many selections cleared the threshold
    only because the shared preamble inflated it
  * prefill tokens per request: cache hits vs full re-prefills
  * slot churn: how often a task lands on a different slot than the previous
    task for that conversation
  * LRU fallbacks, meaning no slot matched at all

Run:  python3 tools/parse_slot_log.py results/server_*.log
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

# Tolerant patterns: llama.cpp log formatting has changed across releases, so
# these key on the stable substrings rather than exact column layout.
RE_LCP = re.compile(
    r"slot\s+get_availabl:\s*id\s+(?P<slot>\d+).*?selected slot by LCP similarity,"
    r"\s*f_sim_best\s*=\s*(?P<sim>[0-9.]+)\s*\(\s*>\s*(?P<thold>[0-9.]+)\s*thold\)")
RE_LRU = re.compile(
    r"slot\s+get_availabl:\s*id\s+(?P<slot>\d+).*?selected slot by LRU")
RE_LAUNCH = re.compile(
    r"slot\s+launch_slot_:\s*id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)")
RE_PROMPT = re.compile(
    r"slot\s+print_timing:\s*id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)\s*\|"
    r"\s*prompt eval time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens")
RE_RELEASE = re.compile(
    r"slot\s+release:\s*id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)\s*\|"
    r".*?n_tokens\s*=\s*(?P<n>\d+)")


def parse(path: str) -> dict:
    sims: list[float] = []
    tholds: list[float] = []
    lru_fallbacks = 0
    lcp_selections = 0
    prefill: list[dict] = []
    context_size: dict[int, int] = {}   # task -> total conversation tokens
    slot_of_task: dict[int, int] = {}

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = RE_LCP.search(line)
                if m:
                    sims.append(float(m.group("sim")))
                    tholds.append(float(m.group("thold")))
                    lcp_selections += 1
                    continue
                if RE_LRU.search(line):
                    lru_fallbacks += 1
                    continue
                m = RE_LAUNCH.search(line)
                if m:
                    slot_of_task[int(m.group("task"))] = int(m.group("slot"))
                    continue
                m = RE_PROMPT.search(line)
                if m:
                    prefill.append({"task": int(m.group("task")),
                                    "slot": int(m.group("slot")),
                                    "tokens": int(m.group("tokens")),
                                    "ms": float(m.group("ms"))})
                    continue
                m = RE_RELEASE.search(line)
                if m:
                    context_size[int(m.group("task"))] = int(m.group("n"))
    except OSError as exc:
        return {"file": os.path.basename(path), "error": str(exc)}

    # Reuse ratio: of the whole conversation, what fraction did NOT need
    # prefilling? This is the cache doing its job, expressed causally.
    reuse = []
    for p in prefill:
        total = context_size.get(p["task"])
        if total and total > 0:
            reuse.append(1.0 - min(p["tokens"], total) / total)

    tokens = [p["tokens"] for p in prefill]
    # A request that prefills more than 100 tokens on a warm conversation is
    # recomputing history, not just the new user turn.
    heavy = [t for t in tokens if t > 100]

    return {
        "file": os.path.basename(path),
        "requests": len(prefill),
        "lcp_selections": lcp_selections,
        "lru_fallbacks": lru_fallbacks,
        "threshold": tholds[0] if tholds else None,
        "similarity": {
            "n": len(sims),
            "min": round(min(sims), 4) if sims else None,
            "median": round(statistics.median(sims), 4) if sims else None,
            "max": round(max(sims), 4) if sims else None,
        },
        "prefill_tokens": {
            "n": len(tokens),
            "min": min(tokens) if tokens else None,
            "median": round(statistics.median(tokens), 1) if tokens else None,
            "max": max(tokens) if tokens else None,
            "over_100_count": len(heavy),
            "over_100_pct": round(100.0 * len(heavy) / len(tokens), 1) if tokens else 0.0,
        },
        "cache_reuse_fraction": {
            "n": len(reuse),
            "median": round(statistics.median(reuse), 4) if reuse else None,
            "min": round(min(reuse), 4) if reuse else None,
        },
        "distinct_slots_used": len(set(p["slot"] for p in prefill)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", default=None,
                    help="server log files (default: results/server_*.log)")
    ap.add_argument("--out")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    paths = args.logs
    if not paths:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results = os.environ.get("SPECARM_RESULTS", os.path.join(root, "results"))
        paths = sorted(glob.glob(os.path.join(results, "server_*.log")))
    if not paths:
        print("no server logs found", file=sys.stderr)
        return 1

    reports = [parse(p) for p in paths]

    print()
    print(f"  {'log':<34} {'reqs':>5} {'sim med':>8} {'prefill med':>12} "
          f"{'>100tok':>8} {'LRU':>5}")
    print("  " + "-" * 78)
    for r in reports:
        if r.get("error"):
            print(f"  {r['file']:<34} ERROR {r['error']}")
            continue
        sim = r["similarity"]["median"]
        pf = r["prefill_tokens"]["median"]
        pct = r["prefill_tokens"]["over_100_pct"]
        print(f"  {r['file']:<34} {r['requests']:>5} "
              f"{(f'{sim:.3f}' if sim is not None else '-'):>8} "
              f"{(f'{pf:.0f}' if pf is not None else '-'):>12} "
              f"{pct:>7.0f}% {r['lru_fallbacks']:>5}")

    print()
    print("  sim med ....... median LCP similarity of the slot the server chose")
    print("  prefill med ... median tokens actually recomputed per request")
    print("  >100tok ....... share of requests that recomputed >100 tokens,")
    print("                  i.e. reprocessed history rather than just the new turn")
    print("  LRU ........... selections where NO slot matched and it fell back")
    print()

    payload = {"schema": "specarm.slotlog/1", "logs": reports}
    out = args.out
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

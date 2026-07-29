#!/usr/bin/env python3
"""
probe_multitenant.py — does prefix caching survive more than one user?

WHY THIS EXISTS
---------------
probe_prefill.py showed llama.cpp reuses a cached prefix beautifully — for ONE
conversation on an idle server. That is not how anything is deployed. A real
agent service has many users hitting the same agent, and llama.cpp caches KV per
SLOT. If two conversations land on the same slot, each evicts the other, and
every turn pays the full cold prefill again.

That failure mode is invisible to single-conversation benchmarks, which is how
nearly everyone benchmarks.

THE EXPERIMENT
--------------
Run C conversations that share an identical system+tools preamble but have
different histories, interleaving their turns round-robin. Compare the resulting
TTFT against the single-conversation baseline.

    TTFT stays near baseline    -> caching is multi-tenant safe. Direction dead.
    TTFT climbs toward cold     -> cache thrashing. Real serving optimization.

KILL CONDITION, STATED IN ADVANCE
---------------------------------
If interleaved TTFT is within 1.5x of the single-conversation baseline, there is
no meaningful thrash and this whole direction should be abandoned rather than
rescued with a more elaborate test.

USAGE
-----
    # baseline first (single conversation)
    python3 tools/probe_multitenant.py --url http://127.0.0.1:8081 --agents 1
    # then the real test
    python3 tools/probe_multitenant.py --url http://127.0.0.1:8081 --agents 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

from probe_prefill import SYSTEM, TURNS  # identical preamble, same workload

# Distinct opening context per tenant, so histories genuinely differ after the
# shared preamble. This is the realistic shape: one agent, many users.
TENANT_FLAVOR = [
    "I am travelling for a wedding and have a strict budget.",
    "I need wheelchair accessible options throughout.",
    "This is a business trip; prioritise schedule over cost.",
    "I am travelling with a toddler and need family facilities.",
    "I am a vegetarian and need meal options confirmed.",
    "I have a tight connection and cannot risk delays.",
]


def one_turn(url: str, messages: list[dict], max_tokens: int,
             timeout: float) -> tuple[float, int, str]:
    """Return (ttft_ms, prompt_tokens, content)."""
    body = json.dumps({
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})

    start = time.perf_counter()
    first = None
    ptok = 0
    content: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens"):
                ptok = int(usage["prompt_tokens"])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter()
                content.append(delta["content"])
    if first is None:
        raise RuntimeError("no tokens streamed")
    return (first - start) * 1000.0, ptok, "".join(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4, help="turns per agent")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    # Each tenant shares the preamble, diverges immediately after it.
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": TENANT_FLAVOR[i % len(TENANT_FLAVOR)]},
              {"role": "assistant", "content": '{"final": "Understood."}'}]
             for i in range(args.agents)]

    print(f"\n  Multi-tenant prefix probe — {args.agents} agent(s), "
          f"{args.turns} turns each, interleaved round-robin")
    print(f"  All tenants share the same ~{len(SYSTEM)//4} token preamble.\n")
    print(f"  {'round':>5}  {'agent':>5}  {'prompt tok':>10}  {'TTFT ms':>9}")
    print("  " + "-" * 40)

    rows = []
    try:
        for turn in range(args.turns):
            for a in range(args.agents):
                convs[a].append({"role": "user",
                                 "content": TURNS[turn % len(TURNS)]})
                ttft, ptok, text = one_turn(args.url, convs[a],
                                            args.max_tokens, args.timeout)
                convs[a].append({"role": "assistant", "content": text[:300] or "{}"})
                print(f"  {turn+1:>5}  {a:>5}  {ptok:>10}  {ttft:>9.1f}")
                rows.append({"round": turn + 1, "agent": a,
                             "prompt_tokens": ptok, "ttft_ms": round(ttft, 2)})
    except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"\n  request failed: {exc}", file=sys.stderr)
        return 2

    # Round 1 is cold for every tenant; rounds 2+ are where caching should pay.
    warm = [r["ttft_ms"] for r in rows if r["round"] > 1]
    cold = [r["ttft_ms"] for r in rows if r["round"] == 1]
    warm_median = statistics.median(warm) if warm else 0.0
    cold_median = statistics.median(cold) if cold else 0.0

    print("\n  " + "=" * 56)
    print(f"  round 1 median TTFT ....... {cold_median:.1f} ms")
    print(f"  rounds 2+ median TTFT ..... {warm_median:.1f} ms")
    if cold_median:
        print(f"  warm / cold ratio ......... {warm_median / cold_median:.3f}")

    report = {"schema": "specarm.multitenant/1", "agents": args.agents,
              "turns": args.turns, "rows": rows,
              "cold_median_ttft_ms": round(cold_median, 2),
              "warm_median_ttft_ms": round(warm_median, 2)}

    if args.agents == 1:
        print("\n  BASELINE recorded. Now re-run with --agents 4 and compare")
        print("  the rounds-2+ median against this one.")
        report["verdict"] = "BASELINE"
    else:
        print("\n  Compare the rounds-2+ median above against your --agents 1 run.")
        print("  Within ~1.5x  -> caching is multi-tenant safe; direction is dead.")
        print("  Much higher   -> slot thrashing; a real serving optimization exists.")
        report["verdict"] = "COMPARE_TO_BASELINE"
    print("  " + "=" * 56 + "\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

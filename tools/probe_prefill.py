#!/usr/bin/env python3
"""
probe_prefill.py — OPPORTUNITY SIZER. Run this before building anything.

THE QUESTION
------------
An agent's prompt is [system + tool schemas][history][new turn]. The first
segment is byte-identical on every turn. Does the server re-process it each
time, or does it reuse the cached KV?

If it re-processes: time-to-first-token grows with TOTAL prompt length, and
there is a large optimization sitting there.
If it reuses: TTFT grows only with NEW tokens, the inefficiency does not
exist, and AgentArm should not be built.

THE DISCRIMINATOR
-----------------
Across turns we track two ratios:

    ms per TOTAL prompt token     flat  => re-prefilling everything (NO REUSE)
    ms per NEW prompt token       flat  => reusing the prefix    (REUSE WORKS)

Whichever stays flat as the conversation grows tells you what the server does.
This is not a subtle statistical question — the two hypotheses predict wildly
different curves.

WHY IT RUNS ANYWHERE
--------------------
This measures llama.cpp's caching behaviour, which is architecture-independent.
Run it on x86 today. Arm matters for the final numbers, not for sizing the
opportunity.

USAGE
-----
    python3 tools/probe_prefill.py --url http://127.0.0.1:8080 --turns 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# A realistic agent preamble: instructions plus tool schemas. This is the block
# that never changes and therefore should never be recomputed.
TOOLS = [
    {"name": "search_flights",
     "description": "Search available flights between two airports on a date.",
     "parameters": {"origin": "IATA code", "destination": "IATA code",
                    "date": "YYYY-MM-DD", "passengers": "integer",
                    "cabin": "economy|premium|business|first",
                    "max_stops": "integer", "refundable": "boolean"}},
    {"name": "book_hotel",
     "description": "Reserve a hotel room in a city for a date range.",
     "parameters": {"city": "string", "checkin": "YYYY-MM-DD",
                    "checkout": "YYYY-MM-DD", "guests": "integer",
                    "rooms": "integer", "star_rating": "integer",
                    "breakfast": "boolean", "cancellable": "boolean"}},
    {"name": "get_weather",
     "description": "Weather forecast for a city on a date.",
     "parameters": {"city": "string", "date": "YYYY-MM-DD", "units": "c|f"}},
    {"name": "convert_currency",
     "description": "Convert an amount between two currencies.",
     "parameters": {"amount": "number", "from": "ISO code", "to": "ISO code"}},
    {"name": "create_calendar_event",
     "description": "Create an event in the user's calendar.",
     "parameters": {"title": "string", "start": "ISO8601", "end": "ISO8601",
                    "location": "string", "attendees": "list of emails",
                    "reminder_minutes": "integer"}},
]

SYSTEM = (
    "You are a meticulous travel planning agent operating in a tool-calling "
    "loop. On each turn you inspect the conversation so far, decide whether a "
    "tool call is required, and if so emit exactly one JSON object and nothing "
    "else. Never invent tool results. Never call a tool you have already called "
    "with identical arguments. If every necessary fact is known, produce a final "
    "answer instead of a tool call. Always prefer refundable options when the "
    "user has not stated a preference. Treat all dates as ISO-8601. "
    "Available tools:\n" + json.dumps(TOOLS, indent=2) + "\n"
    "Respond with a single JSON object of the form "
    '{"tool": "<name>", "arguments": {...}} or {"final": "<answer>"}.'
)

# A scripted multi-step task. Content matters less than the shape: each turn
# appends to a prompt whose opening segment is unchanged.
TURNS = [
    "Plan a trip: Delhi to Singapore, 2026-09-14, two passengers, economy.",
    "Good. Now find a hotel near the airport for two nights from that date.",
    "What will the weather be like on arrival?",
    "Convert the total cost so far to Indian rupees.",
    "Add the outbound flight to my calendar with a 3 hour reminder.",
    "Summarise the whole itinerary for me.",
]


def one_turn(url: str, messages: list[dict], max_tokens: int,
             timeout: float) -> tuple[float, float, int, int]:
    """Send a conversation, return (ttft_ms, total_ms, prompt_tokens, out_tokens)."""
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "cache_prompt": True,   # ask for caching where the server supports it
    }).encode()

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )

    start = time.perf_counter()
    first = None
    out_tokens = 0
    prompt_tokens = 0
    content = []

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
                prompt_tokens = int(usage["prompt_tokens"])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter()
                out_tokens += 1
                content.append(delta["content"])

    end = time.perf_counter()
    if first is None:
        raise RuntimeError("server streamed no content")

    return ((first - start) * 1000.0, (end - start) * 1000.0,
            prompt_tokens, out_tokens), "".join(content)


def approx_tokens(messages: list[dict]) -> int:
    """Rough token count when the server does not report usage. ~4 chars/token
    is crude but we only need the SHAPE of the growth curve, not exact counts."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, chars // 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--label", default="default", help="tag for this configuration")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    messages = [{"role": "system", "content": SYSTEM}]
    rows = []
    prev_total = 0

    print(f"\n  Agent prefill probe — {args.turns} turns, label='{args.label}'")
    print(f"  system+tools preamble: ~{approx_tokens(messages)} tokens\n")
    print(f"  {'turn':>4}  {'prompt tok':>10}  {'new tok':>8}  {'TTFT ms':>9}"
          f"  {'ms/total':>9}  {'ms/new':>9}")
    print("  " + "-" * 62)

    for i in range(args.turns):
        messages.append({"role": "user", "content": TURNS[i % len(TURNS)]})
        try:
            (ttft, total_ms, ptok, otok), text = one_turn(
                args.url, messages, args.max_tokens, args.timeout)
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
            print(f"\n  request failed on turn {i+1}: {exc}", file=sys.stderr)
            return 2

        if not ptok:
            ptok = approx_tokens(messages)
        new_tok = max(1, ptok - prev_total)

        ms_per_total = ttft / ptok
        ms_per_new = ttft / new_tok
        print(f"  {i+1:>4}  {ptok:>10}  {new_tok:>8}  {ttft:>9.1f}"
              f"  {ms_per_total:>9.3f}  {ms_per_new:>9.3f}")

        rows.append({"turn": i + 1, "prompt_tokens": ptok, "new_tokens": new_tok,
                     "ttft_ms": round(ttft, 2), "total_ms": round(total_ms, 2),
                     "out_tokens": otok,
                     "ms_per_total_token": round(ms_per_total, 4),
                     "ms_per_new_token": round(ms_per_new, 4)})

        messages.append({"role": "assistant", "content": text[:400] or "{}"})
        prev_total = ptok

    # ---------------------------------------------------------------- verdict
    #
    # The discriminator is turn 1 versus every later turn, NOT prompt growth.
    # A realistic agent preamble dominates the prompt, so total length barely
    # grows per turn — an earlier version of this script keyed on growth and
    # returned INCONCLUSIVE on data that was in fact unambiguous.
    #
    # Physics: turn 1 MUST prefill everything, nothing is cached yet. Later
    # turns share that whole preamble. So:
    #
    #   later TTFT ~= turn-1 TTFT   -> the preamble is recomputed every turn
    #   later TTFT << turn-1 TTFT   -> the preamble is being reused
    #
    first = rows[0]
    later = rows[1:]
    later_ttfts = sorted(r["ttft_ms"] for r in later)
    later_median = later_ttfts[len(later_ttfts) // 2] if later_ttfts else 0.0
    ratio = (later_median / first["ttft_ms"]) if first["ttft_ms"] else 0.0

    # How much of a later turn's prefill is spent on tokens already seen.
    last = rows[-1]
    redundant_frac = 1.0 - (last["new_tokens"] / last["prompt_tokens"]) \
        if last["prompt_tokens"] else 0.0

    print("\n  " + "=" * 62)
    print(f"  turn 1 TTFT ......... {first['ttft_ms']:.1f} ms  (nothing cacheable yet)")
    print(f"  later turns median .. {later_median:.1f} ms")
    print(f"  ratio ............... {ratio:.3f}")
    print(f"  redundant prefix .... {redundant_frac * 100:.1f}% of the final prompt "
          f"was already sent")

    if len(later) < 2:
        verdict = "INCONCLUSIVE"
        note = "need at least 3 turns — increase --turns"
    elif first["ttft_ms"] < 5.0:
        verdict = "INCONCLUSIVE"
        note = ("turn-1 TTFT is too small to measure against; use a larger model "
                "or a longer preamble")
    elif ratio >= 0.6:
        verdict = "NO_REUSE"
        note = (f"Later turns cost about as much as the first ({ratio:.0%}), even "
                f"though {redundant_frac:.0%} of the prompt was already processed. "
                f"The preamble is being recomputed every turn. "
                f"THE OPPORTUNITY IS REAL — build it.")
    elif ratio <= 0.3:
        verdict = "REUSE_WORKS"
        note = (f"Later turns cost {ratio:.0%} of the first. The server already "
                f"reuses the cached prefix. DO NOT build a prefix cache — "
                f"pick a different optimization.")
    else:
        verdict = "PARTIAL_REUSE"
        note = (f"Later turns cost {ratio:.0%} of the first: partial reuse. "
                f"Some headroom remains but the win is smaller than a naive "
                f"baseline suggests. Size it before committing two weeks.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {note}")
    print("  " + "=" * 62 + "\n")

    report = {"schema": "specarm.prefill_probe/1", "label": args.label,
              "turns": rows,
              "turn1_ttft_ms": first["ttft_ms"],
              "later_median_ttft_ms": round(later_median, 2),
              "later_over_first_ratio": round(ratio, 4),
              "redundant_prefix_fraction": round(redundant_frac, 4),
              "verdict": verdict, "note": note}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.out}\n")

    return 0 if verdict in ("NO_REUSE", "PARTIAL_REUSE") else 1


if __name__ == "__main__":
    sys.exit(main())

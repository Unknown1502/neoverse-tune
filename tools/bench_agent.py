#!/usr/bin/env python3
"""
bench_agent.py — one multi-tenant agent-serving measurement, done properly.

Supersedes probe_multitenant.py, which was an exploration tool. This one is
built for numbers that go in front of a judge:

  * EXACT token counts from the server's own tokenizer, not chars//4
  * per-round, per-tenant TTFT with the request order recorded
  * warm/cold separation, since round 1 is cold for every tenant by definition
  * raw samples retained so confidence intervals are possible downstream

One invocation = one run against an already-running server. Repeats and server
lifecycle are run_matrix.py's job, because a repeat that reuses a warm server is
not an independent sample.

Usage:
    python3 tools/bench_agent.py --url http://127.0.0.1:8081 --agents 4 --out run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

import workloads
from probe_prefill import SYSTEM, TURNS

TENANT_FLAVOR = [
    "I am travelling for a wedding and have a strict budget.",
    "I need wheelchair accessible options throughout.",
    "This is a business trip; prioritise schedule over cost.",
    "I am travelling with a toddler and need family facilities.",
    "I am a vegetarian and need meal options confirmed.",
    "I have a tight connection and cannot risk delays.",
    "I am afraid of flying and prefer daytime departures.",
    "I am a frequent flyer and want lounge access noted.",
]

# Per-tenant itineraries. Tenants MUST diverge in their actual turn content, not
# only in an opening line: if every tenant sends identical turn text, the
# longest-common-prefix between two different tenants is inflated far above what
# real traffic produces, foreign slots look more attractive than they should,
# and the measured regression is exaggerated. Each tenant therefore gets its own
# cities, dates, passenger counts and phrasing.
TENANT_TRIPS = [
    ("Delhi", "Singapore", "2026-09-14", "two", "DEL", "SIN"),
    ("London", "Tokyo", "2026-10-02", "one", "LHR", "HND"),
    ("Mumbai", "Dubai", "2026-08-30", "four", "BOM", "DXB"),
    ("San Francisco", "Seoul", "2026-11-11", "three", "SFO", "ICN"),
    ("Berlin", "Reykjavik", "2026-12-01", "one", "BER", "KEF"),
    ("Sydney", "Auckland", "2026-09-05", "two", "SYD", "AKL"),
    ("Toronto", "Lisbon", "2026-10-19", "five", "YYZ", "LIS"),
    ("Nairobi", "Amsterdam", "2026-11-27", "two", "NBO", "AMS"),
]


def tenant_turn(tenant: int, turn: int) -> str:
    """Turn text for one tenant, distinct from every other tenant's."""
    origin, dest, date, pax, o_iata, d_iata = TENANT_TRIPS[tenant % len(TENANT_TRIPS)]
    templates = [
        f"Plan a trip: {origin} to {dest} on {date} for {pax} passengers, economy.",
        f"Now find a hotel in {dest} near {d_iata} for two nights from {date}.",
        f"What will the weather in {dest} be like when we land on {date}?",
        f"Convert the running total for this {origin}-{dest} trip into rupees.",
        f"Add the {o_iata} to {d_iata} departure to my calendar with a 3 hour reminder.",
        f"Summarise the full {origin} to {dest} itinerary for {pax} travellers.",
    ]
    return templates[turn % len(templates)]


def _post(url: str, path: str, payload: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class Tokenizer:
    """Exact token counts from the server that will actually process them.

    Tries the chat template first so the count includes role markers and other
    template scaffolding. Falls back to tokenizing concatenated content, and
    finally to a character heuristic. Which path was used is recorded, because
    a token count from a fallback is not the same claim as one from the server.
    """

    def __init__(self, url: str):
        self.url = url
        self.method = "unknown"
        self._probe()

    def _probe(self) -> None:
        probe = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
        try:
            r = _post(self.url, "/apply-template", {"messages": probe})
            if isinstance(r, dict) and r.get("prompt"):
                self.method = "apply_template+tokenize"
                return
        except Exception:
            pass
        try:
            r = _post(self.url, "/tokenize", {"content": "hello world"})
            if isinstance(r, dict) and isinstance(r.get("tokens"), list):
                self.method = "tokenize_concat"
                return
        except Exception:
            pass
        self.method = "char_heuristic"

    def count(self, messages: list[dict]) -> int:
        try:
            if self.method == "apply_template+tokenize":
                templated = _post(self.url, "/apply-template",
                                  {"messages": messages})["prompt"]
                return len(_post(self.url, "/tokenize",
                                 {"content": templated})["tokens"])
            if self.method == "tokenize_concat":
                blob = "\n".join(m.get("content", "") for m in messages)
                return len(_post(self.url, "/tokenize", {"content": blob})["tokens"])
        except Exception:
            pass
        return max(1, sum(len(m.get("content", "")) for m in messages) // 4)


def one_turn(url: str, messages: list[dict], max_tokens: int,
             timeout: float) -> tuple[float, float, int, str]:
    """Return (ttft_ms, total_ms, out_tokens, content)."""
    body = json.dumps({
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})

    start = time.perf_counter()
    first = None
    out = 0
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
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter()
                out += 1
                content.append(delta["content"])
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("no tokens streamed")
    return ((first - start) * 1000.0, (end - start) * 1000.0, out, "".join(content))


def wait_healthy(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def run(url: str, agents: int, turns: int, max_tokens: int,
        timeout: float, label: str, workload: str = "agent") -> dict:
    """One measurement run.

    `workload` selects the prompt SHAPE. It defaults to "agent", which is the
    original workload, so every result published before this parameter existed
    remains reproducible by the same call.

    The shape matters because inter-tenant similarity is roughly
    shared_preamble / total_prompt — a property of the prompts, not of
    llama.cpp. Comparing shapes is how we test whether the right threshold is a
    constant or a function of the workload.
    """
    wl = workloads.get(workload)
    tok = Tokenizer(url)

    # Seed each tenant with its private content BEFORE turn 1. For the agent
    # workload that is a one-line preference; for RAG it is the retrieved
    # document, which is the bulk of that tenant's prompt and is what drives
    # inter-tenant similarity down.
    convs = [[{"role": "system", "content": wl.system},
              {"role": "user", "content": wl.seed(i)},
              {"role": "assistant", "content": '{"final": "Understood."}'}]
             for i in range(agents)]

    preamble_tokens = tok.count([{"role": "system", "content": wl.system}])
    # Tokens of one tenant's private seed, so the shared fraction can be
    # reported rather than inferred from character counts.
    seed_tokens = (tok.count([{"role": "user", "content": wl.seed(0)}])
                   if agents else 0)
    rows = []
    seq = 0

    for rnd in range(turns):
        for a in range(agents):
            convs[a].append({"role": "user", "content": wl.turn(a, rnd)})
            ptok = tok.count(convs[a])
            ttft, total_ms, otok, text = one_turn(url, convs[a], max_tokens, timeout)
            convs[a].append({"role": "assistant", "content": text[:300] or "{}"})
            seq += 1
            rows.append({"seq": seq, "round": rnd + 1, "agent": a,
                         "prompt_tokens": ptok, "ttft_ms": round(ttft, 2),
                         "total_ms": round(total_ms, 2), "out_tokens": otok})

    warm = [r["ttft_ms"] for r in rows if r["round"] > 1]
    cold = [r["ttft_ms"] for r in rows if r["round"] == 1]
    final_ptok = rows[-1]["prompt_tokens"] if rows else 0
    return {
        "schema": "specarm.agentbench/2",
        "label": label, "agents": agents, "turns": turns,
        "workload": wl.name,
        "token_count_method": tok.method,
        "preamble_tokens": preamble_tokens,
        "seed_tokens": seed_tokens,
        "final_prompt_tokens": final_ptok,
        # Predicted inter-tenant similarity: the share of a tenant's final
        # prompt that every other tenant also sends. The threshold must sit
        # above this to separate tenants.
        "shared_fraction": round(preamble_tokens / final_ptok, 4) if final_ptok else None,
        "cold_ttft_ms": cold,
        "warm_ttft_ms": warm,
        "warm_median_ms": round(statistics.median(warm), 2) if warm else 0.0,
        "cold_median_ms": round(statistics.median(cold), 2) if cold else 0.0,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--workload", default="agent",
                    help="prompt shape: agent (default) or rag")
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    if not wait_healthy(args.url, 180):
        print(f"server at {args.url} never became healthy", file=sys.stderr)
        return 2

    try:
        result = run(args.url, args.agents, args.turns, args.max_tokens,
                     args.timeout, args.label, args.workload)
    except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 2

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    if not args.quiet:
        print(f"  {args.label}: warm median {result['warm_median_ms']:.1f} ms "
              f"(cold {result['cold_median_ms']:.1f} ms) · "
              f"preamble {result['preamble_tokens']} tok · "
              f"final prompt {result['final_prompt_tokens']} tok · "
              f"counts via {result['token_count_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

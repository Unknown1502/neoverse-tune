#!/usr/bin/env python3
"""
loadgen.py — serving-level load generator for llama-server on Arm64.

The kernel sweep measures a matmul. This measures a *service*: many concurrent
clients, streaming responses, tail latency. That is the regime Cloud AI is
actually judged in, and it is where the batch-size question stops being academic
— a server under concurrent load batches requests together, which is precisely
the regime where i8mm matmul kernels become reachable.

Workload is agentic tool-calling: prompts that force a strict JSON tool call.
That matters for two reasons. It is the workload the Cloud AI track names, and
schema-constrained output is highly predictable, which is what makes speculative
drafting pay off later.

Metrics, per request:
  ttft_ms     time to first token   (dominated by prefill = GEMM regime)
  tpot_ms     time per output token (dominated by decode  = GEMV regime)
  total_ms    end to end
  out_tokens  completion length

Reported as p50/p90/p95/p99 plus aggregate output tokens/sec, because a mean
latency under concurrency hides exactly the behaviour you care about.

Standard library only — no pip install on the benchmark host, nothing to drift.

Usage:
  python3 tools/loadgen.py --url http://127.0.0.1:8080 --concurrency 8 --requests 64
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Agentic tool-call workload -------------------------------------------
# Each prompt demands a single strict JSON object. Deterministic, schema-shaped
# output: the realistic agent case, and the one drafting predicts well.
TOOL_SCHEMA = {
    "name": "search_flights",
    "arguments": {"origin": "str", "destination": "str",
                  "date": "YYYY-MM-DD", "passengers": "int"},
}

REQUESTS = [
    "Book me a flight from Delhi to Singapore on 2026-09-14 for 2 people.",
    "I need to get from London to Tokyo on 2026-10-02, travelling alone.",
    "Find flights, Mumbai to Dubai, 2026-08-30, four passengers.",
    "Three of us are flying San Francisco to Seoul on 2026-11-11.",
    "One seat, Berlin to Reykjavik, 2026-12-01 please.",
    "Get me Sydney to Auckland for 2026-09-05, two travellers.",
]

SYSTEM = (
    "You are a tool-calling agent. Respond with exactly one JSON object matching "
    f"this schema and nothing else: {json.dumps(TOOL_SCHEMA)}"
)


class Result:
    __slots__ = ("ok", "ttft_ms", "tpot_ms", "total_ms", "out_tokens", "error")

    def __init__(self, ok, ttft_ms=0.0, tpot_ms=0.0, total_ms=0.0,
                 out_tokens=0, error=""):
        self.ok, self.error = ok, error
        self.ttft_ms, self.tpot_ms = ttft_ms, tpot_ms
        self.total_ms, self.out_tokens = total_ms, out_tokens


def one_request(url: str, prompt: str, max_tokens: int, timeout: float) -> Result:
    """Issue one streaming chat completion and time the token boundaries."""
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,          # deterministic: we are timing, not sampling
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    start = time.perf_counter()
    first_token_at = None
    tokens = 0

    try:
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
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    tokens += 1
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return Result(False, error=f"{type(exc).__name__}: {exc}")

    end = time.perf_counter()
    if first_token_at is None:
        return Result(False, error="no tokens streamed")

    ttft = (first_token_at - start) * 1000.0
    decode_ms = (end - first_token_at) * 1000.0
    # Time per output token, excluding the first (that one is prefill).
    tpot = decode_ms / max(tokens - 1, 1)
    return Result(True, ttft, tpot, (end - start) * 1000.0, tokens)


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Explicit, so results do not depend on the
    interpolation convention of whatever numpy version is installed."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {"metric": name, "n": 0}
    return {
        "metric": name, "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(pct(values, 50), 3),
        "p90": round(pct(values, 90), 3),
        "p95": round(pct(values, 95), 3),
        "p99": round(pct(values, 99), 3),
        "max": round(max(values), 3),
    }


def wait_for_server(url: str, timeout_s: float) -> bool:
    """Poll /health until the model is loaded. A benchmark that starts while
    weights are still being paged in measures disk, not inference."""
    deadline = time.time() + timeout_s
    health = url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=4,
                    help="discarded requests to page in weights and warm caches")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--label", default="run", help="tag recorded in the output")
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    if not wait_for_server(args.url, 120):
        print(f"server at {args.url} never became healthy", file=sys.stderr)
        return 2

    def fire(i: int) -> Result:
        return one_request(args.url, REQUESTS[i % len(REQUESTS)],
                           args.max_tokens, args.timeout)

    if args.warmup > 0:
        print(f"[loadgen] warmup: {args.warmup} requests (discarded)", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(as_completed([pool.submit(fire, i) for i in range(args.warmup)]))

    print(f"[loadgen] {args.requests} requests @ concurrency {args.concurrency}",
          file=sys.stderr)

    results: list[Result] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(fire, i) for i in range(args.requests)]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_out = sum(r.out_tokens for r in ok)

    report = {
        "schema": "specarm.serve/1",
        "label": args.label,
        "concurrency": args.concurrency,
        "requests_attempted": args.requests,
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "wall_seconds": round(wall, 3),
        # The headline serving number: tokens actually delivered per wall second
        # across all clients.
        "output_tokens_per_sec": round(total_out / wall, 3) if wall else 0.0,
        "completed_requests_per_sec": round(len(ok) / wall, 3) if wall else 0.0,
        "total_output_tokens": total_out,
        "ttft_ms": summarize("ttft_ms", [r.ttft_ms for r in ok]),
        "tpot_ms": summarize("tpot_ms", [r.tpot_ms for r in ok]),
        "total_ms": summarize("total_ms", [r.total_ms for r in ok]),
        "errors": sorted({r.error for r in failed})[:5],
    }

    out_path = args.out
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))

    if failed:
        print(f"\n[loadgen] WARNING: {len(failed)}/{args.requests} requests failed. "
              f"A serving result with failures is not a throughput number.",
              file=sys.stderr)
        # Partial failure must not look like success to a CI gate.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

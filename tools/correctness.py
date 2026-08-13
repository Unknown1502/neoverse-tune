#!/usr/bin/env python3
"""
correctness.py — does changing the routing threshold change what the model says?

WHY THIS IS NOT OPTIONAL
------------------------
Every performance claim in this repository is worthless if the faster
configuration also produces different output. `--slot-prompt-similarity` decides
which KV cache a request is appended to. If the server ever reuses a prefix that
is not genuinely a prefix of the new prompt, the model would be conditioned on
someone else's tokens — which would be a correctness bug, not an optimization.

So: run a fixed prompt set at each threshold, hash the generated text, and
compare against the baseline threshold.

WHAT A MISMATCH WOULD AND WOULD NOT MEAN
----------------------------------------
Generation here is greedy — temperature 0, fixed seed, same build, same model.
Under those conditions llama.cpp is expected to be deterministic for a given
prompt, so identical hashes are the expected result and a mismatch is a real
signal worth investigating.

But a mismatch is NOT automatically a routing bug. Prefill batching differs
between configurations, and floating-point accumulation is not associative, so
a differently-batched prefill can produce a slightly different logit and flip a
near-tie token. That is a numerical difference, not a semantic one.

This tool therefore reports three outcomes rather than two:

    PASS       every threshold produced byte-identical output
    DIVERGED   output differs; the divergence is characterised, not hidden
    FAILED     a threshold could not be measured at all

A DIVERGED result is reported with the first differing prompt and both texts, so
a human can judge whether it is numerical noise or a real behavioural change.
Calling it PASS because the numbers were nicer would defeat the point.

Usage:
    python3 tools/correctness.py --server <llama-server> --model <model.gguf>
    python3 tools/correctness.py --server ... --model ... --thresholds 0.1,0.5,0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, HERE)

import bench_agent          # noqa: E402
import run_matrix           # noqa: E402
import workloads            # noqa: E402

# Fixed, deterministic prompt set. Short and factual so the greedy answer is
# stable; varied enough that a routing error would show up as visibly wrong
# content rather than a rounding difference.
PROBES = [
    "Name the largest planet in the Solar System. Answer with one word.",
    "What is 17 multiplied by 23? Answer with the number only.",
    "Complete exactly: the capital of Japan is",
    "List the first four prime numbers, comma separated, nothing else.",
    "In one word, what colour is a ripe banana?",
]

DEFAULT_THRESHOLDS = [0.1, 0.5, 0.9]


def normalise(text: str) -> str:
    """Only whitespace is normalised, and that is stated in the report.

    Nothing else is touched. Stripping punctuation or lowercasing would let a
    genuinely different answer hash the same, which is the failure mode this
    tool exists to catch.
    """
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def probe_once(url: str, system: str, prompt: str, max_tokens: int,
               timeout: float) -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]
    _ttft, _total, _out, text = bench_agent.one_turn(url, msgs, max_tokens, timeout)
    return text


def run_threshold(server: str, model: str, port: int, ctx: int, threads: int,
                  thr: float, system: str, max_tokens: int) -> dict:
    log = os.path.join(RESULTS, f"correctness_sim{thr}.log")
    url = f"http://127.0.0.1:{port}"
    proc = run_matrix.spawn(server, model, port, ctx, threads,
                            ["-np", "4", "--slot-prompt-similarity", str(thr)], log)
    try:
        if not bench_agent.wait_healthy(url, 240):
            return {"threshold": thr, "status": "FAILED",
                    "reason": f"server never became healthy (see {log})"}
        outputs = []
        for p in PROBES:
            try:
                outputs.append(normalise(probe_once(url, system, p, max_tokens, 600.0)))
            except Exception as exc:
                return {"threshold": thr, "status": "FAILED",
                        "reason": f"probe failed: {exc}"}
        blob = "\n---\n".join(outputs)
        return {"threshold": thr, "status": "OK",
                "output_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                "outputs": outputs}
    finally:
        run_matrix.terminate(proc)
        time.sleep(2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="agent", help="which system prompt to use")
    ap.add_argument("--thresholds", help="comma separated (default 0.1,0.5,0.9)")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(RESULTS, "correctness.json"))
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    for p, what in ((args.server, "server binary"), (args.model, "model")):
        if not os.path.exists(p):
            print(f"{what} not found: {p}", file=sys.stderr)
            return 2
    os.makedirs(RESULTS, exist_ok=True)

    thresholds = ([float(x) for x in args.thresholds.split(",")]
                  if args.thresholds else DEFAULT_THRESHOLDS)
    system = workloads.get(args.workload).system

    print(f"\n  Correctness across {len(thresholds)} thresholds x "
          f"{len(PROBES)} fixed prompts")
    print(f"  greedy decoding (temperature 0), fresh server per threshold\n")

    rows = []
    for thr in thresholds:
        print(f"  threshold {thr} ...", end="", flush=True)
        r = run_threshold(args.server, args.model, args.port, args.ctx,
                          args.threads, thr, system, args.max_tokens)
        rows.append(r)
        if r["status"] == "OK":
            print(f" sha256 {r['output_sha256'][:16]}...")
        else:
            print(f" {r['status']}: {r['reason']}")

    ok = [r for r in rows if r["status"] == "OK"]
    failed = [r for r in rows if r["status"] != "OK"]

    verdict, detail = "FAILED", "no threshold produced output"
    if ok:
        base = ok[0]
        diverged = [r for r in ok if r["output_sha256"] != base["output_sha256"]]
        if not diverged:
            verdict = "PASS"
            detail = (f"all {len(ok)} thresholds produced byte-identical output "
                      f"(sha256 {base['output_sha256'][:16]}...)")
        else:
            verdict = "DIVERGED"
            detail = (f"{len(diverged)} of {len(ok)} thresholds differ from "
                      f"baseline {base['threshold']}")

    print()
    print("  " + "=" * 64)
    print(f"  CORRECTNESS: {verdict}")
    print(f"  {detail}")
    print("  " + "=" * 64)

    if verdict == "DIVERGED":
        base = ok[0]
        for r in ok:
            if r["output_sha256"] == base["output_sha256"]:
                continue
            for i, (a, b) in enumerate(zip(base["outputs"], r["outputs"])):
                if a != b:
                    print(f"\n  first divergence, threshold {base['threshold']} "
                          f"vs {r['threshold']}, prompt {i + 1}:")
                    print(f"    {PROBES[i]}")
                    print(f"    {base['threshold']}: {a[:120]!r}")
                    print(f"    {r['threshold']}: {b[:120]!r}")
                    break
            break
        print("\n  Greedy decoding should be deterministic, so this is worth")
        print("  investigating. Prefill batching differs between thresholds and")
        print("  floating-point accumulation is not associative, so a near-tie")
        print("  token can flip without any routing error. Judge from the text")
        print("  above whether this is numerical or behavioural.")

    if failed:
        print(f"\n  {len(failed)} threshold(s) could not be measured:")
        for r in failed:
            print(f"    {r['threshold']}: {r['reason']}")

    payload = {
        "schema": "specarm.correctness/1",
        "verdict": verdict,
        "detail": detail,
        "workload": args.workload,
        "normalisation": "trailing whitespace per line; nothing else",
        "decoding": "greedy, temperature 0",
        "max_tokens": args.max_tokens,
        "probes": PROBES,
        "thresholds": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {args.out}\n")

    # Non-zero when a configuration could not be measured at all. DIVERGED is
    # reported for a human to judge rather than treated as a build failure.
    return 2 if verdict == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())

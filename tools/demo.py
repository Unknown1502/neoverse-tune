#!/usr/bin/env python3
"""
demo.py — the finding, live, in about three minutes.

WHY THIS EXISTS SEPARATELY FROM run_matrix.py
---------------------------------------------
run_matrix.py is built for numbers that survive scrutiny: 6 configs, 5 repeats,
a fresh server each time, 47 minutes. That is the right shape for evidence and
the wrong shape for showing someone what is wrong. This runs exactly two
configurations, once each, and prints the difference as it happens.

It measures the same thing the same way — same tenants, same preamble, same
warm/cold split, same tokenizer. It is a smaller experiment, not a friendlier
one. The numbers it prints will be noisier than the published table because
n=1; that is stated on screen rather than hidden, and the published table
remains the claim.

WHAT IT SHOWS
-------------
Two runs against a real llama-server, back to back:

    default (0.10)   every tenant matches every slot -> tenants scatter
    fixed   (0.90)   only the tenant's own slot qualifies

Per request it prints time-to-first-token and the tokens the server actually
recomputed, read back from the server's own log at the end. The second number
is the one that matters: it is the cause, and it is immune to how busy the
machine is.

Usage:
    python3 tools/demo.py --server ./llama/llama-server.exe --model ./models/qwen1.5b.gguf
    python3 tools/demo.py --server ... --model ... --tenants 4 --turns 3
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, HERE)

import bench_agent          # noqa: E402
import run_matrix           # noqa: E402
from parse_slot_log import RE_LCP, RE_PROMPT   # noqa: E402

BAR_WIDTH = 46
FULL_SCALE_MS = 3000.0     # bar saturates here; keeps both runs on one scale


class C:
    """ANSI colours. Disabled when output is redirected."""
    on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    RED = "\033[1;31m" if on else ""
    GRN = "\033[1;32m" if on else ""
    YEL = "\033[1;33m" if on else ""
    CYA = "\033[1;36m" if on else ""
    DIM = "\033[2m" if on else ""
    B = "\033[1m" if on else ""
    X = "\033[0m" if on else ""


def bar(ms: float, colour: str) -> str:
    n = max(1, min(BAR_WIDTH, int(BAR_WIDTH * ms / FULL_SCALE_MS)))
    return f"{colour}{'#' * n}{C.X}{'.' * (BAR_WIDTH - n)}"


def read_log(path: str) -> tuple[float | None, int | None]:
    """Median chosen-slot similarity and median tokens recomputed, from the
    server's own log. This is the causal evidence; TTFT is the symptom."""
    sims, toks = [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = RE_LCP.search(line)
                if m:
                    sims.append(float(m.group("sim")))
                    continue
                m = RE_PROMPT.search(line)
                if m:
                    toks.append(int(m.group("tokens")))
    except OSError:
        return None, None
    return (statistics.median(sims) if sims else None,
            int(statistics.median(toks)) if toks else None)


def one_config(server: str, model: str, port: int, ctx: int, threads: int,
               tenants: int, turns: int, sim: float, label: str,
               colour: str) -> dict:
    log_path = os.path.join(RESULTS, f"demo_sim{sim}.log")
    url = f"http://127.0.0.1:{port}"
    extra = ["-np", "4", "--slot-prompt-similarity", str(sim)]

    print(f"\n{C.B}{label}{C.X}")
    print(f"{C.DIM}  llama-server {' '.join(extra)}{C.X}\n")

    proc = run_matrix.spawn(server, model, port, ctx, threads, extra, log_path)
    try:
        if not bench_agent.wait_healthy(url, 240):
            print(f"  {C.RED}server never became healthy{C.X} — see {log_path}")
            return {}

        tok = bench_agent.Tokenizer(url)
        convs = [[{"role": "system", "content": bench_agent.SYSTEM},
                  {"role": "user",
                   "content": bench_agent.TENANT_FLAVOR[i % len(bench_agent.TENANT_FLAVOR)]},
                  {"role": "assistant", "content": '{"final": "Understood."}'}]
                 for i in range(tenants)]

        warm = []
        for rnd in range(turns):
            cold = rnd == 0
            tag = f"{C.DIM}cold{C.X}" if cold else "warm"
            for a in range(tenants):
                convs[a].append({"role": "user",
                                 "content": bench_agent.tenant_turn(a, rnd)})
                ttft, _tot, _out, text = bench_agent.one_turn(
                    url, convs[a], 32, 600.0)
                convs[a].append({"role": "assistant",
                                 "content": text[:300] or "{}"})
                if not cold:
                    warm.append(ttft)
                    print(f"  turn {rnd+1} tenant {a}  {tag}  "
                          f"{bar(ttft, colour)} {colour}{ttft:7.0f} ms{C.X}")
                else:
                    print(f"  turn {rnd+1} tenant {a}  {tag}  "
                          f"{C.DIM}{'.' * BAR_WIDTH}{C.X} {C.DIM}{ttft:7.0f} ms{C.X}")
    finally:
        run_matrix.terminate(proc)
        time.sleep(2.0)

    sim_med, tok_med = read_log(log_path)
    med = statistics.median(warm) if warm else 0.0
    print(f"\n  {C.B}warm median {colour}{med:.0f} ms{C.X}"
          f"   slot chosen at similarity {sim_med if sim_med is not None else '?'}"
          f"   {C.B}{tok_med if tok_med is not None else '?'} tokens recomputed{C.X}")
    return {"warm_median_ms": med, "similarity": sim_med,
            "tokens": tok_med, "n_warm": len(warm)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tenants", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
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

    print(f"\n{C.B}  {args.tenants} users of the same agent. "
          f"Same 550-token system prompt and tool schemas.{C.X}")
    print(f"{C.DIM}  Requests are sequential — nothing overlaps, so this is not "
          f"queueing.{C.X}")

    bad = one_config(args.server, args.model, args.port, args.ctx, args.threads,
                     args.tenants, args.turns, 0.1,
                     "1 · llama.cpp default  --slot-prompt-similarity 0.10", C.RED)
    good = one_config(args.server, args.model, args.port, args.ctx, args.threads,
                      args.tenants, args.turns, 0.9,
                      "2 · one flag changed   --slot-prompt-similarity 0.90", C.GRN)

    if not bad or not good:
        print(f"\n{C.RED}  a run failed — see the logs above{C.X}\n")
        return 1

    print(f"\n{C.B}  " + "=" * 64 + f"{C.X}")
    print(f"  {'':22}{C.RED}default 0.10{C.X}      {C.GRN}fixed 0.90{C.X}")
    print(f"  {'time to first token':22}{bad['warm_median_ms']:8.0f} ms   "
          f"{good['warm_median_ms']:11.0f} ms")
    print(f"  {'slot chosen at':22}{str(bad['similarity']):>11}   "
          f"{str(good['similarity']):>14}")
    print(f"  {'tokens recomputed':22}{str(bad['tokens']):>11}   "
          f"{str(good['tokens']):>14}")
    if good["warm_median_ms"]:
        print(f"\n  {C.B}{bad['warm_median_ms'] / good['warm_median_ms']:.1f}x "
              f"faster. One flag.{C.X}")
    print(f"{C.B}  " + "=" * 64 + f"{C.X}")
    print(f"\n{C.DIM}  n=1 per configuration — this is the demo, not the evidence."
          f"\n  The adjudicated result is 5 repeats per config with a fresh server"
          f"\n  each time: python3 tools/run_matrix.py{C.X}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

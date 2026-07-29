#!/usr/bin/env python3
"""
run_matrix.py — run the config matrix with independent repeats.

WHY A FRESH SERVER PER REPEAT
-----------------------------
A second run against a still-warm server is not an independent sample: its slots
already hold KV from the previous run, so it measures carry-over rather than the
configuration. Every repeat therefore gets a freshly spawned server, and the
process is torn down before the next one starts. It is slower and it is the only
way the confidence intervals downstream mean anything.

WHAT IT VARIES
--------------
The matrix isolates one variable at a time against a common baseline:

    tenants          1 vs 4 vs 8        does concurrency break caching?
    slots (-np)      default vs 4       does giving everyone a slot fix it?
    similarity       0.10 vs 0.9        is the default threshold the cause?

Cross-platform: owns the server process on Windows and Linux alike, so the same
command produces the Arm numbers later that it produces on a laptop today.

Usage:
    python3 tools/run_matrix.py --server .\\llama\\llama-server.exe \\
        --model .\\models\\qwen1.5b.gguf --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, HERE)

import bench_agent  # noqa: E402

# name -> (agents, extra server args). Each row changes ONE thing from baseline.
MATRIX = [
    ("solo_baseline",      1, ["-np", "4", "--slot-prompt-similarity", "0.1"]),
    ("4tenant_default",    4, ["-np", "4", "--slot-prompt-similarity", "0.1"]),
    ("8tenant_default",    8, ["-np", "4", "--slot-prompt-similarity", "0.1"]),
    ("4tenant_sim09",      4, ["-np", "4", "--slot-prompt-similarity", "0.9"]),
    ("8tenant_sim09",      8, ["-np", "4", "--slot-prompt-similarity", "0.9"]),
    ("8tenant_sim09_np8",  8, ["-np", "8", "--slot-prompt-similarity", "0.9"]),
]


def spawn(server: str, model: str, port: int, ctx: int, threads: int,
          extra: list[str], log_path: str) -> subprocess.Popen:
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-t", str(threads), *extra]
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    kwargs = {"stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        # New process group so we can signal the whole tree on Windows.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def terminate(proc: subprocess.Popen) -> None:
    """Stop the server and wait for the port to be released."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="path to llama-server binary")
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    # Defaults to 5 because the adjudicator's pre-registered MIN_REPS is 5.
    # Running 3 would guarantee every verdict comes back UNCERTAIN.
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--only", help="comma-separated config names to run")
    ap.add_argument("--out", default=os.path.join(RESULTS, "matrix.json"))
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    if not os.path.exists(args.server):
        print(f"server binary not found: {args.server}", file=sys.stderr)
        return 2
    if not os.path.exists(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    os.makedirs(RESULTS, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    wanted = set(args.only.split(",")) if args.only else None
    matrix = [m for m in MATRIX if not wanted or m[0] in wanted]

    total = len(matrix) * args.repeats
    print(f"\n  Config matrix: {len(matrix)} configs x {args.repeats} repeats "
          f"= {total} runs, fresh server each time")
    print(f"  ctx={args.ctx} threads={args.threads} turns={args.turns}\n")

    collected: dict[str, list[dict]] = {}
    done = 0
    t_start = time.time()

    for name, agents, extra in matrix:
        collected[name] = []
        for rep in range(args.repeats):
            done += 1
            log_path = os.path.join(RESULTS, f"server_{name}_r{rep}.log")
            print(f"  [{done}/{total}] {name} rep {rep+1} "
                  f"({agents} tenants, {' '.join(extra)})")

            proc = spawn(args.server, args.model, args.port, args.ctx,
                         args.threads, extra, log_path)
            try:
                if not bench_agent.wait_healthy(url, 240):
                    print(f"      server never became healthy — see {log_path}",
                          file=sys.stderr)
                    continue
                result = bench_agent.run(url, agents, args.turns, 48, 600.0,
                                         f"{name}_r{rep}")
                result["config"] = name
                result["repeat"] = rep
                result["server_args"] = extra
                collected[name].append(result)
                print(f"      warm median {result['warm_median_ms']:.1f} ms "
                      f"· preamble {result['preamble_tokens']} tok "
                      f"· final {result['final_prompt_tokens']} tok")
            except Exception as exc:
                print(f"      run failed: {exc}", file=sys.stderr)
            finally:
                terminate(proc)
                time.sleep(2.0)   # let the port clear before the next spawn

    payload = {"schema": "specarm.matrix/1",
               "server": args.server, "model": os.path.basename(args.model),
               "ctx": args.ctx, "threads": args.threads,
               "turns": args.turns, "repeats": args.repeats,
               "elapsed_s": round(time.time() - t_start, 1),
               "configs": collected}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  wrote {args.out}  ({payload['elapsed_s']/60:.1f} min)")
    print(f"  next: python3 tools/analyze_agent.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

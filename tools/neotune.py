#!/usr/bin/env python3
"""
neotune — one entrypoint for the whole optimization workflow.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a dispatcher. Every subcommand delegates to the tool that already does
the work — it adds no measurement of its own and invents no numbers. The value
it adds is that a developer no longer has to know that the profiler is called
probe_prefill, the search is a flag on tune_similarity, and adjudication lives
in analyze_agent.

The workflow it exposes is the one the project actually performs:

    profile    what am I running on, and is there an opportunity here?
    bench      measure the configuration matrix, fresh server per repeat
    optimize   search the threshold space for THIS workload
    verify     adjudicate, and check for regression against a saved baseline
    report     regenerate the tables, mechanism evidence and chart
    demo       the five-minute version

Run `neotune <command> --help` for the options each one accepts.

USAGE
-----
    python3 tools/neotune.py profile  --server <llama-server> --model <gguf>
    python3 tools/neotune.py bench    --server <llama-server> --model <gguf>
    python3 tools/neotune.py optimize --server <llama-server> --model <gguf> --workload rag
    python3 tools/neotune.py verify   --baseline results/agent_analysis.latest.json
    python3 tools/neotune.py report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
PY = sys.executable or "python3"

# Regression gates for `verify`. Deliberately looser than skeptic.py's
# thresholds: this answers "did something get worse since the baseline", not
# "is this improvement real".
REGRESSION_PCT = 10.0


class C:
    on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    R = "\033[1;31m" if on else ""
    G = "\033[1;32m" if on else ""
    Y = "\033[1;33m" if on else ""
    B = "\033[1m" if on else ""
    D = "\033[2m" if on else ""
    X = "\033[0m" if on else ""


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _run(script: str, *args: str) -> int:
    """Delegate to a tool, streaming its output. Returns its exit code."""
    cmd = [PY, os.path.join(HERE, script), *[a for a in args if a is not None]]
    print(f"{C.D}  $ {' '.join(os.path.basename(c) for c in cmd)}{C.X}\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"{C.R}  cannot run {script}: {exc}{C.X}", file=sys.stderr)
        return 2


def _need(path: str, what: str, fix: str) -> bool:
    """Readable precondition failure instead of a stack trace."""
    if os.path.exists(path):
        return True
    print(f"\n{C.R}  {what} not found{C.X}: {path}")
    print(f"  {C.B}fix:{C.X} {fix}\n")
    return False


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"{C.R}  cannot read {path}: {exc}{C.X}", file=sys.stderr)
        return None


# ----------------------------------------------------------------- commands

def cmd_profile(a) -> int:
    """Identify the host, then size the opportunity before optimizing anything."""
    print(f"\n{C.B}  1/2 · host{C.X}")
    env = os.path.join(ROOT, "scripts", "00_env_report.sh")
    if os.path.exists(env):
        rc = subprocess.call(["bash", env])
        if rc == 3:
            print(f"{C.Y}  (required feature absent — continuing){C.X}")
    else:
        print(f"{C.D}  no env script; skipping{C.X}")

    print(f"\n{C.B}  2/2 · opportunity{C.X}")
    print(f"{C.D}  Is prefix reuse already working? If it is, do not build a cache.{C.X}")
    if not a.url:
        print(f"{C.Y}  --url not given; start a server and re-run to size the opportunity{C.X}")
        return 0
    return _run("probe_prefill.py", "--url", a.url, "--turns", str(a.turns))


def cmd_bench(a) -> int:
    """The full configuration matrix. Fresh server per repeat."""
    if not (_need(a.server, "server binary", "build it with scripts/01_build_llama.sh")
            and _need(a.model, "model", "fetch it with scripts/fetch_model.sh")):
        return 2
    return _run("run_matrix.py", "--server", a.server, "--model", a.model,
                "--repeats", str(a.repeats), "--turns", str(a.turns),
                *(["--only", a.only] if a.only else []))


def cmd_optimize(a) -> int:
    """Search the threshold space for this workload and report the best value."""
    if not (_need(a.server, "server binary", "build it with scripts/01_build_llama.sh")
            and _need(a.model, "model", "fetch it with scripts/fetch_model.sh")):
        return 2
    print(f"\n{C.B}  searching --slot-prompt-similarity for workload "
          f"'{a.workload}'{C.X}")
    print(f"{C.D}  The right value depends on prompt shape, not on this repo's"
          f" defaults.{C.X}")
    return _run("tune_similarity.py", "--sweep",
                "--server", a.server, "--model", a.model,
                "--workload", a.workload,
                "--agents", str(a.agents), "--turns", str(a.turns),
                "--repeats", str(a.repeats),
                *(["--thresholds", a.thresholds] if a.thresholds else []))


def cmd_verify(a) -> int:
    """Adjudicate the current matrix, then compare against a saved baseline."""
    matrix = os.path.join(RESULTS, "matrix.json")
    if not _need(matrix, "matrix.json", "run: neotune bench --server ... --model ..."):
        return 2

    print(f"\n{C.B}  1/2 · adjudicate{C.X}")
    rc = _run("analyze_agent.py")
    if rc != 0:
        return rc

    if not a.baseline:
        print(f"\n{C.D}  no --baseline given; skipping regression check{C.X}")
        return 0

    print(f"\n{C.B}  2/2 · regression vs baseline{C.X}")
    base = _load(a.baseline)
    cur = _load(os.path.join(RESULTS, "agent_analysis.latest.json"))
    if not base or not cur:
        return 2

    bc, cc = base.get("configs", {}), cur.get("configs", {})
    shared = [k for k in cc if k in bc]
    if not shared:
        print(f"{C.Y}  no configurations in common — nothing to compare{C.X}")
        return 0

    print(f"\n  {'config':<22} {'baseline':>11} {'current':>11} {'change':>9}  status")
    print("  " + "-" * 68)
    worst = 0.0
    regressed = []
    for k in sorted(shared):
        b, c = bc[k].get("mean"), cc[k].get("mean")
        if not b or not c:
            continue
        delta = (c - b) / b * 100.0          # TTFT: higher is worse
        worst = max(worst, delta)
        if delta > REGRESSION_PCT:
            tag, col = "REGRESSED", C.R
            regressed.append((k, delta))
        elif delta < -REGRESSION_PCT:
            tag, col = "improved", C.G
        else:
            tag, col = "stable", C.D
        print(f"  {k:<22} {b:>9.1f}ms {c:>9.1f}ms {delta:>+8.1f}%  {col}{tag}{C.X}")

    print()
    if regressed:
        print(f"{C.R}{C.B}  PERFORMANCE REGRESSION DETECTED{C.X}")
        for k, d in regressed:
            print(f"    {k}: {d:+.1f}%  (gate is +{REGRESSION_PCT:.0f}%)")
        print(f"\n  {C.B}Status: REJECTED{C.X}\n")
        return 1
    print(f"{C.G}{C.B}  No regression beyond +{REGRESSION_PCT:.0f}% "
          f"(worst {worst:+.1f}%){C.X}")
    print(f"\n  {C.B}Status: PASS{C.X}\n")
    return 0


def cmd_report(a) -> int:
    """Regenerate every artifact a reader looks at."""
    steps = [("adjudicated tables", "analyze_agent.py", []),
             ("mechanism evidence + cost model", "parse_slot_log.py", []),
             ("chart", "make_chart.py", [])]
    rc_all = 0
    for n, (title, script, args) in enumerate(steps, 1):
        print(f"\n{C.B}  {n}/{len(steps)} · {title}{C.X}")
        rc = _run(script, *args)
        rc_all = rc_all or rc
    print(f"\n  wrote into {RESULTS}\n")
    return rc_all


def cmd_demo(a) -> int:
    if not (_need(a.server, "server binary", "build it with scripts/01_build_llama.sh")
            and _need(a.model, "model", "fetch it with scripts/fetch_model.sh")):
        return 2
    return _run("demo.py", "--server", a.server, "--model", a.model,
                "--tenants", str(a.tenants), "--turns", str(a.turns))


# ----------------------------------------------------------------- argparse

def main() -> int:
    _utf8()
    p = argparse.ArgumentParser(
        prog="neotune", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    sp = sub.add_parser("profile", help="identify the host and size the opportunity")
    sp.add_argument("--url", help="a running llama-server, e.g. http://127.0.0.1:8080")
    sp.add_argument("--turns", type=int, default=6)
    sp.set_defaults(fn=cmd_profile)

    sb = sub.add_parser("bench", help="run the configuration matrix")
    sb.add_argument("--server", required=True)
    sb.add_argument("--model", required=True)
    sb.add_argument("--repeats", type=int, default=5)
    sb.add_argument("--turns", type=int, default=4)
    sb.add_argument("--only", help="comma-separated config names")
    sb.set_defaults(fn=cmd_bench)

    so = sub.add_parser("optimize", help="search the threshold space for a workload")
    so.add_argument("--server", required=True)
    so.add_argument("--model", required=True)
    so.add_argument("--workload", default="agent", help="agent (default) or rag")
    so.add_argument("--agents", type=int, default=4)
    so.add_argument("--turns", type=int, default=4)
    so.add_argument("--repeats", type=int, default=3)
    so.add_argument("--thresholds", help="comma-separated, e.g. 0.1,0.5,0.9")
    so.set_defaults(fn=cmd_optimize)

    sv = sub.add_parser("verify", help="adjudicate, and detect regression vs a baseline")
    sv.add_argument("--baseline", help="a saved agent_analysis.latest.json")
    sv.set_defaults(fn=cmd_verify)

    sr = sub.add_parser("report", help="regenerate tables, evidence and chart")
    sr.set_defaults(fn=cmd_report)

    sd = sub.add_parser("demo", help="the five-minute version")
    sd.add_argument("--server", required=True)
    sd.add_argument("--model", required=True)
    sd.add_argument("--tenants", type=int, default=4)
    sd.add_argument("--turns", type=int, default=4)
    sd.set_defaults(fn=cmd_demo)

    a = p.parse_args()
    if not a.cmd:
        p.print_help()
        print("\n  start with:  neotune profile --url http://127.0.0.1:8080\n")
        return 0
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("\n  interrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())

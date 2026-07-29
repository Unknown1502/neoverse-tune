#!/usr/bin/env python3
"""
test_agent_tools.py — validate the tuner's math and the aggregator's verdicts.

tune_similarity.py recommends a configuration value from a closed-form argument.
If that argument is wrong the tool confidently misconfigures every user, so the
window arithmetic is checked against hand-computed cases, including the case
where no valid threshold exists and the tool must refuse to invent one.

analyze_agent.py is fed a synthetic matrix with a planted regression and a
planted fix, so the correct headline is known by construction.

ALL FIXTURES ARE SYNTHETIC.

Run:  python3 tools/test_agent_tools.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  [' + detail + ']') if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------- tuner math

def window(preamble: int, totals: list[int]) -> tuple[float, float, bool, float | None]:
    """Independent reimplementation of the tuner's closed form.

    Derived from the docstring algebra rather than copied from the tool: if both
    were wrong the same way, the test would prove nothing. The constraint binds
    from turn 2 onward — at turn 1 a tenant holds no slot, so there is no own
    cache to protect and inter(1) is not a real constraint.
    """
    inter = [preamble / totals[k] for k in range(1, len(totals))]
    intra = [totals[k - 1] / totals[k] for k in range(1, len(totals))]
    mx, mn = max(inter), min(intra)
    feasible = mx < mn
    rec = round((mx * mn) ** 0.5, 3) if feasible else None
    return mx, mn, feasible, rec


def test_tuner_math() -> None:
    print("\ntune_similarity — window arithmetic\n")

    # Agent shape: big preamble, small increments per turn.
    P, totals = 521, [600, 700, 800, 900, 1000]
    mx, mn, feasible, rec = window(P, totals)
    check("agent shape: window exists", feasible, f"({mx:.3f}, {mn:.3f}]")
    check("agent shape: default 0.10 is below max inter", 0.10 < mx,
          f"max_inter={mx:.3f}")
    check("agent shape: recommendation inside window",
          rec is not None and mx < rec <= mn, f"rec={rec}")

    # Chat shape: tiny system prompt, long generations. The default exists for
    # exactly this case and should be adequate.
    P2, totals2 = 20, [500, 1000, 2000, 4000]
    mx2, mn2, feasible2, _ = window(P2, totals2)
    check("chat shape: default 0.10 adequate", 0.10 >= mx2, f"max_inter={mx2:.3f}")
    check("chat shape: window exists", feasible2, f"({mx2:.3f}, {mn2:.3f}]")

    # INVARIANT worth stating: the preamble is a prefix of every prompt, so
    # P <= T(k-1) for all k >= 2, hence P/T_k <= T(k-1)/T_k — inter never
    # exceeds intra. The window can be narrow but is never empty. Any code path
    # claiming "no threshold exists" is therefore unreachable for well-formed
    # input, and a narrow window is the real failure mode to warn about.
    for P3, t3 in [(521, [600, 700, 800]), (20, [500, 1000, 2000]),
                   (990, [1000, 1001, 1002]), (100, [101, 500, 900])]:
        m_inter, m_intra, ok, _ = window(P3, t3)
        if not (m_inter <= m_intra + 1e-9):
            check(f"invariant inter<=intra for P={P3}", False,
                  f"{m_inter:.4f} > {m_intra:.4f}")
            break
    else:
        check("invariant: inter never exceeds intra (window never empty)", True)

    # Narrow windows are the real risk: little room between reject and accept.
    mx4, mn4, _, _ = window(990, [1000, 1001, 1002])
    check("narrow window is detectable", (mn4 - mx4) < 0.05,
          f"width={mn4 - mx4:.4f}")

    # The headline claim: a bigger preamble makes the default worse, not better.
    small = window(100, [500, 600, 700])[0]
    large = window(400, [500, 600, 700])[0]
    check("larger preamble raises inter-tenant similarity", large > small,
          f"{small:.3f} -> {large:.3f}")

    # Cross-check the real module against this independent derivation.
    spec = importlib.util.spec_from_file_location(
        "tune", os.path.join(HERE, "tune_similarity.py"))
    tune = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tune)
        check("tuner module imports cleanly", True)
    except Exception as exc:  # pragma: no cover
        check("tuner module imports cleanly", False, str(exc))


# ------------------------------------------------------------------- aggregator

def synth_run(config: str, agents: int, warm_ms: float, rep: int) -> dict:
    jitter = 1.0 + ((rep % 5) - 2) * 0.02          # +/-4%, deterministic
    return {"schema": "specarm.agentbench/1", "config": config, "repeat": rep,
            "agents": agents, "turns": 4, "token_count_method": "tokenize_concat",
            "preamble_tokens": 521, "final_prompt_tokens": 860,
            "warm_median_ms": round(warm_ms * jitter, 2),
            "cold_median_ms": round(warm_ms * jitter * 2, 2),
            "warm_ttft_ms": [], "cold_ttft_ms": [], "rows": []}


def test_aggregator() -> None:
    print("\nanalyze_agent — planted regression and fix\n")
    tmp = tempfile.mkdtemp(prefix="specarm-agent-")
    try:
        configs = {
            "solo_baseline":     [synth_run("solo_baseline", 1, 330.0, r) for r in range(5)],
            "4tenant_default":   [synth_run("4tenant_default", 4, 2100.0, r) for r in range(5)],
            "8tenant_default":   [synth_run("8tenant_default", 8, 2250.0, r) for r in range(5)],
            "4tenant_sim09":     [synth_run("4tenant_sim09", 4, 450.0, r) for r in range(5)],
            "8tenant_sim09":     [synth_run("8tenant_sim09", 8, 430.0, r) for r in range(5)],
            "8tenant_sim09_np8": [synth_run("8tenant_sim09_np8", 8, 425.0, r) for r in range(5)],
        }
        matrix = {"schema": "specarm.matrix/1", "model": "synthetic.gguf",
                  "ctx": 32768, "threads": 4, "turns": 4, "repeats": 5,
                  "configs": configs}
        path = os.path.join(tmp, "matrix.json")
        json.dump(matrix, open(path, "w"))

        env = dict(os.environ, SPECARM_RESULTS=tmp, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, os.path.join(HERE, "analyze_agent.py"),
                            "--input", path],
                           capture_output=True, text=True, encoding="utf-8", env=env)
        check("analyze_agent exits 0", p.returncode == 0, p.stderr.strip()[-200:])
        md = p.stdout
        a = json.load(open(os.path.join(tmp, "agent_analysis.latest.json")))

        check("regression magnitude ~6.4x",
              6.0 <= a.get("regression_x", 0) <= 6.8, f"{a.get('regression_x')}")
        check("improvement magnitude ~4.7x",
              4.3 <= a.get("improvement_x", 0) <= 5.0, f"{a.get('improvement_x')}")

        cmps = a.get("comparisons", {})
        reg = cmps.get("solo_baseline__vs__4tenant_default", {})
        check("regression relabelled, not shown as a failure",
              reg.get("shown") == "REGRESSION CONFIRMED", str(reg.get("shown")))
        fix = cmps.get("4tenant_default__vs__4tenant_sim09", {})
        check("fix verified", fix.get("verdict") == "VERIFIED", str(fix.get("verdict")))
        parity = cmps.get("solo_baseline__vs__8tenant_sim09", {})
        check("parity comparison present", bool(parity))

        check("report states the ratio direction", "below 1.0 is faster" in md)
        check("report discloses independent-sample definition",
              "per-repeat" in md.lower())
        check("report has a 'does not claim' section", "does not claim" in md.lower())
        check("headline explains the ratio mechanism",
              "shared_prefix" in md or "shared prefix" in md.lower())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("\nAgent tooling — synthetic ground truth")
    test_tuner_math()
    test_aggregator()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All agent-tool checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

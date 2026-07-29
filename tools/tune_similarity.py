#!/usr/bin/env python3
"""
tune_similarity.py — find the right --slot-prompt-similarity for YOUR agent.

TWO MODES, AND THE DEFAULT ONE IS THE HONEST ONE
------------------------------------------------
  --sweep      Measures. Spawns a server at each candidate threshold, runs the
               multi-tenant workload, reports warm TTFT. Authoritative, slow.
  (default)    Estimates from a closed-form model. Fast, and EXPLICITLY a
               heuristic, because the model rests on an assumption about
               llama.cpp internals that has not been verified against source.

WHY THE DISTINCTION IS LABELLED SO LOUDLY
-----------------------------------------
The analytic model assumes llama-server's slot similarity is
`common_prefix_tokens / prompt_tokens`. That is a guess from the documented
flag description, not something read out of the implementation. On a real
measured workload the model predicted the usable window closes at 0.857 while
a threshold of 0.9 empirically worked. When a model contradicts a measurement,
the measurement wins. So the analytic mode is a starting point for the sweep,
never a substitute for it.

THE MODEL, FOR WHAT IT IS WORTH
-------------------------------
Let  P   = tokens in the shared preamble (system prompt + tool schemas)
     T_k = total prompt tokens at turn k

  Landing on ANOTHER tenant's slot shares only the preamble:  inter(k) = P / T_k
  Landing on YOUR OWN slot from last turn:                    intra(k) = T(k-1)/T_k

A threshold must reject foreign slots and accept your own, for every turn from
the second onward — turn 1 is excluded because a tenant has no slot of its own
yet, so nothing needs protecting:

      max(inter(k), k>=2)  <  t  <=  min(intra(k), k>=2)

The instructive part survives regardless of the exact formula: inter(k) rises
with P. **The larger your system prompt and tool schema, the higher the
similarity between any two tenants, and the more inadequate a low fixed default
becomes.** That is the counterintuitive bit, and it is what the measurement
actually demonstrated.

USAGE
-----
    # fast estimate against a running server
    python3 tools/tune_similarity.py --url http://127.0.0.1:8081

    # authoritative: measure it
    python3 tools/tune_similarity.py --sweep \\
        --server ./llama/llama-server.exe --model ./models/qwen1.5b.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, HERE)

from bench_agent import Tokenizer, TENANT_FLAVOR  # noqa: E402
from probe_prefill import SYSTEM, TURNS  # noqa: E402

DEFAULT_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


# ----------------------------------------------------------------- analytic mode

def build_curve(tok: Tokenizer, system: str, turns: int) -> tuple[int, list[int]]:
    preamble = tok.count([{"role": "system", "content": system}])
    convo = [{"role": "system", "content": system},
             {"role": "user", "content": TENANT_FLAVOR[0]},
             {"role": "assistant", "content": '{"final": "Understood."}'}]
    totals = []
    for k in range(turns):
        convo.append({"role": "user", "content": TURNS[k % len(TURNS)]})
        totals.append(tok.count(convo))
        convo.append({"role": "assistant",
                      "content": '{"tool": "search_flights", "arguments": '
                                 '{"origin": "DEL", "destination": "SIN", '
                                 '"date": "2026-09-14", "passengers": 2}}'})
    return preamble, totals


def analytic(preamble: int, totals: list[int]) -> dict:
    """Closed-form window. Constraint binds from turn 2 onward: at turn 1 a
    tenant holds no slot, so there is no own-cache to protect."""
    if len(totals) < 3:
        return {"feasible": False, "reason": "need at least 3 turns"}

    inter = [preamble / totals[k] for k in range(1, len(totals))]
    intra = [totals[k - 1] / totals[k] for k in range(1, len(totals))]
    mx, mn = max(inter), min(intra)
    feasible = mx < mn
    return {
        "preamble_tokens": preamble,
        "prompt_tokens_by_turn": totals,
        "max_inter_tenant_similarity": round(mx, 4),
        "min_intra_tenant_similarity": round(mn, 4),
        "feasible": feasible,
        "estimated_threshold": round((mx * mn) ** 0.5, 3) if feasible else None,
        "llama_cpp_default": 0.10,
        "default_below_inter_tenant": 0.10 <= mx,
    }


# ----------------------------------------------------------------- empirical mode

def sweep(server: str, model: str, port: int, ctx: int, threads: int,
          agents: int, turns: int, repeats: int,
          thresholds: list[float]) -> list[dict]:
    import bench_agent
    import run_matrix

    url = f"http://127.0.0.1:{port}"
    out = []
    for t in thresholds:
        warm = []
        for rep in range(repeats):
            log = os.path.join(RESULTS, f"sweep_sim{t}_r{rep}.log")
            proc = run_matrix.spawn(server, model, port, ctx, threads,
                                    ["-np", "4", "--slot-prompt-similarity", str(t)],
                                    log)
            try:
                if not bench_agent.wait_healthy(url, 240):
                    print(f"    sim={t} rep={rep}: server unhealthy, see {log}",
                          file=sys.stderr)
                    continue
                r = bench_agent.run(url, agents, turns, 48, 600.0, f"sim{t}_r{rep}")
                warm.append(r["warm_median_ms"])
            except Exception as exc:
                print(f"    sim={t} rep={rep}: {exc}", file=sys.stderr)
            finally:
                run_matrix.terminate(proc)
                time.sleep(2.0)
        if warm:
            warm_sorted = sorted(warm)
            median = warm_sorted[len(warm_sorted) // 2]
            out.append({"threshold": t, "warm_median_ms": round(median, 2),
                        "samples": [round(x, 2) for x in warm]})
            print(f"    sim={t:<5} warm median {median:8.1f} ms   "
                  f"(n={len(warm)})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--system-file")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--sweep", action="store_true",
                    help="measure instead of estimating (authoritative)")
    ap.add_argument("--server")
    ap.add_argument("--model")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--thresholds",
                    help="comma-separated sweep values (default: "
                         + ",".join(str(t) for t in DEFAULT_SWEEP) + ")")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    os.makedirs(RESULTS, exist_ok=True)
    report: dict = {"schema": "specarm.tune/2"}

    # ---------------------------------------------------------------- sweep
    if args.sweep:
        if not args.server or not args.model:
            print("--sweep requires --server and --model", file=sys.stderr)
            return 2
        thresholds = ([float(x) for x in args.thresholds.split(",")]
                      if args.thresholds else DEFAULT_SWEEP)
        print(f"\n  Measuring {len(thresholds)} thresholds x {args.repeats} repeats, "
              f"fresh server each run\n")
        rows = sweep(args.server, args.model, args.port, args.ctx, args.threads,
                     args.agents, args.turns, args.repeats, thresholds)
        report["mode"] = "measured"
        report["sweep"] = rows
        if rows:
            best = min(rows, key=lambda r: r["warm_median_ms"])
            worst = max(rows, key=lambda r: r["warm_median_ms"])
            report["recommended_slot_prompt_similarity"] = best["threshold"]
            report["best_warm_median_ms"] = best["warm_median_ms"]
            report["worst_warm_median_ms"] = worst["warm_median_ms"]
            report["speedup_vs_worst"] = round(
                worst["warm_median_ms"] / best["warm_median_ms"], 2)
            print(f"\n  MEASURED BEST: --slot-prompt-similarity {best['threshold']}")
            print(f"  {worst['warm_median_ms']:.0f} ms (worst) -> "
                  f"{best['warm_median_ms']:.0f} ms (best) = "
                  f"{report['speedup_vs_worst']}x\n")
        else:
            print("\n  sweep produced no usable runs\n", file=sys.stderr)

    # ---------------------------------------------------------------- analytic
    else:
        system = SYSTEM
        if args.system_file:
            with open(args.system_file, encoding="utf-8") as fh:
                system = fh.read()
        tok = Tokenizer(args.url)
        preamble, totals = build_curve(tok, system, args.turns)
        est = analytic(preamble, totals)
        est["token_count_method"] = tok.method
        report["mode"] = "estimated"
        report.update(est)

        if not args.json:
            print()
            print("  ┌─ slot-prompt-similarity ESTIMATE " + "─" * 34)
            print("  │ HEURISTIC. The similarity formula is inferred from the")
            print("  │ flag documentation, not read from llama.cpp source, and")
            print("  │ has already disagreed with measurement once. Use --sweep")
            print("  │ before relying on any of this.")
            print("  │")
            print(f"  │ shared preamble ............ {preamble} tokens")
            print(f"  │ prompt turn 1 / turn {len(totals)} ..... "
                  f"{totals[0]} / {totals[-1]} tokens")
            print("  │")
            if est.get("feasible"):
                print(f"  │ estimated window ... "
                      f"({est['max_inter_tenant_similarity']:.3f}, "
                      f"{est['min_intra_tenant_similarity']:.3f}]")
                print(f"  │ starting point ..... {est['estimated_threshold']}")
            else:
                print("  │ model says no window exists — but the model is a")
                print("  │ heuristic. Measure with --sweep before concluding.")
            print("  │")
            if est.get("default_below_inter_tenant"):
                print(f"  │ ⚠  default 0.10 sits below the inter-tenant similarity")
                print(f"  │    of {est['max_inter_tenant_similarity']:.3f}: every slot "
                      f"qualifies for")
                print(f"  │    every tenant, so tenants scatter across slots.")
            else:
                print("  │ ✓  0.10 appears adequate for this prompt shape")
            print("  └" + "─" * 68)
            print()
            print("  Authoritative next step:")
            print("    python3 tools/tune_similarity.py --sweep \\")
            print("        --server <llama-server> --model <model.gguf>")
            print()

    if args.json:
        print(json.dumps(report, indent=2))

    out_path = args.out or os.path.join(RESULTS, "tune.latest.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

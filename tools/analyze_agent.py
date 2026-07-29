#!/usr/bin/env python3
"""
analyze_agent.py — aggregate the config matrix into an adjudicated result.

Independent samples are the PER-REPEAT warm medians, not individual requests.
Requests within one run share a server, a cache state and a schedule, so they
are not independent; treating them as such would shrink the confidence
intervals to nothing and manufacture significance. n therefore equals the
number of repeats, which is why run_matrix.py defaults to 5 — the adjudicator's
pre-registered MIN_REPS.

Latency is inverted to a rate before adjudication, because verdict() is written
for higher-is-better and TTFT is lower-is-better.

Run:  python3 tools/analyze_agent.py [--input results/matrix.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("SPECARM_RESULTS", os.path.join(ROOT, "results"))

_spec = importlib.util.spec_from_file_location("analyze", os.path.join(HERE, "analyze.py"))
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)
Stat, verdict = analyze.Stat, analyze.verdict

# Each comparison states in advance what it is testing and which direction
# counts as the interesting outcome.
COMPARISONS = [
    ("solo_baseline", "4tenant_default",
     "Does adding tenants break prefix caching?", "regression_expected"),
    ("solo_baseline", "8tenant_default",
     "Does it get worse with more tenants?", "regression_expected"),
    ("4tenant_default", "4tenant_sim09",
     "Does raising the similarity threshold fix it?", "improvement_expected"),
    ("8tenant_default", "8tenant_sim09",
     "Does the fix hold when tenants exceed slots?", "improvement_expected"),
    ("solo_baseline", "8tenant_sim09",
     "With the fix, is multi-tenant back at single-tenant parity?", "parity_expected"),
    ("8tenant_sim09", "8tenant_sim09_np8",
     "Do extra slots add anything once the threshold is right?", "improvement_expected"),
]


def samples_for(cfg: dict) -> list[float]:
    """Per-repeat warm medians = the independent samples."""
    return [r["warm_median_ms"] for r in cfg if r.get("warm_median_ms")]


def as_rate(ms: list[float]) -> Stat:
    """TTFT -> requests/sec-equivalent, so higher is better for verdict()."""
    return Stat([1000.0 / m for m in ms if m > 0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(RESULTS, "matrix.json"))
    ap.add_argument("--out")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    if not os.path.exists(args.input):
        print(f"No matrix at {args.input}. Run tools/run_matrix.py first.",
              file=sys.stderr)
        return 1
    with open(args.input) as fh:
        data = json.load(fh)

    configs = data.get("configs", {})
    env: dict[str, Any] = {}
    env_path = os.path.join(RESULTS, "env.latest.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))

    lines: list[str] = []
    core = env.get("core", "x86 laptop (no env report)")
    lines.append("# Multi-tenant agent serving — adjudicated result")
    lines.append("")
    lines.append(f"- **Host**: `{core}`, {data.get('threads')} threads")
    lines.append(f"- **Model**: `{data.get('model')}` · ctx {data.get('ctx')}")
    lines.append(f"- **Repeats**: {data.get('repeats')} per config, "
                 f"**fresh server each time**")
    lines.append(f"- **Workload**: N tenants sharing one agent, "
                 f"{data.get('turns')} turns each, round-robin")
    lines.append("")

    # ---------------------------------------------------------------- per-config
    lines.append("## Configurations")
    lines.append("")
    lines.append("| config | tenants | warm TTFT (ms, mean ± 95% CI) | RSD | n |")
    lines.append("|:--|--:|--:|--:|--:|")

    stats: dict[str, Stat] = {}
    meta: dict[str, dict] = {}
    for name, runs in configs.items():
        ms = samples_for(runs)
        if not ms:
            lines.append(f"| `{name}` | – | no data | – | 0 |")
            continue
        s = Stat(ms)
        stats[name] = s
        meta[name] = {"agents": runs[0].get("agents"),
                      "preamble_tokens": runs[0].get("preamble_tokens"),
                      "final_prompt_tokens": runs[0].get("final_prompt_tokens"),
                      "token_method": runs[0].get("token_count_method"),
                      "samples_ms": [round(x, 2) for x in ms]}
        lines.append(f"| `{name}` | {runs[0].get('agents')} | "
                     f"{s.mean:.1f} ± {s.hw:.1f} | {s.rsd:.1f}% | {s.n} |")
    lines.append("")

    any_meta = next(iter(meta.values()), {})
    if any_meta:
        lines.append(f"Preamble: **{any_meta.get('preamble_tokens')} tokens** shared "
                     f"byte-identically by every tenant. Final prompt: "
                     f"~{any_meta.get('final_prompt_tokens')} tokens. "
                     f"Counts via `{any_meta.get('token_method')}`.")
        lines.append("")

    # ---------------------------------------------------------------- comparisons
    lines.append("## Adjudicated comparisons")
    lines.append("")
    lines.append("Lower TTFT is better; the ratio column is "
                 "`candidate / reference`, so **below 1.0 is faster**.")
    lines.append("")
    lines.append("| question | reference | candidate | ratio | verdict | note |")
    lines.append("|:--|:--|:--|--:|:--|:--|")

    out: dict[str, Any] = {"schema": "specarm.agent_analysis/1",
                           "configs": {k: v.as_dict() for k, v in stats.items()},
                           "meta": meta, "comparisons": {}}

    for ref, cand, question, expectation in COMPARISONS:
        if ref not in stats or cand not in stats:
            continue
        r_ms, c_ms = stats[ref], stats[cand]
        v, why = verdict(as_rate(meta[ref]["samples_ms"]),
                         as_rate(meta[cand]["samples_ms"]))
        ratio = c_ms.mean / r_ms.mean if r_ms.mean else 0.0

        # verdict() answers "is the candidate faster?". For a comparison whose
        # point is to expose a REGRESSION, a REJECTED is the finding, so relabel
        # rather than printing a red cross next to the headline result.
        if expectation == "regression_expected":
            if v == "REJECTED":
                shown, icon = "REGRESSION CONFIRMED", "🔴"
                why = f"{ratio:.2f}x slower than reference"
            elif v == "VERIFIED":
                shown, icon = "NO REGRESSION", "✅"
            else:
                shown, icon = v, "⚠️"
        else:
            shown = v
            icon = {"VERIFIED": "✅", "UNCERTAIN": "⚠️", "REJECTED": "❌"}[v]

        lines.append(f"| {question} | `{ref}` | `{cand}` | {ratio:.2f}x "
                     f"| {icon} {shown} | {why} |")
        out["comparisons"][f"{ref}__vs__{cand}"] = {
            "question": question, "expectation": expectation,
            "ratio": round(ratio, 4), "verdict": v, "shown": shown, "reason": why}
    lines.append("")

    # ---------------------------------------------------------------- headline
    lines.append("## Headline")
    lines.append("")
    base = stats.get("solo_baseline")
    bad = stats.get("4tenant_default")
    fixed = stats.get("4tenant_sim09")
    if base and bad and fixed:
        regress = bad.mean / base.mean if base.mean else 0
        improve = bad.mean / fixed.mean if fixed.mean else 0
        out["regression_x"] = round(regress, 3)
        out["improvement_x"] = round(improve, 3)
        lines.append(f"With llama.cpp's **default** `--slot-prompt-similarity`, "
                     f"moving from one tenant to four costs **{regress:.1f}x** "
                     f"time-to-first-token ({base.mean:.0f} ms → {bad.mean:.0f} ms) "
                     f"even though every tenant shares an identical "
                     f"{any_meta.get('preamble_tokens')}-token preamble.")
        lines.append("")
        lines.append(f"Raising the threshold recovers **{improve:.1f}x** "
                     f"({bad.mean:.0f} ms → {fixed.mean:.0f} ms).")
        lines.append("")
        lines.append("The threshold is a ratio — `shared_prefix / total_prompt`. "
                     "Agent workloads make that ratio structurally large between "
                     "*any two tenants*, so a low fixed default lets every slot "
                     "look like a valid match for every tenant. **The bigger your "
                     "system prompt and tool schema, the worse the default "
                     "behaves** — the opposite of what anyone would assume, and "
                     "invisible to single-conversation benchmarks.")
    else:
        lines.append("Not enough configurations completed to state a headline.")
    lines.append("")

    lines.append("## What this does not claim")
    lines.append("")
    lines.append("- Not a claim about llama.cpp being poorly engineered. The "
                 "default is reasonable for chat; it is wrong for agents.")
    lines.append("- Not a throughput claim. TTFT only.")
    lines.append("- Independent samples are per-repeat medians, so `n` equals the "
                 "repeat count, not the request count.")
    lines.append("")

    md = "\n".join(lines)
    out_md = args.out or os.path.join(RESULTS, "agent_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(RESULTS, "agent_analysis.latest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(md)
    print(f"\n[analyze_agent] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

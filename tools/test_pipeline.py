#!/usr/bin/env python3
"""
test_pipeline.py — end-to-end test of the reporting pipeline.

Feeds analyze.py synthetic files shaped like real `llama-bench -o json` output
and asserts the report comes out right. This exists because the exact key names
llama-bench emits are the least-verified assumption in this repo: if that shape
drifts, the parsing must fail loudly here rather than silently produce an empty
table on a runner.

EVERYTHING IN THIS FILE IS SYNTHETIC. The fixtures are constructed with a knee
deliberately planted at N=4 so we can check the detector finds it. None of these
numbers measure any hardware, and nothing here is written to results/.

Run:  python3 tools/test_pipeline.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "analyze.py")

BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
KNEE_AT = 4          # planted: kleidi jumps here and nowhere else
STAMP = "19700101T000000Z"


def bench_rows(rng, batches, base_ts, boost_from):
    """Rows using llama-bench's real key names."""
    rows = []
    for n in batches:
        ts = base_ts * (1 + 0.35 * (n ** 0.5))
        if boost_from and n >= boost_from:
            ts *= 1.9
        rows.append({
            "build_commit": "0000000", "model_filename": "synthetic.gguf",
            "n_threads": 4, "n_prompt": n, "n_gen": 0,
            "avg_ts": ts, "stddev_ts": ts * 0.01,
            "samples_ts": [rng.gauss(ts, ts * 0.01) for _ in range(10)],
        })
    return rows


def gen_rows(rng, ts):
    return [{"build_commit": "0000000", "model_filename": "synthetic.gguf",
             "n_threads": 4, "n_prompt": 0, "n_gen": 64,
             "avg_ts": ts, "stddev_ts": ts * 0.01,
             "samples_ts": [rng.gauss(ts, ts * 0.01) for _ in range(10)]}]


def build_fixtures(d: str) -> None:
    rng = random.Random(7)
    w = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w"))

    w(f"sweep_kleidi_pp_{STAMP}.json", bench_rows(rng, BATCHES, 20.0, KNEE_AT))
    w(f"sweep_base_pp_{STAMP}.json",   bench_rows(rng, BATCHES, 20.0, None))
    w(f"sweep_kleidi_tg_{STAMP}.json", gen_rows(rng, 21.0))
    w(f"sweep_base_tg_{STAMP}.json",   gen_rows(rng, 20.8))
    w("env.latest.json", {
        "schema": "specarm.env/1", "core": "Synthetic-Core", "nproc": 4,
        "crux_role": "subject", "virtualization": "none", "sve_vector_bits": 128,
        "llama_cpp_sha": "0" * 40,
        "features": {"i8mm": True, "sve2": True, "bf16": True, "asimddp": True}})
    w("kernels.latest.json", {
        "schema": "specarm.kernels/1", "perf_available": True,
        "evidence_class": "execution", "kernel_switch_observed": True,
        "batches": {
            "1":  {"measured": True, "kai_cycles_pct": 61.2, "hottest_kai_isa": "dotprod",
                   "hottest_kai_symbol": "kai_synthetic_neon_dotprod"},
            "16": {"measured": True, "kai_cycles_pct": 73.5, "hottest_kai_isa": "i8mm",
                   "hottest_kai_symbol": "kai_synthetic_neon_i8mm"}},
        "linked_kernels_by_isa": {"dotprod": ["a"], "i8mm": ["b"]},
        "linked_kernel_count": 2})
    open(os.path.join(d, "run.latest"), "w").write(STAMP)


def main() -> int:
    print("\nPipeline end-to-end — SYNTHETIC fixtures, knee planted at N=%d\n" % KNEE_AT)
    tmp = tempfile.mkdtemp(prefix="specarm-pipeline-")
    failures: list[str] = []
    try:
        build_fixtures(tmp)
        env = dict(os.environ, SPECARM_RESULTS=tmp, PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, ANALYZE], env=env,
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            print("FAIL  analyze.py exited nonzero")
            return 1

        md = proc.stdout
        analysis_path = os.path.join(tmp, "analysis.latest.json")
        if not os.path.exists(analysis_path):
            print("FAIL  analysis.latest.json was not written")
            return 1
        analysis = json.load(open(analysis_path))

        def check(name: str, ok: bool, detail: str = "") -> None:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail and not ok else ''}")
            if not ok:
                failures.append(name)

        # Parsing actually found every batch — an empty table is the failure
        # mode this test exists to catch.
        check("all batch sizes parsed",
              len(analysis["batches"]) == len(BATCHES),
              f"got {len(analysis['batches'])} of {len(BATCHES)}")

        # The knee detector must land on the planted discontinuity.
        knee = analysis.get("knee") or {}
        check(f"knee detected at N={KNEE_AT}",
              knee.get("to_batch") == KNEE_AT, f"got {knee.get('to_batch')}")

        # Big clean effects must be VERIFIED; the flat low-N region must not be.
        check("large effects verified",
              analysis["verdicts"][str(KNEE_AT)]["verdict"] == "VERIFIED",
              analysis["verdicts"][str(KNEE_AT)]["verdict"])
        check("flat region not verified",
              analysis["verdicts"]["1"]["verdict"] != "VERIFIED",
              analysis["verdicts"]["1"]["verdict"])

        # The decode row is the whole point of the project; it must appear.
        check("decode row present", "decode" in analysis)

        # Report sections a reader depends on.
        for section in ("## Batch sweep", "## Knee", "## Mechanism", "## Skeptic summary"):
            check(f"report contains '{section}'", section in md)
        check("mechanism reports the kernel switch",
              "KERNEL SWITCH OBSERVED" in md)
        check("thresholds are disclosed in the report",
              "pre-registered" in md.lower())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        return 1
    print("Pipeline checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

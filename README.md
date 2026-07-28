# SpecArm

**Wake the kernels that sleep during decode.**

On Arm CPUs, LLM decode runs one token at a time. A batch-1 matmul is a **GEMV**,
and GEMV cannot use KleidiAI's **i8mm** microkernels — those are built on `SMMLA`,
a matrix-multiply instruction that needs multiple rows to pay off. The fastest
integer matmul path on the chip is therefore largely idle during the phase where
almost all inference time goes.

This matters most when you are **serving**. A server batches concurrent requests
together, so client concurrency drives the real matmul batch size — and
speculative decoding verifies *N* draft tokens in a single forward pass. SpecArm
measures whether either is enough to reach the regime where i8mm kernels get
selected, and proves which kernel actually ran rather than inferring it from a
faster stopwatch.

The practical question it answers: **which Arm cloud instance should you run LLM
serving on, and why** — with kernel-level evidence instead of vibes.

> **Status: measurement harness complete, results not yet collected.**
> No performance numbers appear in this repo yet. When they do, they will arrive
> with confidence intervals, the machine they came from, and a verdict — including
> the ones that get rejected. See [`docs/methodology.md`](docs/methodology.md),
> which was committed **before** any measurement was taken.

---

## Quickstart

Needs an **arm64 Linux** host. Everything is free to reproduce:

```bash
git clone <your-fork-url> && cd specarm

bash scripts/00_env_report.sh      # which Arm core is this? has it got i8mm?
bash scripts/fetch_model.sh        # Qwen2.5-0.5B-Instruct Q4_0 (Apache-2.0)
bash scripts/01_build_llama.sh     # llama.cpp built twice: KleidiAI ON and OFF
bash scripts/02_batch_sweep.sh     # the crux measurement
bash scripts/03_kernel_attrib.sh   # which kernel actually ran
bash scripts/04_serve_bench.sh     # serving under concurrency (the Cloud AI regime)
python3 tools/analyze.py           # adjudicated result table
```

**No Arm machine?** Fork this repo and run the `crux` workflow. It executes on
`ubuntu-24.04-arm`, a free Arm-hosted GitHub runner (Cobalt 100 / Neoverse N2 —
Armv9 with i8mm, bf16 and SVE2), free for public repositories. Every number here
is reproducible by a stranger at zero cost.

Validate the adjudicator on any machine, including x86:

```bash
python3 tools/test_skeptic.py
```

## Not all free Arm is the same Arm

This trips people up, and it decides whether the experiment means anything:

| Host | Core | i8mm | Role |
|:--|:--|:--:|:--|
| GitHub `ubuntu-24.04-arm` | Cobalt 100 / **Neoverse N2** (Armv9) | ✅ | **Subject** — the experiment is valid here |
| AWS `t4g.small` (free trial) | Graviton2 / **Neoverse N1** | ❌ | **Control only** |
| AWS `c7g` / `c8g` | Graviton3 / 4 | ✅ | Subject, cleaner numbers (not free) |

On a core with no i8mm there is no SMMLA kernel to wake up, so a null result
proves nothing. `00_env_report.sh` detects this and labels the host `control`
rather than letting it produce a misleading negative.

### The N1 box is the control group, not a consolation prize

Throughput rises with batch size on *any* core, simply because streaming the
weights once amortizes across more tokens. So a knee on a single machine cannot
distinguish "the i8mm kernel woke up" from "batching is just good."

The design that separates them measures KleidiAI **ON vs OFF on each core
independently**, then compares the two deltas:

| | KleidiAI delta at low batch | KleidiAI delta at high batch | reading |
|:--|:--|:--|:--|
| **N2** (has i8mm) | small | **grows** | consistent with reaching i8mm |
| **N1** (no i8mm) | small | flat | amortization alone |

If the KleidiAI advantage widens with batch size on N2 but stays flat on N1 —
which has no i8mm path to reach — the widening is attributable to i8mm rather
than to batching. That is a difference-in-differences, and it is far stronger
than any single-machine curve.

Being explicit about its limits: N1 and N2 differ in far more than i8mm (IPC,
caches, memory, generation), so **cross-machine absolute numbers are not
comparable**. Only the within-machine ON/OFF deltas are, which is exactly what
this design compares.

## What each piece does

| Path | Role |
|:--|:--|
| `scripts/00_env_report.sh` | Identifies the core via MIDR, detects i8mm/sve2/bf16/dotprod, records SVE vector length, flags virtualization. **Gates the whole experiment.** |
| `scripts/01_build_llama.sh` | Builds llama.cpp twice from one checkout — only `GGML_CPU_KLEIDIAI` differs. Counts `kai_*` symbols to catch a flag that silently did nothing. |
| `scripts/02_batch_sweep.sh` | Sweeps `-p N` across small N. One forward pass over N tokens is the matmul shape of verifying N draft tokens. |
| `scripts/03_kernel_attrib.sh` | **Mechanism proof.** KleidiAI kernel names encode their ISA (`..._neon_dotprod` vs `..._neon_i8mm`), so the hot symbol names the kernel that ran. Degrades honestly to a capability inventory where `perf` is blocked. |
| `tools/analyze.py` | Finds the knee, adjudicates every comparison, writes the report. |
| `tools/test_skeptic.py` | Proves the adjudicator itself is sound, against synthetic data with known answers. |

## The Skeptic

Every comparison gets **VERIFIED**, **UNCERTAIN**, or **REJECTED** against
thresholds fixed in advance: ≥5 reps, relative stdev ≤10%, effect ≥5%, and
disjoint 95% confidence intervals. A failed measurement is UNCERTAIN, never
REJECTED — "we measured nothing" and "we measured no gain" are different claims.

Rejected rows get published. A harness that reports only its wins is a harness
nobody should believe.

The adjudicator is itself tested (`tools/test_skeptic.py`) against synthetic
distributions with known ground truth, including the case that matters most: a
large-looking mean difference drowned in variance must come back UNCERTAIN.

## Either outcome is a result

If the hot kernel is the same ISA path at N=1 and N=32, the "wake the kernels"
framing is wrong — and *that* is the more valuable finding: KleidiAI leaving i8mm
unused for small-batch work would be a real gap worth reporting upstream.
[Section 7 of the methodology](docs/methodology.md) commits to reporting that
outcome with equal prominence, written down before the data existed.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[Arm KleidiAI](https://gitlab.arm.com/kleidi/kleidiai). Model:
[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
(Apache-2.0). Motivated by llama.cpp
[issue #21453](https://github.com/ggml-org/llama.cpp/issues/21453), an open
request for speculative decoding work targeting low-latency CPU inference.

# Methodology (pre-registered)

**This document was committed before any measurement was taken.** Check `git log`
on this file. Thresholds chosen after seeing results are not thresholds, they are
decoration — the timestamp is the point.

---

## 1. The claim under test

> On Arm CPUs, single-token LLM decode runs at batch size 1. A batch-1 matmul is
> a **GEMV** (matrix × vector), which cannot make use of KleidiAI's **i8mm**
> microkernels — those are built on `SMMLA`, a matrix-multiply instruction that
> needs multiple rows to be worthwhile. So the fastest integer matmul path on the
> chip is largely idle during the phase where nearly all inference time is spent.
>
> Speculative decoding verifies *N* draft tokens in one forward pass. If that
> moves the matmul into a regime where the i8mm kernels are selected, then
> speculative decoding on Arm buys **two** things, not one: fewer weight-streaming
> passes, *and* a better kernel.

The second half of that is what this repo is actually testing. It may be false.

## 2. Why `llama-bench -p N` is the right proxy

Verifying N draft tokens is one forward pass over N tokens. Processing a prompt
of N tokens is also one forward pass over N tokens. The matmul shapes are the
same, so `-p N` for small N measures the cost of speculative verification of N
tokens without needing a working speculative decoder first.

This is a proxy, and it is worth being precise about the ways it is imperfect:

- It excludes draft-model cost. Real speculative decoding pays for the draft
  model too. This measures the **verification side only** — the side where the
  kernel question lives.
- It excludes acceptance-rate effects entirely. End-to-end speedup depends on how
  many drafted tokens survive, which is a property of the model pair and the
  workload, not of the kernel.
- KV-cache state differs slightly between prompt processing and mid-generation
  verification.

So: `-p N` is evidence about **kernel selection**, not a prediction of end-to-end
speedup. Those are reported as separate claims and must never be merged into one
headline number.

## 3. Pre-registered thresholds

Defined in `tools/analyze.py` and enforced by `tools/test_skeptic.py`.

| Parameter | Value | Why |
|:--|--:|:--|
| `MIN_EFFECT_PCT` | 5.0% | Below this, a difference is not worth a developer's time even if it is real |
| `MIN_REPS` | 5 | Fewer samples cannot support a confidence interval worth quoting |
| `MAX_RSD_PCT` | 10.0% | Above this the host was too noisy for the run to mean anything |
| Confidence level | 95%, two-sided | Small-sample `t` critical values, not a normal approximation |

## 4. Verdict rules

Every KleidiAI-vs-baseline comparison gets exactly one verdict:

- **VERIFIED** — ≥ `MIN_REPS` samples on both sides, both relative standard
  deviations within `MAX_RSD_PCT`, effect ≥ `MIN_EFFECT_PCT`, and the 95%
  confidence intervals are **disjoint**.
- **UNCERTAIN** — real-looking but not defensible: CIs overlap, or the effect is
  below the floor, or the host was too noisy, or too few reps, or the measurement
  failed outright.
- **REJECTED** — no improvement, or a regression.

A failed measurement is **UNCERTAIN**, never REJECTED. "We measured nothing" and
"we measured no gain" are different claims and conflating them would let a broken
run masquerade as a negative result.

**Rejected and uncertain rows are published.** A harness that reports only its
wins is a harness nobody should believe, including its author.

## 5. Evidence classes

Throughput curves show *correlation*. The mechanism claim needs more, so every
kernel finding is tagged:

- **`execution`** — `perf` sampled the running process and a KleidiAI symbol was
  hot. KleidiAI microkernel names encode their ISA path
  (`..._neon_dotprod` vs `..._neon_i8mm`), so the hot symbol *names* the kernel
  that ran. This is mechanism proof.
- **`capability_only`** — `perf` was unavailable (typical in virtualized CI), so
  we can only report which kernels were **linked into the binary**, not which
  **executed**. This is *not* mechanism proof and is labelled as such everywhere
  it appears.

No result is upgraded from `capability_only` to `execution` by inference.

## 6. Known confounds, and what we do about them

| Confound | Handling |
|:--|:--|
| Throughput rises with batch size anyway (weight-streaming amortization) | A knee is reported as *consistent with* a kernel switch, never as proof. Symbol evidence decides. |
| Burstable instances (AWS `t4g`) throttle mid-run | `virtualization` and load are recorded; noisy runs fail the RSD gate automatically |
| CI runners are shared and noisy | ≥10 reps by default; RSD gate; CI-overlap gate |
| Build differences other than the flag | Both variants built from one checkout, one compiler, one flag differs |
| KleidiAI silently not compiled in | `01_build_llama.sh` counts `kai_*` symbols and warns if zero |
| Core lacks i8mm entirely (Graviton2/N1) | `00_env_report.sh` marks the host `control`; a null result there is meaningless and is labelled so |

## 7. What would falsify the thesis

Stated in advance so it cannot be quietly retired later:

1. If the hot KleidiAI symbol is **the same ISA path at N=1 and N=32**, there is
   no kernel switch to exploit and the "wake the kernels" framing is wrong.
2. If i8mm kernels are already selected at N=1, the premise is simply false.
3. If the throughput knee sits far above realistic draft lengths (say N > 32),
   the effect is real but useless — speculative decoding does not reach it.

**Outcome 1 or 2 does not end the project.** It inverts it: "KleidiAI leaves i8mm
unused for small-batch work on Neoverse" is a more valuable upstream finding than
confirming the expected result, and it is reported with equal prominence. This is
written down now, in advance, so that reporting a negative result later is
visibly the plan rather than a consolation prize.

## 8. Claims this repo will not make

- No end-to-end speedup number until an actual speculative decoder is measured
  end-to-end. Proxy measurements stay labelled as proxy measurements.
- No SVE2 gain attributed to vector width. On Neoverse N2, SVE2 is **128-bit** —
  the same width as NEON. Any SVE2 benefit must come from predication removing
  tail loops and branches, and will be described that way.
- No cross-vendor claims (Graviton vs Cobalt vs Axion) from a single instance of
  each. Different silicon, different neighbours, different noise.
- No number without the machine it came from, in the same table.

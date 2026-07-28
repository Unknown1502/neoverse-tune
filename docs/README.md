# SpecArm documentation

Start here. Each document answers a different question.

| Document | Answers | Read it when |
|:--|:--|:--|
| [methodology.md](methodology.md) | *What is being tested, and what would prove it wrong?* | Before trusting any number here |
| [architecture.md](architecture.md) | *How is it built, and why that way?* | Before changing code |
| [flows.md](flows.md) | *What happens, in what order?* | When something behaves unexpectedly |
| [metrics.md](metrics.md) | *What does this number actually mean?* | When reading a report |
| [reproducing.md](reproducing.md) | *How do I run this myself?* | To reproduce or extend the results |

## The short version

Batch-1 LLM decode on Arm is a **GEMV**, and GEMV cannot use KleidiAI's **i8mm**
(`SMMLA`) matmul microkernels — those need multiple rows. So the fastest integer
matmul path on the chip is largely idle during the phase that consumes nearly all
inference time.

Two things can raise the effective batch size: **serving concurrency** (a server
batches concurrent requests) and **speculative decoding** (N draft tokens
verified in one forward pass). SpecArm measures whether either reaches the regime
where i8mm kernels get selected, and proves which kernel actually ran instead of
inferring it from a faster stopwatch.

## Reading order for a first-time reader

1. [methodology.md §1–2](methodology.md) — the claim and why the proxy is valid
2. [architecture.md §2](architecture.md#2-high-level-architecture) — the four stages
3. [reproducing.md §A](reproducing.md#a-github-arm-runner--the-subject-free) — run it free
4. [metrics.md](metrics.md) — interpret what comes out

## Three things that are easy to get wrong

**A knee is not a kernel switch.** Throughput rises with batch size on any core,
because streaming the weights once amortizes across more tokens. Only the symbol
evidence in [`03_kernel_attrib.sh`](../scripts/03_kernel_attrib.sh) attributes it.

**`capability_only` is not proof.** When `perf` is restricted — normal in hosted
CI — SpecArm can report which kernels were *linked*, not which *ran*. It labels
this and never upgrades it by inference.

**A `control` host cannot produce a negative result.** Neoverse N1 (AWS
Graviton2) has no i8mm at all, so there is nothing to wake up. A null result
there is meaningless, not evidence, and is labelled that way.

## Claims this project will not make

Listed in full in [methodology.md §8](methodology.md#8-claims-this-repo-will-not-make).
The short list: no end-to-end speedup number from a proxy measurement, no SVE2
gain attributed to vector width (N2's SVE2 is 128-bit — the same as NEON), no
cross-vendor claims from one instance each, and no number without the machine it
came from in the same table.

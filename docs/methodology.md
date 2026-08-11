# Methodology

## 1. Honest history

This repository tested three hypotheses before this one and killed all three
with measurement. That history is in `git log` and is not hidden, because it is
the reason to trust the fourth.

| # | Hypothesis | How it died |
|--:|:--|:--|
| 1 | Decode fails to reach KleidiAI's i8mm kernels, so speculative decoding can "wake" them | KleidiAI ships an `mr=1` dotprod kernel precisely for batch-1 decode. The dispatch was correct; the framing described intended behaviour as a defect. |
| 2 | Agent servers re-prefill the identical system prompt every turn | Measured. `probe_prefill.py` returned **REUSE_WORKS** — single-tenant prefix caching already works, TTFT falls 811 ms → 327 ms across turns. |
| 3 | The regression is slot *capacity* | Ruled out. `-np 4` for 4 tenants recovered 14% of a 610% regression; 4x the per-slot context changed nothing. |
| 4 | **Slot *selection* cannot distinguish tenants that share a large preamble** | Survives. This document. |

Each was killed in under an hour by measuring before building. Hypothesis 2 in
particular was killed *before* two weeks of work went into a fix for a problem
that did not exist.

## 2. The claim

> `llama-server` assigns requests to slots by longest-common-prefix similarity,
> accepting any slot above `--slot-prompt-similarity` (default **0.10**).
> Similarity is approximately `shared_prefix / total_prompt`. Tenants of one
> agent share a large system prompt and tool schema, so any two tenants are
> already 0.6–0.9 similar. Against a 0.10 threshold, every slot qualifies for
> every tenant, tenants scatter, and each recomputes its own history each turn.

The corollary is the interesting part, and it is counterintuitive: **a larger
shared preamble makes the default worse, not better.**

## 3. Pre-registered thresholds

Defined in `tools/skeptic.py` and unchanged since commit `20a6031`, which
predates every measurement in this repository. The *hypothesis* changed three
times; the *adjudication rules* never did. Check `git log`.

| Parameter | Value | Why |
|:--|--:|:--|
| `MIN_EFFECT_PCT` | 5.0% | Below this, a difference is not worth a developer's time |
| `MIN_REPS` | 5 | Fewer samples cannot support an interval worth quoting |
| `MAX_RSD_PCT` | 10.0% | Above this the host was too noisy to conclude anything |
| Confidence | 95%, two-sided | Small-sample `t` values, not a normal approximation |

**VERIFIED** requires all four: enough repeats, acceptable noise, effect above
the floor, and disjoint intervals. A failed measurement is **UNCERTAIN**, never
REJECTED.

## 4. Experimental design

**Independent samples are per-repeat medians, not individual requests.**
Requests inside one run share a server, a cache state and a schedule. Pooling
them would shrink the intervals to nothing and manufacture significance. `n`
therefore equals the repeat count.

**A fresh server per repeat.** A second run against a warm server measures
carry-over, not the configuration.

**Sequential requests.** Tenants take turns; nothing overlaps in time. This
removes queueing and thread contention as explanations — four users would thrash
this server even if they never collided.

**Distinct tenant content.** Every tenant has its own cities, airports, dates and
phrasing. An earlier version gave all tenants identical turn text differing only
in an opening line, which inflated inter-tenant similarity to 0.906 and
exaggerated the regression. That run was discarded.

**Exact token counts** from the server's own tokenizer via `/apply-template` and
`/tokenize`, with the method recorded — a count from a fallback heuristic is not
the same claim as one from the server.

## 5. Two classes of evidence

| Metric | What it is | Why it matters |
|:--|:--|:--|
| **Tokens recomputed** | From the server's own log | The **cause**. Immune to CPU noise, thermal state, scheduling. |
| **Time to first token** | Wall clock | The **consequence**. Machine-dependent. |

The claim rests on the first. The second is what a user feels.

## 6. What would falsify this

Stated in advance so it cannot be quietly retired:

1. If tokens-recomputed is the same at 1 tenant and 4, there is no scattering
   and the mechanism is wrong.
2. If raising the threshold does not reduce tokens-recomputed, the threshold is
   not the lever.
3. If the effect vanishes on Arm, the finding is x86-specific and must be
   reported as such — it is scheduler behaviour, so this would be surprising.

### Outcome of test 3 — the attempt failed

Run on 2026-08-10, Neoverse N2 (Cobalt 100, 4 vCPU), same GGUF verified
byte-identical by SHA-256, llama.cpp `030ebb55`. Raw data in
`results/arm-neoverse-n2/`.

| | Neoverse N2 | x86 8-thread |
|:--|--:|--:|
| Regression, 1 → 4 tenants | **10.89x** | 4.60x |
| Recovery at `0.9` | **10.5x** | 5.0x |
| Median LCP similarity at default | **0.716** | 0.716 |
| Median tokens recomputed at default | **220** | 220 |
| Median RSD across configs | **0.4%** | 4.9% |

The effect did not vanish; it is more than twice as large. The mechanism
evidence is **byte-identical** across the two architectures, which is what a
deterministic scheduler should produce and is the strongest available evidence
that the finding is about slot selection rather than about arithmetic.

Tests 1 and 2 also hold on Arm: tokens-recomputed goes 35 → 220 from 1 tenant to
4, and raising the threshold returns it to 36.

## 7. Claims deliberately not made

- Not a claim that llama.cpp is poorly engineered. The default is reasonable for
  chat, where the shared prefix is small.
- Not a throughput claim. TTFT only.
- Not a claim that the *mechanism* is Arm-specific — it reproduces identically on
  x86. What differs by machine is the **cost**.
- Not an attribution of the 10.9x-vs-4.6x gap to microarchitecture. The two hosts
  differ in architecture *and* core count, so that comparison is confounded. A
  same-core-count run across Neoverse generations would be needed, and has not
  been done.
- No number without the machine it came from in the same table.

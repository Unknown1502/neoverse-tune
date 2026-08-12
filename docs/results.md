# Results — complete reference

Every number this project has produced, with the machine that produced it. No
figure appears here without its host, its sample count, and its interval.

Generated from `results/agent_analysis.latest.json`,
`results/arm-neoverse-n2/agent_analysis.latest.json`, `results/slot_evidence.json`
and `results/tune.latest.json`. Regenerate any of it with
`python3 tools/analyze_agent.py` and `python3 tools/parse_slot_log.py`.

- [Hosts](#hosts)
- [Workload](#workload)
- [Latency, both machines](#latency-both-machines)
- [Adjudicated comparisons](#adjudicated-comparisons)
- [Mechanism evidence](#mechanism-evidence)
- [The prefill cost model](#the-prefill-cost-model)
- [Threshold sweep](#threshold-sweep)
- [Prior experiments that failed](#prior-experiments-that-failed)
- [What none of this establishes](#what-none-of-this-establishes)

---

## Hosts

| | **Neoverse N2** | **x86** |
|:--|:--|:--|
| CPU | Arm Neoverse N2, MIDR `0x41`/`0xd49` | x86-64 |
| Product | Microsoft Cobalt 100 | consumer laptop |
| Provider | Azure, via GitHub-hosted runner | local |
| Cores | 4 vCPU | 8 threads |
| ISA | `i8mm`, `bf16`, `sve2` @ 128-bit, `asimddp` | AVX-family |
| `int8_matmul_path` | **`i8mm`** | n/a |
| OS | Ubuntu 24.04.4 LTS, kernel 6.17.0-azure | Windows |
| Compiler | cc 13.3.0 | Clang 20.1.8 |
| llama.cpp | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` | `10178 (992c32532)` |
| Virtualization | `microsoft` | none |
| `perf` usable | no (`perf_event_paranoid=4`) | n/a |
| Wall time, 30 runs | 47.0 min | 46.9 min |

**The two columns used different llama.cpp builds.** Stated because it matters
twice: it weakens any claim that depends on version, and it strengthens the
central one — the same 0.716 similarity and 220-token recompute appeared across
two versions *and* two architectures.

---

## Workload

| | |
|:--|:--|
| Model | Qwen2.5-1.5B-Instruct **Q4_K_M** |
| SHA-256 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Size | 1,117,320,736 bytes — verified identical on both hosts |
| Shared preamble | **550 tokens** — system prompt + 5 JSON tool schemas, byte-identical across tenants |
| Final prompt | ~836 tokens |
| Token counting | `apply_template+tokenize` — the server's own tokenizer |
| Turns | 4 per tenant, round-robin |
| Concurrency | **none** — strictly sequential |
| Context | 32768 |
| `max_tokens` | 48, temperature 0.0, `cache_prompt: true` |
| Repeats | 5 per configuration, **fresh server each** |
| Independent sample | per-repeat warm median (`n` = repeats, not requests) |

---

## Latency, both machines

Warm TTFT — median of rounds 2+, since round 1 is cold for every tenant by
definition.

### Neoverse N2 (4 vCPU)

| config | tenants | `-np` | sim | mean | 95% CI | sd | RSD | n |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| `solo_baseline` | 1 | 4 | 0.1 | **385.9** | [383.4, 388.4] | 2.0 | 0.53% | 5 |
| `4tenant_default` | 4 | 4 | 0.1 | **4203.7** | [4199.0, 4208.5] | 3.8 | 0.09% | 5 |
| `8tenant_default` | 8 | 4 | 0.1 | **4204.2** | [4200.3, 4208.1] | 3.1 | 0.07% | 5 |
| `4tenant_sim09` | 4 | 4 | 0.9 | **399.0** | [398.0, 400.0] | 0.8 | 0.20% | 5 |
| `8tenant_sim09` | 8 | 4 | 0.9 | **401.9** | [400.0, 403.8] | 1.5 | 0.38% | 5 |
| `8tenant_sim09_np8` | 8 | 8 | 0.9 | **398.2** | [396.0, 400.4] | 1.8 | 0.44% | 5 |

### x86 (8 threads)

| config | tenants | `-np` | sim | mean | 95% CI | sd | RSD | n |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| `solo_baseline` | 1 | 4 | 0.1 | **503.2** | [459.9, 546.5] | 34.9 | 6.93% | 5 |
| `4tenant_default` | 4 | 4 | 0.1 | **2312.7** | [2060.8, 2564.5] | 202.9 | 8.77% | 5 |
| `8tenant_default` | 8 | 4 | 0.1 | **2101.5** | [1973.9, 2229.0] | 102.7 | 4.89% | 5 |
| `4tenant_sim09` | 4 | 4 | 0.9 | **458.2** | [432.1, 484.4] | 21.1 | 4.59% | 5 |
| `8tenant_sim09` | 8 | 4 | 0.9 | **495.6** | [473.8, 517.4] | 17.6 | 3.55% | 5 |
| `8tenant_sim09_np8` | 8 | 8 | 0.9 | **464.6** | [450.2, 479.1] | 11.7 | 2.51% | 5 |

**Median RSD: 0.41% on N2, 4.89% on x86** — a dedicated cloud instance is an
order of magnitude quieter than a laptop running a desktop session. That
precision is what let the Arm run resolve a 4% effect the laptop could not
(see parity, below).

---

## Adjudicated comparisons

Every comparison is declared in `COMPARISONS` before any data is read, with the
direction that counts as interesting. Latency is inverted to a rate before
adjudication, because `verdict()` is written higher-is-better.

| Question | ref → cand | **N2** | ratio | **x86** | ratio |
|:--|:--|:--|--:|:--|--:|
| Does adding tenants break prefix caching? | `solo` → `4t_default` | 🔴 **REGRESSION** | 10.894 | 🔴 REGRESSION | 4.596 |
| Does it get worse with more tenants? | `solo` → `8t_default` | 🔴 **REGRESSION** | 10.895 | 🔴 REGRESSION | 4.176 |
| Does raising the threshold fix it? | `4t_default` → `4t_sim09` | ✅ **VERIFIED** +953.6% | 0.095 | ✅ VERIFIED +402.3% | 0.198 |
| Does the fix hold when tenants exceed slots? | `8t_default` → `8t_sim09` | ✅ **VERIFIED** +946.0% | 0.096 | ✅ VERIFIED +323.6% | 0.236 |
| Back at single-tenant parity? | `solo` → `8t_sim09` | ❌ **REJECTED** −4.0% | 1.042 | ⚠️ UNCERTAIN | 0.985 |
| Do extra slots help? | `8t_sim09` → `8t_sim09_np8` | ⚠️ UNCERTAIN +0.9% | 0.991 | ⚠️ UNCERTAIN +6.6% | 0.937 |

### The three rows that cost us something

**Parity is REJECTED on Arm.** The fix recovers 10.5x of a 10.9x regression —
but not all of it. At 0.4% RSD the measurement resolves a real residual 4% gap
that the laptop's overlapping intervals could not see. A better instrument
produced a worse-sounding, truer answer, and it stays in the table.

**Extra slots do nothing.** `-np 8` for 8 tenants is indistinguishable from
`-np 4` once the threshold is right, on both machines. This is what rules out
slot *capacity* as the explanation, and it is the reason the finding is about
selection rather than provisioning.

**8 tenants ≈ 4 tenants.** 4203.7 vs 4204.2 ms at 0.1% RSD — half a millisecond
apart. Scattering saturates at 4 tenants against 4 slots: every tenant already
matches every slot, and an already-total failure cannot become more total. The
data therefore supports "multi-tenant is far worse than single-tenant" and does
**not** support "more tenants is monotonically worse." The second claim is made
nowhere in this repository.

---

## Mechanism evidence

From `llama-server`'s own log. Deterministic at temperature 0, and **identical
on both architectures** — which is exactly what a deterministic scheduler should
produce, and the strongest available evidence that this is about slot selection
rather than arithmetic.

| config | median LCP similarity | median tokens recomputed | >100 tok | LRU fallbacks |
|:--|--:|--:|--:|--:|
| `solo_baseline` | 0.955 | **35** | 25% | 1 |
| `4tenant_default` | **0.716** | **220** | 81% | 1 |
| `8tenant_default` | **0.716** | **220** | 78% | 1 |
| `4tenant_sim09` | 0.955 | **36** | 25% | 4 |
| `8tenant_sim09` | 0.905 | **56** | 31% | 25 |
| `8tenant_sim09_np8` | 0.955 | **36** | 25% | 8 |

Verbatim from the server, at the default:

```
slot get_availabl: id  0 | task 12 | selected slot by LCP similarity, f_sim_best = 0.716 (> 0.100 thold), f_keep = 1.000
slot print_timing: id  0 | task 12 | prompt eval time = 4238.90 ms / 220 tokens (19.27 ms per token, 51.90 tokens per second)
```

**Why `8tenant_sim09` is the interesting row.** 8 tenants against 4 slots at a
0.9 threshold means most tenants have no acceptable slot, so selection falls
through to LRU 25 times. Median similarity drops to 0.905 and tokens rise to 56.
It still beats the default by 4.2x — but it shows the fix is not free when
tenants outnumber slots, and `8tenant_sim09_np8` (8 slots) returns it to 0.955
and 36.

---

## The prefill cost model

`llama-server` reports prefill throughput on every request, which converts the
scheduler's decision into a wall-clock cost:

```
    latency lost  =  tokens recomputed  ÷  prefill throughput
                     └── the scheduler ──┘   └── the hardware ──┘
                        identical on both      differs by host
```

| host | tokens | prefill rate | **modelled** | **measured** | error |
|:--|--:|--:|--:|--:|--:|
| Neoverse N2, 4 vCPU | 220 | 51.9 tok/s | **4332 ms** | 4204 ms | 3.0% |
| x86, 8 threads | 220 | 98.6 tok/s | **2282 ms** | 2313 ms | 1.4% |

Two numbers from the server's own log predict measured TTFT within 3% on both
architectures.

It also accounts for the headline gap without invoking anything exotic:
regressions of 10.89x and 4.60x are a ratio of **2.37**, and the prefill rates
differ by **1.90x**. Most of how bad the bug *feels* is simply how fast the host
can re-prefill.

**This does not isolate microarchitecture.** The N2 host has 4 vCPU against the
x86 host's 8 threads, so its lower prefill rate confounds architecture with core
count. The model says cost tracks prefill throughput; it does not say why one
host prefills faster. Separating that needs two Neoverse generations at matched
thread count — N1 (`dotprod` only) against N2 (`i8mm`) — which has not been run.

---

## Threshold sweep

7 thresholds × 3 repeats, fresh server each run, 4 tenants, x86.

| threshold | 0.1 | 0.3 | 0.5 | 0.7 | **0.8** | 0.9 | 0.95 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| warm median | 2360 | 1995 | 1990 | 1934 | **399** | 485 | 448 |
| samples | 2360, 2446, 2027 | 1995, 2129, 1983 | 2055, 1990, 1895 | 2031, 1934, 1910 | 419, 399, 395 | 416, 485, 485 | 462, 448, 432 |

Complete separation: **1934–2360 ms below, 399–485 ms above.** No overlap.

**The cliff falls between 0.7 and 0.8, and measured inter-tenant similarity is
0.716.** A threshold at or below 0.716 admits foreign slots; above it they are
rejected. The sweep was not designed to test that prediction and confirms it.

Two qualifications:

- **`0.8`, `0.9`, `0.95` are not distinguishable.** n=3, no intervals. `--sweep` reports the lowest median and labels it "measured best"; that is a ranking, not a result.
- **n=3 is below `MIN_REPS`.** The sweep is deliberately not run through `verdict()` — every comparison would return UNCERTAIN, correctly. It is a search tool, not evidence.

What it establishes is the **shape**: a cliff, not a gradient. Tuning means
clearing a number you can measure, not hunting an optimum.

---

## Prior experiments that failed

Three hypotheses died before this one survived. All in `git log`.

| # | Hypothesis | How it died |
|--:|:--|:--|
| 1 | Decode never reaches KleidiAI's `i8mm` kernels | KleidiAI ships an `mr=1` dotprod kernel *for* batch-1 decode. Dispatch was correct; the framing described intended behaviour as a defect. Killed by reading. |
| 2 | Agent servers re-prefill the system prompt every turn | Measured. `probe_prefill.py` returned **REUSE_WORKS** — TTFT 811 ms → 327 ms across turns, ratio 0.054. Killed in 45 minutes, cancelling ~2 weeks of planned work. |
| 3 | The regression is slot *capacity* | `-np 4` for 4 tenants recovered 14% of a 610% regression; 4x per-slot context changed nothing. |
| 4 | Slot *selection* cannot distinguish tenants sharing a large preamble | Survives. This document. |

An earlier version of the workload also had all tenants sending identical turn
text, which inflated inter-tenant similarity to **0.906** and exaggerated the
regression — in the flattering direction. That run was discarded and the
workload rewritten with per-tenant itineraries.

---

## What none of this establishes

- **Concurrency.** Every measurement is strictly sequential. This isolates slot selection from queueing, and it deletes the dominant real-world variable. Largest open gap.
- **Generality across models.** One model, one quantization.
- **Preamble-size scaling.** The central claim is *about* preamble size; one value (550 tokens) was tested.
- **Microarchitecture attribution.** Two hosts differing in two variables at once.
- **Throughput.** TTFT only. Nothing here says what `0.9` costs in tokens/sec.
- **Memory.** Higher slot affinity changes KV residency; unmeasured.
- **That `0.9` is optimal.** It is above the cliff. So is 0.8, and so is 0.95.
- **A failure case for the recommendation.** No workload was found where raising the threshold *hurts*. One probably exists — chat with short prefixes and many users is the obvious candidate — and it was not looked for.

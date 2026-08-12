# Documentation

| Document | Answers |
|:--|:--|
| [reproducing.md](reproducing.md) | **How to run it and check the numbers** — four routes, expected values, troubleshooting, how to prove it wrong |
| [results.md](results.md) | **Every number, with its machine** — both hosts, all comparisons, mechanism, cost model, sweep, and the failed hypotheses |
| [architecture.md](architecture.md) | What the pieces are — system context, module map, internals, data model, deployment |
| [flows.md](flows.md) | How it runs — the mechanism, one turn, the matrix, adjudication, CI |
| [methodology.md](methodology.md) | What is being tested, how, and what would prove it wrong |
| [../README.md](../README.md) | The finding and how to reproduce it |

**Reviewing this for the first time?** [reproducing.md](reproducing.md) is the
one to read — it states what you should see, and how to tell a real difference
from a broken run.

All diagrams are Mermaid and render inline on GitHub. Start with
[flows.md §1](flows.md#1-the-mechanism--how-a-slot-gets-chosen-wrong) — it is the
one diagram that explains why the project exists.

## The short version

`llama-server` picks a slot by longest-common-prefix similarity and accepts any
slot above `--slot-prompt-similarity`, which defaults to **0.10**. Tenants of a
single agent all carry the same system prompt and tool schemas, so any two of
them are already 0.6–0.9 similar. Every slot therefore qualifies for every
tenant, tenants scatter across slots, and each recomputes its own conversation
history on every turn.

Measured: 4 tenants, 550-token shared preamble, 836-token conversations.
**220 tokens recomputed per request at the default, 36 at `0.9`.**

| | Neoverse N2 (4 vCPU) | x86 (8 threads) |
|:--|--:|--:|
| 1 tenant | 386 ms | 503 ms |
| 4 tenants, default | **4204 ms** (10.9x) | 2313 ms (4.6x) |
| 4 tenants, `0.9` | 399 ms | 458 ms |
| tokens recomputed at default | **220** | **220** |

The token counts are byte-identical across architectures. The scheduler makes
the same decision on both; only the price differs.

## Three things that are easy to get wrong

**Tokens-recomputed is the finding; TTFT is the symptom.** Latency folds in
scheduling, thermal state and memory pressure. The token count comes from the
server's own log and is immune to all of it.
→ [flows.md §7](flows.md#7-mechanism-extraction-from-logs)

**Independent samples are per-repeat medians.** Requests inside one run share a
server and a cache state. Treating them as independent would shrink the
confidence intervals to nothing.
→ [architecture.md — data model](architecture.md#data-model)

**A bigger system prompt makes the default worse.** More shared context raises
inter-tenant similarity, so a fixed low threshold separates tenants *less* well.
This is backwards from intuition and is why the problem is not obvious.
→ [flows.md §1](flows.md#1-the-mechanism--how-a-slot-gets-chosen-wrong)

## Why the git history is worth reading

Three earlier hypotheses were killed by measurement before this one survived —
including one that was killed *before* two weeks went into fixing a problem that
did not exist. [methodology.md §1](methodology.md#1-honest-history) lists them,
and [flows.md §10](flows.md#10-how-three-hypotheses-died) diagrams how each one
died. The adjudication thresholds have not changed across any of it.

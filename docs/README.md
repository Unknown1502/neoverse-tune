# Documentation

| Document | Answers |
|:--|:--|
| [methodology.md](methodology.md) | What is being tested, how, and what would prove it wrong |
| [../README.md](../README.md) | The finding and how to reproduce it |

## The short version

`llama-server` picks a slot by longest-common-prefix similarity and accepts any
slot above `--slot-prompt-similarity`, which defaults to **0.10**. Tenants of a
single agent all carry the same system prompt and tool schemas, so any two of
them are already 0.6–0.9 similar. Every slot therefore qualifies for every
tenant, tenants scatter across slots, and each recomputes its own conversation
history on every turn.

Measured: 4 tenants, 550-token shared preamble, 836-token conversations.
**220 tokens recomputed per request at the default, 36 at `0.9`.** Time to first
token 2313 ms versus 458 ms.

## Three things that are easy to get wrong

**Tokens-recomputed is the finding; TTFT is the symptom.** Latency folds in
scheduling, thermal state and memory pressure. The token count comes from the
server's own log and is immune to all of it.

**Independent samples are per-repeat medians.** Requests inside one run share a
server and a cache state. Treating them as independent would shrink the
confidence intervals to nothing.

**A bigger system prompt makes the default worse.** More shared context raises
inter-tenant similarity, so a fixed low threshold separates tenants *less* well.
This is backwards from intuition and is why the problem is not obvious.

## Why the git history is worth reading

Three earlier hypotheses were killed by measurement before this one survived —
including one that was killed *before* two weeks went into fixing a problem that
did not exist. [methodology.md §1](methodology.md#1-honest-history) lists them.
The adjudication thresholds have not changed across any of it.

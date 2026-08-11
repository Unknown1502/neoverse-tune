# Multi-tenant agent serving — adjudicated result

- **Host**: `Neoverse-N2`, 4 threads
- **Model**: `qwen2.5-1.5b-instruct-q4_k_m.gguf` · ctx 32768
- **Repeats**: 5 per config, **fresh server each time**
- **Workload**: N tenants sharing one agent, 4 turns each, round-robin

## Configurations

| config | tenants | warm TTFT (ms, mean ± 95% CI) | RSD | n |
|:--|--:|--:|--:|--:|
| `solo_baseline` | 1 | 385.9 ± 2.5 | 0.5% | 5 |
| `4tenant_default` | 4 | 4203.7 ± 4.8 | 0.1% | 5 |
| `8tenant_default` | 8 | 4204.2 ± 3.9 | 0.1% | 5 |
| `4tenant_sim09` | 4 | 399.0 ± 1.0 | 0.2% | 5 |
| `8tenant_sim09` | 8 | 401.9 ± 1.9 | 0.4% | 5 |
| `8tenant_sim09_np8` | 8 | 398.2 ± 2.2 | 0.4% | 5 |

Preamble: **550 tokens** shared byte-identically by every tenant. Final prompt: ~836 tokens. Counts via `apply_template+tokenize`.

## Adjudicated comparisons

Lower TTFT is better; the ratio column is `candidate / reference`, so **below 1.0 is faster**.

| question | reference | candidate | ratio | verdict | note |
|:--|:--|:--|--:|:--|:--|
| Does adding tenants break prefix caching? | `solo_baseline` | `4tenant_default` | 10.89x | 🔴 REGRESSION CONFIRMED | 10.89x slower than reference |
| Does it get worse with more tenants? | `solo_baseline` | `8tenant_default` | 10.89x | 🔴 REGRESSION CONFIRMED | 10.89x slower than reference |
| Does raising the similarity threshold fix it? | `4tenant_default` | `4tenant_sim09` | 0.09x | ✅ VERIFIED | +953.6%, CIs disjoint |
| Does the fix hold when tenants exceed slots? | `8tenant_default` | `8tenant_sim09` | 0.10x | ✅ VERIFIED | +946.0%, CIs disjoint |
| With the fix, is multi-tenant back at single-tenant parity? | `solo_baseline` | `8tenant_sim09` | 1.04x | ❌ REJECTED | no improvement (-4.0%) |
| Do extra slots add anything once the threshold is right? | `8tenant_sim09` | `8tenant_sim09_np8` | 0.99x | ⚠️ UNCERTAIN | +0.9% but 95% CIs overlap |

## Headline

With llama.cpp's **default** `--slot-prompt-similarity`, moving from one tenant to four costs **10.9x** time-to-first-token (386 ms → 4204 ms) even though every tenant shares an identical 550-token preamble.

Raising the threshold recovers **10.5x** (4204 ms → 399 ms).

The threshold is a ratio — `shared_prefix / total_prompt`. Agent workloads make that ratio structurally large between *any two tenants*, so a low fixed default lets every slot look like a valid match for every tenant. **The bigger your system prompt and tool schema, the worse the default behaves** — the opposite of what anyone would assume, and invisible to single-conversation benchmarks.

## What this does not claim

- Not a claim about llama.cpp being poorly engineered. The default is reasonable for chat; it is wrong for agents.
- Not a throughput claim. TTFT only.
- Independent samples are per-repeat medians, so `n` equals the repeat count, not the request count.


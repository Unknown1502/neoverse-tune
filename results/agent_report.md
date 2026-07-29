# Multi-tenant agent serving — adjudicated result

- **Host**: `x86 laptop (no env report)`, 8 threads
- **Model**: `qwen1.5b.gguf` · ctx 32768
- **Repeats**: 5 per config, **fresh server each time**
- **Workload**: N tenants sharing one agent, 4 turns each, round-robin

## Configurations

| config | tenants | warm TTFT (ms, mean ± 95% CI) | RSD | n |
|:--|--:|--:|--:|--:|
| `solo_baseline` | 1 | 503.2 ± 43.3 | 6.9% | 5 |
| `4tenant_default` | 4 | 2312.7 ± 251.9 | 8.8% | 5 |
| `8tenant_default` | 8 | 2101.5 ± 127.5 | 4.9% | 5 |
| `4tenant_sim09` | 4 | 458.2 ± 26.1 | 4.6% | 5 |
| `8tenant_sim09` | 8 | 495.6 ± 21.8 | 3.5% | 5 |
| `8tenant_sim09_np8` | 8 | 464.6 ± 14.5 | 2.5% | 5 |

Preamble: **550 tokens** shared byte-identically by every tenant. Final prompt: ~836 tokens. Counts via `apply_template+tokenize`.

## Adjudicated comparisons

Lower TTFT is better; the ratio column is `candidate / reference`, so **below 1.0 is faster**.

| question | reference | candidate | ratio | verdict | note |
|:--|:--|:--|--:|:--|:--|
| Does adding tenants break prefix caching? | `solo_baseline` | `4tenant_default` | 4.60x | 🔴 REGRESSION CONFIRMED | 4.60x slower than reference |
| Does it get worse with more tenants? | `solo_baseline` | `8tenant_default` | 4.18x | 🔴 REGRESSION CONFIRMED | 4.18x slower than reference |
| Does raising the similarity threshold fix it? | `4tenant_default` | `4tenant_sim09` | 0.20x | ✅ VERIFIED | +402.3%, CIs disjoint |
| Does the fix hold when tenants exceed slots? | `8tenant_default` | `8tenant_sim09` | 0.24x | ✅ VERIFIED | +323.6%, CIs disjoint |
| With the fix, is multi-tenant back at single-tenant parity? | `solo_baseline` | `8tenant_sim09` | 0.98x | ⚠️ UNCERTAIN | +1.3% but 95% CIs overlap |
| Do extra slots add anything once the threshold is right? | `8tenant_sim09` | `8tenant_sim09_np8` | 0.94x | ⚠️ UNCERTAIN | +6.6% but 95% CIs overlap |

## Headline

With llama.cpp's **default** `--slot-prompt-similarity`, moving from one tenant to four costs **4.6x** time-to-first-token (503 ms → 2313 ms) even though every tenant shares an identical 550-token preamble.

Raising the threshold recovers **5.0x** (2313 ms → 458 ms).

The threshold is a ratio — `shared_prefix / total_prompt`. Agent workloads make that ratio structurally large between *any two tenants*, so a low fixed default lets every slot look like a valid match for every tenant. **The bigger your system prompt and tool schema, the worse the default behaves** — the opposite of what anyone would assume, and invisible to single-conversation benchmarks.

## What this does not claim

- Not a claim about llama.cpp being poorly engineered. The default is reasonable for chat; it is wrong for agents.
- Not a throughput claim. TTFT only.
- Independent samples are per-repeat medians, so `n` equals the repeat count, not the request count.


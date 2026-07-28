# Metrics

Every number SpecArm reports, defined precisely: how it is computed, what it is
sensitive to, and — the part usually left out — what it does **not** mean.

---

## 1. Kernel-regime metrics (`02_batch_sweep.sh`)

### `pp<N>` — prompt-processing throughput at batch N

| | |
|:--|:--|
| **Unit** | tokens/second |
| **Source** | `llama-bench -p N -o json`, field `avg_ts` / `samples_ts` |
| **Meaning** | Rate of processing N tokens in a single forward pass |
| **Why we use it** | One forward pass over N tokens is the matmul shape of verifying N speculative draft tokens |

**Does not mean:** the speed of speculative decoding. It excludes draft-model
cost and acceptance rate entirely. It is evidence about *kernel selection*, not a
prediction of end-to-end speedup.

**Sensitive to:** thread count, `ubatch` size (must exceed N or the batch is
split), memory bandwidth, CPU frequency.

### `tg<N>` — token-generation throughput

| | |
|:--|:--|
| **Unit** | tokens/second |
| **Source** | `llama-bench -n N -o json` |
| **Meaning** | Single-token autoregressive decode — the batch=1 GEMV regime |

This is the baseline the entire project is about: the regime where i8mm kernels
are hypothesised to be unreachable.

### `knee`

| | |
|:--|:--|
| **Unit** | percent |
| **Formula** | `max over adjacent (N₀,N₁) of (ts(N₁) − ts(N₀)) / ts(N₀) × 100` |
| **Meaning** | Largest single step-up in throughput between adjacent batch sizes |

**Does not mean:** a kernel switch occurred. Throughput rises with batch size on
any core because streaming weights once amortizes across more tokens. The knee
*locates a candidate*; only symbol evidence can attribute it.

## 2. Mechanism metrics (`03_kernel_attrib.sh`)

### `hottest_kai_isa`

| | |
|:--|:--|
| **Values** | `dotprod` · `i8mm` · `sme` · `sme2` · `neon` · `other` |
| **Source** | `perf report --sort symbol`, classified by KleidiAI symbol name |

KleidiAI microkernel names encode their ISA path, so the hot symbol names the
kernel that executed. Classification tests most-specific first, because i8mm and
SME kernel names also contain the substring `neon`.

### `kai_cycles_pct`

Share of sampled cycles attributed to symbols beginning `kai_`. Indicates how
much of runtime KleidiAI actually owns — if this is low, KleidiAI is not the
thing determining performance and any A/B difference needs another explanation.

### `evidence_class`

| Value | Means | Proof? |
|:--|:--|:--|
| `execution` | `perf` sampled the process; a `kai_` symbol was hot | **Yes** |
| `capability_only` | `perf` blocked; only *linked* kernels enumerable | **No** |

Never inferred upward. A `capability_only` result is reported as
`capability_only` even when the throughput curve looks convincing.

### `kernel_switch_observed`

`true` / `false` / `null`. Null means could not measure — distinct from `false`,
which means measured and no switch found.

## 3. Serving metrics (`tools/loadgen.py`)

### `ttft_ms` — time to first token

| | |
|:--|:--|
| **Formula** | `t(first content delta) − t(request sent)` |
| **Dominated by** | prefill — the **GEMM** regime |

Includes queueing delay when concurrency exceeds server slots. That is why
`-np` is set to peak concurrency: otherwise TTFT measures a queue.

### `tpot_ms` — time per output token

| | |
|:--|:--|
| **Formula** | `(t(last token) − t(first token)) / max(tokens − 1, 1)` |
| **Dominated by** | decode — the **GEMV** regime |

The first token is deliberately excluded. Including it would fold prefill cost
into a decode metric and blur exactly the boundary under study.

`test_loadgen.py` asserts this separation holds against a mock server with a
planted 120 ms prefill stall and 20 ms inter-token stall.

### `output_tokens_per_sec`

| | |
|:--|:--|
| **Formula** | `total output tokens across all clients / wall seconds` |
| **Meaning** | The headline serving number: aggregate delivered throughput |

**Does not mean:** per-request speed. Aggregate throughput rises with concurrency
even while individual requests get slower. Read it alongside `ttft_ms.p95`.

### Percentiles

Nearest-rank, computed explicitly rather than via a library, so results do not
depend on which interpolation convention a given numpy version uses.

`p95` and `p99` are reported because a mean latency under concurrency hides the
tail, and the tail is what a production service is actually judged on.

### `requests_failed`

Any non-zero value invalidates the throughput number for that cell and produces a
non-zero exit. A "throughput" measured while requests were erroring is not
throughput.

## 4. Statistical quantities (`tools/analyze.py`)

| Quantity | Definition |
|:--|:--|
| `mean` | Arithmetic mean of per-repetition samples |
| `median` | Reported alongside the mean; large divergence signals skew |
| `stdev` | Sample standard deviation (n−1) |
| `rsd_pct` | `stdev / mean × 100` — noise gate input |
| `ci95_lo/hi` | `mean ± t₀.₉₅(n−1) × stdev / √n` |

Small-sample **t** critical values are used rather than 1.96, because at n=5 the
normal approximation understates the interval by roughly 30% — which would
manufacture VERIFIED verdicts out of thin data.

### Verdicts

| Verdict | Requires |
|:--|:--|
| **VERIFIED** | n ≥ 5 both sides · both RSD ≤ 10% · effect ≥ 5% · **95% CIs disjoint** |
| **UNCERTAIN** | measurable but not defensible, or measurement failed |
| **REJECTED** | effect ≤ 0 |

Thresholds are pre-registered in [methodology.md](methodology.md) §3 and
git-timestamped before any data existed.

## 5. Environment metrics (`00_env_report.sh`)

| Field | Why it is recorded |
|:--|:--|
| `core` | Resolved from MIDR; unknown parts reported raw, never guessed |
| `features.i8mm` | Determines whether the experiment is meaningful at all |
| `sve_vector_bits` | N2's SVE2 is 128-bit = NEON width. Prevents attributing a gain to vector width that cannot come from vector width. |
| `virtualization` | Burstable/virtualized hosts throttle and restrict PMU |
| `llama_cpp_sha` | The actual reproducibility key |
| `perf_event_paranoid` | Predicts which evidence class is achievable |
| `crux_role` | `subject` or `control` |

## 6. Metrics deliberately not collected

| Not collected | Why |
|:--|:--|
| Energy / power | No trustworthy interface on shared cloud instances |
| Absolute cross-machine comparisons | N1 vs N2 differ in IPC, cache, memory, generation — only within-machine ON/OFF deltas are comparable |
| Model quality / perplexity | KleidiAI changes kernel selection, not numerics; would be measuring noise |
| Cost per token | Requires pricing assumptions that would date immediately |

## Related

- [methodology.md](methodology.md) — thresholds and falsification criteria
- [architecture.md](architecture.md) — how the components produce these numbers

# src/

Compiled components. Everything here talks to Arm libraries directly.

| Component | What it is |
|:--|:--|
| [`kai_probe/`](kai_probe/) | C++ benchmark that calls KleidiAI int4 matmul microkernels **directly**, with no llama.cpp in between |

## Why a direct probe exists

`scripts/02_batch_sweep.sh` infers kernel behaviour *through* llama.cpp. That
measurement carries every confound llama.cpp brings: threading policy, KV cache
growth, memory layout, graph scheduling, tokenizer overhead. If throughput bends
at some batch size, you cannot say from that alone whether a kernel changed.

`kai_probe` removes the middle. It packs matrices and calls the microkernel in a
loop, sweeping the row count **M** — which, in the decode regime, is exactly the
number of tokens being processed in one forward pass.

## The specific question it answers

KleidiAI microkernel names encode their tiling:

```
kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm
                                               └──────┘
                                               8 rows x 8 cols x 32 depth
```

The i8mm variant's natural block is **8 rows**, because `SMMLA` multiplies 2x2
integer tiles and the kernel widens that to an 8x8 macro-tile. Decode runs at
**M=1**. So the kernel is invoked on a tile where seven eighths of the row
capacity is unused.

KleidiAI exposes that geometry as a plain API call — `kai_get_mr_*()` and
`kai_get_m_step_*()` — so the *structural* claim needs no benchmark at all. The
probe reports those numbers, then measures the efficiency curve across M to
quantify what the waste costs in practice, and finds where i8mm overtakes the
dotprod path.

## Correctness first

Every configuration is validated against a scalar reference matmul before its
timing is recorded. A fast wrong answer is not a benchmark, and a microkernel
called with mis-packed inputs will happily produce one.

## Build

```bash
bash scripts/05_kai_probe.sh          # vendors KleidiAI, builds, runs, writes JSON
```

Requires an arm64 host. The script prints the microkernel variants actually
present in your KleidiAI checkout, so if a variant name has moved between
releases you can see the real list rather than guessing.

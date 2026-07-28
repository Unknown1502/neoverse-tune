# Reproducing

Three ways to run SpecArm, cheapest first. The first costs nothing and needs no
hardware.

- [A. GitHub Arm runner — the subject](#a-github-arm-runner--the-subject-free)
- [B. AWS t4g.small — the control](#b-aws-t4gsmall--the-control-free-trial)
- [C. Any other arm64 host](#c-any-other-arm64-host)
- [Reading the output](#reading-the-output)
- [Troubleshooting](#troubleshooting)

---

## A. GitHub Arm runner — the subject (free)

**Gets you:** Cobalt 100 / Neoverse N2 — Armv9 with i8mm, bf16, SVE2.
**Cost:** nothing, on a public repository.
**Time:** ~30–60 min.

1. Fork this repository, or push it to a **public** repo of your own.
2. **Actions** → **crux** → **Run workflow**.
3. Read the **job summary** when it finishes; download the `specarm-results-*`
   artifact for raw JSON.

Optional inputs: `reps` (default 10 — raise for tighter CIs), `sweep` (batch
sizes), `model_url` (any GGUF).

> The repo must be **public**. Arm-hosted runners are free for public
> repositories only; on a private repo the job will not pick up a runner.

**Expect `evidence_class: capability_only`.** Hosted runners normally restrict
`perf`, so kernel attribution degrades to a linked-kernel inventory. That is
handled and labelled, not a failure. For `execution`-class evidence use B or C.

## B. AWS t4g.small — the control (free trial)

**Gets you:** Graviton2 / Neoverse **N1** — **no i8mm, no SVE**.
**Cost:** free trial covers 750 h/month through 31 Dec 2026.
**Role:** the control group. See the difference-in-differences design in the
[README](../README.md#the-n1-box-is-the-control-group-not-a-consolation-prize).

```bash
# Launch: t4g.small, Ubuntu 24.04 LTS (arm64), 20 GB gp3, then:
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl python3 binutils linux-tools-generic

git clone <your-fork-url> specarm && cd specarm

bash scripts/00_env_report.sh          # expect: CONTROL banner, i8mm=false
bash scripts/fetch_model.sh
bash scripts/01_build_llama.sh --jobs 2
bash scripts/02_batch_sweep.sh --reps 10
bash scripts/03_kernel_attrib.sh
bash scripts/04_serve_bench.sh --conc 1,2,4 --requests 24
python3 tools/analyze.py
```

### t4g caveats that will bite you

| Issue | Why it matters | What to do |
|:--|:--|:--|
| **Burstable CPU credits** | Long benchmarks exhaust credits and get throttled mid-run, silently corrupting results | Keep runs short; `--jobs 2`; watch the `CPUCreditBalance` metric; the RSD gate will flag a throttled run as UNCERTAIN |
| **2 GB RAM** | Fine for the 0.5B Q4_0 default; a larger model will OOM during build or load | Add swap, or stay on the small model |
| **2 vCPU** | Building llama.cpp takes ~15–25 min | `--jobs 2`, be patient |
| **Virtualized PMU** | Fewer counters than metal | Expect `capability_only` or partial `execution` |

**The control is expected to show no kernel switch.** N1 has no i8mm path to
switch to. That is the result, not a failure — it is what licenses attributing
N2's behaviour to i8mm rather than to batching.

### Cleaning up

Stop or terminate the instance when done. The free trial covers instance hours,
not EBS storage beyond the free allowance.

## C. Any other arm64 host

Graviton3/4 (`c7g`/`c8g`), Ampere, Oracle A1, a Raspberry Pi 5 — anything arm64
Linux works. For **`execution`**-class kernel evidence you need `perf`:

```bash
cat /proc/sys/kernel/perf_event_paranoid    # need <= 2
sudo sysctl -w kernel.perf_event_paranoid=1 # if you own the box
```

Then the same sequence as B. Bare-metal or `.metal` instances expose the most PMU
counters; virtualized instances expose fewer.

### Useful environment variables

| Variable | Default | Purpose |
|:--|:--|:--|
| `SPECARM_REPS` | 10 | repetitions per measurement |
| `SPECARM_SWEEP` | `1,2,3,4,6,8,12,16,24,32` | batch sizes |
| `SPECARM_THREADS` | `nproc` | inference threads |
| `SPECARM_MODEL_URL` | Qwen2.5-0.5B Q4_0 | any GGUF URL |
| `SPECARM_CONC` | `1,2,4,8,16` | serving concurrency levels |
| `SPECARM_LLAMA_REF` | `master` | llama.cpp ref to build |
| `SPECARM_RESULTS` | `./results` | output directory |

## Reading the output

`results/report_<stamp>.md` is the human-readable result. Read it in this order:

1. **Crux role** — `control` means the kernel result there is not evidence.
2. **Evidence class** — `capability_only` means no mechanism proof was captured.
3. **Verdict column** — only VERIFIED rows are defensible claims.
4. **Mechanism section** — did the hot ISA change with batch size?

The knee is a *candidate*, not a conclusion. Throughput rises with batch size on
any core; only the symbol evidence attributes it.

## Troubleshooting

| Symptom | Cause | Fix |
|:--|:--|:--|
| `bad interpreter: ...^M` | CRLF line endings | `git config core.autocrlf input`, re-clone. `.gitattributes` should prevent this. |
| `SpecArm measures Arm kernel selection; this host is 'x86_64'` | Wrong architecture | Run on arm64. The x86 test suites still work: `python3 tools/test_skeptic.py` |
| `No kai_* symbols in the KleidiAI build` | Flag accepted but inert, or binary stripped | Check `results/build_kleidi_configure.log`; do not report the A/B until explained |
| `evidence_class: capability_only` | `perf` restricted | Expected in CI. Use a host where you control `perf_event_paranoid`. |
| Everything UNCERTAIN with "host too noisy" | Throttling or noisy neighbours | Raise `SPECARM_REPS`, use a non-burstable instance, re-run when idle |
| `server at ... never became healthy` | Model too large for RAM, or server crashed | Check `results/serve_*.server.log` |
| Serving run exits 1 | Some requests failed | Check `errors` in the serve JSON; lower concurrency or raise `-np` |
| cmake configure fails | Missing deps | `build-essential cmake git curl python3 binutils` |

## Related

- [flows.md](flows.md) — what happens at each step
- [metrics.md](metrics.md) — what the numbers mean
- [methodology.md](methodology.md) — thresholds and falsification criteria

# Reproducing this result

Written so a stranger can either confirm the numbers or prove them wrong. Both
outcomes are useful; the second is more useful.

- [Route A — free Arm64 cloud, one click](#route-a--free-arm64-cloud-one-click)
- [Route B — your own Arm64 instance](#route-b--your-own-arm64-instance)
- [Route C — any laptop, no Arm hardware](#route-c--any-laptop-no-arm-hardware)
- [Route D — five minutes, just show me](#route-d--five-minutes-just-show-me)
- [What you should see](#what-you-should-see)
- [How to tell a real difference from a broken run](#how-to-tell-a-real-difference-from-a-broken-run)
- [Applying it to your own deployment](#applying-it-to-your-own-deployment)
- [Troubleshooting](#troubleshooting)
- [How to prove this wrong](#how-to-prove-this-wrong)

---

## Route A — free Arm64 cloud, one click

**Cost: $0. Time: 47 minutes, unattended. Hardware required: none.**

1. Fork https://github.com/Unknown1502/neoverse-tune
2. **Actions** → **bench** → **Run workflow**
3. Leave `repeats: 5`, `turns: 4`

Runs on `ubuntu-24.04-arm` — Microsoft Cobalt 100, Arm Neoverse N2, 4 vCPU — free
on public repositories. The adjudicated table renders in the job summary; raw
data uploads as an artifact with 30-day retention.

This is the route that matters. Every number published here came from it, and a
benchmark you can re-run for nothing on someone else's hardware is the only kind
worth believing.

---

## Route B — your own Arm64 instance

**AWS Graviton, Azure Cobalt, Google Axion, Ampere. Tested on Ubuntu 24.04 aarch64.**

⚠️ **Do not use a burstable instance** (`t4g`, `t3`). They throttle mid-benchmark
and silently corrupt results — the harness detects the shape and warns, but the
right fix is a compute instance. `c6g.large`, `c7g.large` or equivalent, a few
cents an hour.

```bash
sudo apt-get update -qq
sudo apt-get install -y build-essential cmake git curl python3

git clone https://github.com/Unknown1502/neoverse-tune.git
cd neoverse-tune

# 1. Identify the core. Records MIDR, ISA features, and the int8 matmul path.
bash scripts/00_env_report.sh

# 2. Build llama-server. Records the resolved upstream SHA.
bash scripts/01_build_llama.sh

# 3. Fetch the model. Pinned by SHA-256; refuses to continue on a mismatch.
bash scripts/fetch_model.sh

# 4. The matrix: 6 configs x 5 repeats, fresh server every repeat.
python3 tools/run_matrix.py \
  --server "$(python3 -c "import json;print(json.load(open('results/build.latest.json'))['server'])")" \
  --model  "$(cat results/model.path)" \
  --repeats 5

# 5. Adjudicated table with confidence intervals.
python3 tools/analyze_agent.py

# 6. Mechanism evidence and the prefill cost model.
python3 tools/parse_slot_log.py
```

**Match the thread count if you intend to compare across machines.** Add
`--threads N` to step 4. Comparing a 4-vCPU host against an 8-thread host
confounds architecture with core count — a mistake this project made and
documents rather than hides.

Terminate the instance when finished.

---

## Route C — any laptop, no Arm hardware

The mechanism is scheduler behaviour and reproduces on x86. You will get the
same similarity values and the same token counts; only wall-clock differs.

```bash
git clone https://github.com/Unknown1502/neoverse-tune.git
cd neoverse-tune

python3 tools/run_matrix.py --server <llama-server> --model <model.gguf> --repeats 5
python3 tools/analyze_agent.py
python3 tools/parse_slot_log.py
```

Bring your own `llama-server` and any instruct GGUF. To compare against the
published x86 column, use Qwen2.5-1.5B-Instruct **Q4_K_M** — a different
quantization changes the token counts and the comparison stops meaning anything.

---

## Route D — five minutes, just show me

```bash
python3 tools/demo.py --server <llama-server> --model <model.gguf>
```

Two configurations, once each, with live per-request bars and a final
comparison. Explicitly `n=1` and labelled as such on screen. It is the same
experiment as the matrix at a smaller scale, not a friendlier one.

---

## What you should see

### Mechanism — deterministic, identical on every machine tested

These come from `llama-server`'s own log. They are set by the scheduler, not by
your CPU, so they should match **exactly** regardless of hardware.

| config | median LCP similarity | median tokens recomputed | >100 tok |
|:--|--:|--:|--:|
| `solo_baseline` | **0.955** | **35** | 25% |
| `4tenant_default` | **0.716** | **220** | 81% |
| `8tenant_default` | **0.716** | **220** | 78% |
| `4tenant_sim09` | **0.955** | **36** | 25% |
| `8tenant_sim09` | **0.905** | **56** | 31% |
| `8tenant_sim09_np8` | **0.955** | **36** | 25% |

**If these differ, something meaningful is different** — a different model, a
different quantization, a different preamble, or a llama.cpp release that
changed the selection heuristic. Any of those is worth reporting.

### Latency — machine-dependent, will not match

| config | Neoverse N2, 4 vCPU | x86, 8 threads |
|:--|--:|--:|
| `solo_baseline` | 385.9 ± 2.5 ms | 503.2 ± 43.3 ms |
| `4tenant_default` | 4203.7 ± 4.8 ms | 2312.7 ± 251.9 ms |
| `4tenant_sim09` | 399.0 ± 1.0 ms | 458.2 ± 26.1 ms |

Your absolute numbers will differ. What should hold is the **shape**: a large
regression at the default, near-baseline recovery at `0.9`.

Predict your own host from the cost model:

```
ms lost per mis-routed request  ≈  220 / (your prefill tok/s)
```

`parse_slot_log.py` prints your prefill rate. On the two hosts measured this
predicted TTFT within 3%.

### Verdicts

| Question | Expected |
|:--|:--|
| Does adding tenants break prefix caching? | 🔴 REGRESSION CONFIRMED |
| Does raising the threshold fix it? | ✅ VERIFIED |
| Does the fix hold when tenants exceed slots? | ✅ VERIFIED |
| Is multi-tenant back at single-tenant parity? | ⚠️ UNCERTAIN or ❌ REJECTED |
| Do extra slots help once the threshold is right? | ⚠️ UNCERTAIN |

The last two are *supposed* to be unimpressive. Parity came back UNCERTAIN on a
noisy laptop and REJECTED on a quiet cloud instance, because the quieter
measurement could resolve a real residual ~4% gap. Extra slots returning
UNCERTAIN is what rules out slot capacity as the explanation.

**If every row comes back VERIFIED, be suspicious of the harness, not pleased.**

---

## How to tell a real difference from a broken run

`tools/skeptic.py` will not certify a result it cannot support. Read the verdict
reason, not just the number.

| You see | It means | Do |
|:--|:--|:--|
| `no valid samples` | Every run failed | Read `results/server_*.log` |
| `only N reps (need 5)` | Runs crashed, or `--repeats` too low | Use `--repeats 5` or higher |
| `host too noisy (rsd 14.2% > 10%)` | Machine was busy or throttling | Close everything; avoid burstable instances |
| `+3.1% is below the 5.0% floor` | Real but too small to be worth claiming | Nothing — that is the correct answer |
| `+9.6% but 95% CIs overlap` | Cannot distinguish these | More repeats, or accept it |
| `VERIFIED` | ≥5 reps, RSD ≤10%, effect ≥5%, disjoint CIs | Believe it |

A **non-empty server log that yields 0 requests** from `parse_slot_log.py` means
llama.cpp's log format has drifted and the parser needs updating. That is a
parser outage, not a null result. `tools/test_parse_slot_log.py` guards it.

---

## Applying it to your own deployment

The threshold that works depends on your prompt shape, not on this repository.

```bash
# authoritative: measures each candidate against a fresh server
python3 tools/tune_similarity.py --sweep --server <llama-server> --model <model.gguf>

# fast estimate — explicitly a heuristic, has disagreed with measurement before
python3 tools/tune_similarity.py --url http://127.0.0.1:8080
```

Then:

```bash
llama-server -m model.gguf -np 4 --slot-prompt-similarity <value>
```

**The rule of thumb, from the sweep:** it is a cliff, not a gradient. Pick any
value comfortably **above your inter-tenant similarity** — the number
`parse_slot_log.py` reports as `sim med` when tenants are colliding. Below it,
nothing works; above it, everything does. On the workload here inter-tenant
similarity is 0.716 and every threshold from 0.8 up performs identically.

**When you do not need this:** single-tenant deployments, and chat workloads with
short system prompts. The `0.10` default is reasonable there. This is an agent
problem, and specifically a *large shared preamble* problem.

---

## Troubleshooting

| Symptom | Cause | Fix |
|:--|:--|:--|
| `couldn't bind HTTP server socket` | Port in use | `--port 8100` |
| `server never became healthy` | Model load exceeded 240 s | Smaller model, or more RAM |
| `SHA-256 MISMATCH — removed <path>` | Wrong or corrupt model | Re-run; the artifact was deleted deliberately |
| `bad interpreter: ^M` | CRLF line endings | `git config core.autocrlf input` and re-clone |
| `this host is 'x86_64'` from `00_env_report.sh` | Arm-only script | Expected on x86. Use Route C. |
| RSD above 10% on every config | Noisy or throttling host | Not burstable; close other work |
| `counts via char_heuristic` | Tokenizer endpoints unavailable | Token counts are approximate — say so if you publish |
| Sweep says "measured best 0.8" | n=3, no intervals | A ranking, not a result. Anything above the cliff is fine. |

---

## How to prove this wrong

Stated in advance in [methodology.md §6](methodology.md#6-what-would-falsify-this),
and worth restating here because falsification is the most valuable
contribution anyone can make:

1. **Tokens-recomputed identical at 1 tenant and 4** → no scattering, mechanism wrong.
2. **Raising the threshold does not reduce tokens-recomputed** → threshold is not the lever.
3. ~~**The effect vanishes on Arm**~~ → tested on Neoverse N2. It did not vanish; it more than doubled.

Additional things that would change the conclusion, none of which have been done:

- **Concurrent load.** Every measurement here is strictly sequential. A concurrent generator might show a different magnitude, or interactions this design cannot see. This is the largest open gap.
- **A preamble-size sweep.** The central claim is *about* preamble size and exactly one value (550 tokens) was tested.
- **Other models and quantizations.** One model, one quantization.
- **A host where the cliff is not at inter-tenant similarity.** That would falsify the mechanism while leaving the effect intact — the most interesting possible outcome.

If you run any of these, the result is worth an issue whichever way it lands.

# Architecture

Three levels of zoom: what the system talks to, what the modules are, and how
the two load-bearing pieces actually work inside.

Every diagram here is generated from the code as it exists, not from a design
that preceded it. Where a diagram and the code disagree, the code is right and
this file is a bug.

- [Level 1 — System context](#level-1--system-context)
- [Level 2 — Modules and data flow](#level-2--modules-and-data-flow)
- [Level 3 — Inside the two pieces that matter](#level-3--inside-the-two-pieces-that-matter)
- [Process model](#process-model)
- [Data model](#data-model)
- [Deployment](#deployment)
- [Design decisions](#design-decisions-and-what-they-cost)

---

## Level 1 — System context

The harness owns no inference code. It spawns an unmodified upstream
`llama-server`, drives it over HTTP, and reads its log. That boundary is
deliberate: a finding about upstream behaviour is worthless if the measurement
tool has patched the thing it is measuring.

```mermaid
flowchart LR
    dev(["Developer<br/>or judge"])
    ci(["GitHub Actions<br/>ubuntu-24.04-arm"])

    subgraph harness["Measurement harness — Python 3, stdlib only"]
        tools["tools/*.py"]
    end

    srv["llama-server<br/><i>unmodified upstream binary</i>"]
    gguf[("GGUF model")]
    logs[("server logs<br/>results/server_*.log")]
    art[("results/*.json<br/>results/agent_report.md")]

    dev -->|"CLI"| tools
    ci -->|"workflow_dispatch"| tools
    tools -->|"spawn / signal"| srv
    tools <-->|"HTTP"| srv
    srv --> gguf
    srv -->|"stdout + stderr"| logs
    logs -->|"parse"| tools
    tools --> art

    style srv fill:#2d3748,color:#fff
    style harness fill:#1a365d,color:#fff
```

**The HTTP surface used, and why each endpoint is there:**

| Endpoint | Used for | Why not something simpler |
|:--|:--|:--|
| `GET /health` | Gate before measuring | A server that is listening is not yet a server that has loaded a model |
| `POST /apply-template` | Apply the chat template | Token counts must include role markers and template scaffolding |
| `POST /tokenize` | Exact token count | `len(text)//4` is a guess; this is the tokenizer that will actually run |
| `POST /v1/chat/completions` | The measurement, `stream: true` | Streaming is the only way to observe *first* token separately from last |

---

## Level 2 — Modules and data flow

Seven modules. Each writes a file the next one reads, so any stage can be
re-run without repeating the 47 minutes above it.

```mermaid
flowchart TB
    subgraph probe["Sizing — run once, before building anything"]
        pp["probe_prefill.py<br/><small>is there an opportunity at all?</small>"]
    end

    subgraph measure["Measurement"]
        rm["run_matrix.py<br/><small>owns server lifecycle<br/>6 configs x 5 repeats</small>"]
        ba["bench_agent.py<br/><small>one run, exact token counts</small>"]
        rm -->|"import, in-process"| ba
    end

    subgraph adjudicate["Adjudication"]
        aa["analyze_agent.py<br/><small>aggregate + compare</small>"]
        sk["skeptic.py<br/><small>VERIFIED / UNCERTAIN / REJECTED</small>"]
        aa -->|"from skeptic import"| sk
    end

    subgraph mech["Mechanism"]
        ps["parse_slot_log.py<br/><small>what the server admitted</small>"]
    end

    subgraph tune["Recommendation"]
        ts["tune_similarity.py<br/><small>--sweep: measured<br/>default: heuristic</small>"]
    end

    mx[("matrix.json")]
    lg[("server_*.log")]
    rep[("agent_report.md<br/>agent_analysis.latest.json")]
    ev[("slot_evidence.json")]

    pp -->|"verdict:<br/>REUSE_WORKS"| rm
    rm --> mx
    rm --> lg
    mx --> aa
    aa --> rep
    lg --> ps
    ps --> ev
    mx --> ts

    style sk fill:#22543d,color:#fff
    style rep fill:#742a2a,color:#fff
    style ev fill:#742a2a,color:#fff
```

| Module | Reads | Writes | One-line job |
|:--|:--|:--|:--|
| `probe_prefill.py` | — | `prefill_default.json` | Is prefix reuse already working? (It was — this killed hypothesis 2) |
| `run_matrix.py` | — | `matrix.json`, `server_*.log` | 30 runs, **fresh server each** |
| `bench_agent.py` | — | in-process dict | One run: per-turn TTFT + exact token counts |
| `analyze_agent.py` | `matrix.json` | `agent_report.md`, `agent_analysis.latest.json` | Aggregate to CIs, run the 6 pre-declared comparisons |
| `skeptic.py` | — | — | The single definition of VERIFIED |
| `parse_slot_log.py` | `server_*.log` | `slot_evidence.json` | Similarity chosen, **tokens recomputed** |
| `tune_similarity.py` | `matrix.json` | stdout | Right threshold for a given workload |

**The dependency that matters:** `skeptic.py` imports nothing from this project.
It is a leaf. Nothing in it knows what is being measured, so it cannot be
tempted into a verdict by context.

---

## Level 3 — Inside the two pieces that matter

### `skeptic.verdict` — the gate order is the design

The order is not arbitrary. Each gate answers a question that must be settled
before the next one is meaningful, and the first four exist to stop a *failed
measurement* being published as a *negative result*.

```mermaid
flowchart TD
    start(["verdict(base, cand)"]) --> g1{"any samples<br/>at all?"}
    g1 -->|"no"| u1["UNCERTAIN<br/><i>measurement failed,<br/>not a null result</i>"]
    g1 -->|"yes"| g2{"n ≥ MIN_REPS<br/>(5)?"}
    g2 -->|"no"| u2["UNCERTAIN<br/><i>too few reps</i>"]
    g2 -->|"yes"| g3{"RSD ≤ MAX_RSD<br/>(10%)?"}
    g3 -->|"no"| u3["UNCERTAIN<br/><i>host too noisy</i>"]
    g3 -->|"yes"| g4{"baseline<br/>non-zero?"}
    g4 -->|"no"| u4["UNCERTAIN<br/><i>baseline measured zero</i>"]
    g4 -->|"yes"| calc["effect = (cand-base)/base"]

    calc --> g5{"effect > 0?"}
    g5 -->|"no"| rej["REJECTED<br/><i>no improvement</i>"]
    g5 -->|"yes"| g6{"95% CIs<br/>disjoint?"}
    g6 -->|"overlap"| u5["UNCERTAIN<br/><i>CIs overlap</i>"]
    g6 -->|"disjoint"| g7{"effect ≥<br/>MIN_EFFECT (5%)?"}
    g7 -->|"no"| u6["UNCERTAIN<br/><i>below the floor</i>"]
    g7 -->|"yes"| ver["✅ VERIFIED"]

    style ver fill:#22543d,color:#fff
    style rej fill:#742a2a,color:#fff
    style u1 fill:#744210,color:#fff
    style u2 fill:#744210,color:#fff
    style u3 fill:#744210,color:#fff
    style u4 fill:#744210,color:#fff
    style u5 fill:#744210,color:#fff
    style u6 fill:#744210,color:#fff
```

**The distinction that most harnesses get wrong:** four different failure paths
land on UNCERTAIN, and only one path — a measured non-improvement — lands on
REJECTED. "We measured nothing" and "we measured no gain" are different claims,
and conflating them lets a broken run masquerade as evidence.

`Stat` is the only thing `verdict` sees:

```mermaid
classDiagram
    class Stat {
        +list samples
        +int n
        +float mean
        +float median
        +float sd
        +float hw
        +lo() float
        +hi() float
        +rsd() float
        +as_dict() dict
    }
    class verdict {
        <<function>>
        +verdict(base, cand) tuple
    }
    class t_crit {
        <<function>>
        +t_crit(n) float
    }
    Stat ..> t_crit : half-width uses<br/>small-sample t
    verdict ..> Stat : reads only
```

`hw = t_crit(n) * sd / sqrt(n)` — and `t_crit` uses a **small-sample t table**,
not 1.96. At n=5 the normal approximation understates the interval by roughly
30%, which is exactly enough to manufacture VERIFIED verdicts out of thin data.

### `Tokenizer` — a fallback chain that records which rung it used

```mermaid
flowchart LR
    p(["__init__"]) --> t1{"POST /apply-template<br/>returns a prompt?"}
    t1 -->|"yes"| m1["method =<br/>apply_template+tokenize"]
    t1 -->|"no"| t2{"POST /tokenize<br/>returns tokens?"}
    t2 -->|"yes"| m2["method =<br/>tokenize_concat"]
    t2 -->|"no"| m3["method =<br/>char_heuristic"]

    style m1 fill:#22543d,color:#fff
    style m2 fill:#744210,color:#fff
    style m3 fill:#742a2a,color:#fff
```

`method` travels into every result as `token_count_method`. A count from
`char_heuristic` is a *different claim* from one produced by the server's own
tokenizer, and a report that does not say which it used is not reproducible.

---

## Process model

This is the part that makes the confidence intervals mean anything.

```mermaid
flowchart TB
    subgraph loop["for each of 6 configs × 5 repeats = 30 iterations"]
        direction TB
        s1["spawn llama-server<br/>new process group / session"]
        s2["wait_healthy — up to 240 s"]
        s3["bench_agent.run<br/>N tenants × 4 turns, sequential"]
        s4["terminate<br/>CTRL_BREAK_EVENT (nt) or SIGTERM<br/>20 s grace, then kill"]
        s5["sleep 2 s — let the port clear"]
        s1 --> s2 --> s3 --> s4 --> s5
    end
    s5 -.->|"next repeat"| s1

    style s1 fill:#1a365d,color:#fff
    style s4 fill:#742a2a,color:#fff
```

| Choice | Reason | Cost |
|:--|:--|:--|
| **Fresh server every repeat** | A warm server's slots still hold the previous run's KV, so repeat 2 measures *carry-over*, not the configuration | 30 model loads ≈ most of the 47 minutes |
| **New process group / session** | So the whole tree can be signalled on Windows and POSIX alike | A little platform-specific code |
| **Sequential requests** | Removes queueing and thread contention as competing explanations | Deletes the realistic concurrent case — a real limitation, see [methodology](methodology.md) |
| **`sleep 2` after terminate** | The port outlives the process; the next spawn fails without it | 60 s across the matrix |

---

## Data model

Every artifact carries a `schema` string. When a schema changes, old results
stay readable instead of silently mis-parsed.

```mermaid
erDiagram
    MATRIX ||--o{ RUN : "configs[name][]"
    RUN ||--o{ ROW : "rows[]"
    MATRIX ||--|| ANALYSIS : "adjudicated into"
    ANALYSIS ||--o{ COMPARISON : "comparisons{}"
    LOG ||--|| EVIDENCE : "parsed into"

    MATRIX {
        string schema "specarm.matrix/1"
        string server
        string model
        int ctx
        int threads
        int turns
        int repeats
        float elapsed_s
    }
    RUN {
        string schema "specarm.agentbench/1"
        string label
        int agents
        string token_count_method
        int preamble_tokens
        int final_prompt_tokens
        float warm_median_ms "THE independent sample"
        float cold_median_ms
    }
    ROW {
        int seq
        int round "1 = cold"
        int agent
        int prompt_tokens
        float ttft_ms
        float total_ms
        int out_tokens
    }
    ANALYSIS {
        string schema "specarm.agent_analysis/1"
        float regression_x
        float improvement_x
    }
    COMPARISON {
        string question
        string expectation
        float ratio
        string verdict
        string reason
    }
    EVIDENCE {
        float median_similarity
        int median_tokens_recomputed
        float pct_over_100_tokens
        int lru_fallbacks
    }
```

**The single most important field is `RUN.warm_median_ms`.** It is the
independent sample. `n` equals the repeat count — never the request count,
because requests inside one run share a server, a cache state and a schedule.
Pooling 96 requests instead of 5 medians would shrink the intervals to nothing
and manufacture significance.

`ROW.round == 1` is cold for every tenant by definition, so it is separated out
rather than averaged in.

---

## Deployment

```mermaid
flowchart LR
    subgraph local["Local — development"]
        w["Windows x86, 8 threads<br/><i>the x86 column</i>"]
    end
    subgraph gha["GitHub Actions — free on public repos"]
        j1["skeptic gate<br/>ubuntu-latest<br/><small>3 test suites</small>"]
        j2["bench<br/>ubuntu-24.04-arm<br/><small>Cobalt 100 / Neoverse N2, 4 vCPU</small>"]
        j1 -->|"needs:"| j2
    end
    subgraph aws["AWS free trial to 2026-12-31"]
        g2["t4g.small<br/>Graviton2 / Neoverse N1<br/><small>no i8mm — the control</small>"]
    end

    w -.->|"push"| j1
    j2 --> sum["job summary + artifact<br/>30 day retention"]

    style j2 fill:#22543d,color:#fff
    style g2 fill:#744210,color:#fff
    style w fill:#1a365d,color:#fff
```

**Status:** the N2 runner has produced a full 30-run matrix
(`results/arm-neoverse-n2/`, 2026-08-10). The N1 host is the outstanding
experiment — it is the only way to separate microarchitecture from core count in
the cost comparison.

**The gate is the point.** If the adjudicator is broken there is no reason to
spend an Arm runner on measurements it would misjudge, so `skeptic` runs the
three test suites on cheap x86 first and `bench` declares `needs: skeptic`.

Why two Arm targets: **N1 has no i8mm and N2 does.** Prefill throughput
therefore differs, so the *cost of a slot miss* differs, so the right threshold
may differ. One core cannot show that; two can.

---

## Design decisions, and what they cost

| Decision | Bought | Paid |
|:--|:--|:--|
| stdlib only — no numpy, requests, pytest | Runs on any Python 3.8+ with zero install; a judge cannot hit a dependency error | Hand-rolled t-table; tests are `__main__` scripts, so no collection or coverage |
| Unmodified upstream `llama-server` | The finding is about upstream, provably | No ability to instrument internals — mechanism comes from log text, which is brittle |
| Log parsing for mechanism evidence | The server *itself* states what it did; immune to CPU noise and thermal state | Two regexes against unstructured text, **no fixtures committed** — silent zero-match on format drift |
| `skeptic.py` as a leaf module | Cannot be influenced by what it is judging | Latency must be inverted to a rate by the caller, since `verdict` is higher-is-better |
| Pre-registered thresholds | Cannot be tuned to fit a result | Some real effects land on UNCERTAIN and stay there |
| Per-repeat medians as samples | Intervals that survive scrutiny | n=5 instead of n=96, so only large effects can clear the bar |

### Known architectural weaknesses

Stated here rather than discovered by a reviewer:

1. **`parse_slot_log.py` has no fixtures.** An upstream log-format change makes it return zero matches, silently. It should ship a captured log and a test.
2. **No packaging.** No `pyproject.toml`, so there is no `pip install`, no entry points, and no version.
3. **50-minute minimum feedback loop.** No smoke mode, so every change to the measurement path costs most of an hour to validate.
4. **The analytic half of `tune_similarity.py` models an inferred formula.** It is labelled a heuristic because it has already disagreed with measurement once. `--sweep` is authoritative and slower.
5. **Hardcoded default port.** A busy port fails the run rather than choosing another.

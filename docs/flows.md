# Flows

Every path through the system, end to end. Start with
[the mechanism](#1-the-mechanism--how-a-slot-gets-chosen-wrong) — it is the only
diagram here that explains *why the project exists*.

1. [The mechanism — how a slot gets chosen wrong](#1-the-mechanism--how-a-slot-gets-chosen-wrong)
2. [Slot lifetime across turns](#2-slot-lifetime-across-turns)
3. [One measurement run](#3-one-measurement-run)
4. [One turn, in detail](#4-one-turn-in-detail)
5. [The full matrix](#5-the-full-matrix)
6. [Adjudication](#6-adjudication)
7. [Mechanism extraction from logs](#7-mechanism-extraction-from-logs)
8. [Tuning for your own workload](#8-tuning-for-your-own-workload)
9. [CI on Arm](#9-ci-on-arm)
10. [How three hypotheses died](#10-how-three-hypotheses-died)

---

## 1. The mechanism — how a slot gets chosen wrong

`llama-server` holds conversations in **slots**, each keeping its KV cache so
already-processed text is not reprocessed. To route an incoming request it scores
every slot by longest-common-prefix similarity and accepts any slot clearing
`--slot-prompt-similarity`.

Similarity is approximately `shared_prefix / total_prompt`. **The default
threshold is 0.10.**

```mermaid
flowchart TB
    req["Tenant B, turn 3<br/>836-token prompt<br/><small>550 of them are the shared preamble</small>"]

    req --> score["score every slot by<br/>longest common prefix"]

    score --> s0["slot 0 — holds tenant A<br/>similarity 0.716"]
    score --> s1["slot 1 — holds tenant B<br/>similarity 0.955 ✓ correct"]
    score --> s2["slot 2 — holds tenant C<br/>similarity 0.702"]
    score --> s3["slot 3 — holds tenant D<br/>similarity 0.698"]

    s0 --> gate{"similarity ><br/>threshold?"}
    s1 --> gate
    s2 --> gate
    s3 --> gate

    gate -->|"threshold 0.10<br/><b>ALL FOUR QUALIFY</b>"| bad["takes whichever it finds<br/>→ lands on a foreign slot<br/><br/><b>220 tokens recomputed</b><br/>2313 ms TTFT"]
    gate -->|"threshold 0.90<br/><b>only slot 1 qualifies</b>"| good["lands on its own slot<br/><br/><b>36 tokens recomputed</b><br/>458 ms TTFT"]

    style bad fill:#742a2a,color:#fff
    style good fill:#22543d,color:#fff
    style s1 fill:#1a365d,color:#fff
```

Because every tenant carries the same 550-token preamble, **any two tenants are
already 0.6–0.9 similar to each other.** Against 0.10, every slot looks like a
valid match for every tenant.

The server states the decision itself, which is why this is evidence rather than
inference:

```
slot get_availabl: id  0 | task 12 | selected slot by LCP similarity, f_sim_best = 0.716 (> 0.100 thold)
slot print_timing: id  0 | prompt eval time = 1893.44 ms / 220 tokens
```

### The corollary that makes this non-obvious

```mermaid
flowchart LR
    a["small shared preamble<br/><small>ordinary chat</small>"] --> b["inter-tenant similarity<br/><b>low</b>"] --> c["0.10 separates<br/>tenants fine ✅"]
    d["large shared preamble<br/><small>agent + tool schemas</small>"] --> e["inter-tenant similarity<br/><b>0.6 – 0.9</b>"] --> f["0.10 separates<br/>nothing ❌"]

    style c fill:#22543d,color:#fff
    style f fill:#742a2a,color:#fff
```

**The bigger your system prompt, the worse the default behaves.** That is
backwards from intuition, which is why the default is reasonable for chat and
wrong for agents — and why nobody had reported it.

It is also invisible to ordinary benchmarking: a single-conversation benchmark
never has a second tenant to be confused with.

---

## 2. Slot lifetime across turns

What actually happens to one tenant's cache over four turns.

```mermaid
stateDiagram-v2
    [*] --> Cold: turn 1 — nothing cached anywhere
    Cold --> Owned: full prefill, tenant now holds a slot

    Owned --> Owned: turn N routed to its OWN slot<br/>~36 tokens recomputed<br/>458 ms
    Owned --> Evicted: turn N routed to a FOREIGN slot<br/>because every slot cleared 0.10

    Evicted --> Owned: full re-prefill<br/>~220 tokens<br/>2313 ms<br/><i>and it evicts whoever was there</i>

    note right of Evicted
        This is the bug.
        Two tenants take turns
        evicting each other and
        both pay full prefill,
        forever.
    end note
```

The self-sustaining part matters: tenant B displacing tenant A does not merely
cost B a re-prefill — it destroys A's cache, so A pays too on its next turn. The
regression does not decay. It is a stable thrashing state.

---

## 3. One measurement run

`bench_agent.run()` — one run against an already-running server. Repeats and
server lifecycle belong to `run_matrix.py`, because a repeat reusing a warm
server is not an independent sample.

```mermaid
sequenceDiagram
    autonumber
    participant R as run_matrix
    participant B as bench_agent
    participant T as Tokenizer
    participant S as llama-server

    R->>S: spawn(server, model, port, ctx, threads, extra)
    R->>S: GET /health  (poll, up to 240 s)
    S-->>R: 200 OK

    R->>B: run(url, agents, turns=4, max_tokens=48)
    B->>T: __init__ — probe token-count path
    T->>S: POST /apply-template {messages:[x,y]}
    S-->>T: {prompt: "..."}
    Note over T: method = apply_template+tokenize

    B->>B: seed N conversations<br/>system + flavor + "Understood."
    B->>T: count([system]) → preamble_tokens

    loop round 1..turns
        loop tenant 0..N-1
            B->>B: append tenant_turn(tenant, round)
            B->>T: count(conversation) → prompt_tokens
            T->>S: POST /apply-template → POST /tokenize
            B->>S: POST /v1/chat/completions<br/>stream, temp 0, cache_prompt
            S-->>B: SSE: first delta → <b>ttft_ms</b>
            S-->>B: SSE: ... → [DONE] → total_ms, out_tokens
            B->>B: append assistant reply (truncated 300 ch)
            B->>B: rows.append({seq, round, agent, ...})
        end
    end

    B->>B: warm = round>1 · cold = round==1
    B-->>R: {warm_median_ms, preamble_tokens, rows, ...}
    R->>S: terminate — CTRL_BREAK / SIGTERM, 20 s grace
    R->>R: sleep 2 s — let the port clear
```

**Round-robin, not per-tenant:** the outer loop is the round and the inner loop
is the tenant. All tenants advance one turn before anyone advances two — which is
what creates the interleaving that triggers the bug. Draining one tenant at a
time would never reproduce it.

**Round 1 is cold for every tenant by definition**, so it is separated rather
than averaged in. `warm_median_ms` — the median of rounds 2+ — is the single
number that becomes an independent sample.

---

## 4. One turn, in detail

Where the two numbers come from, and why streaming is required.

```mermaid
sequenceDiagram
    autonumber
    participant B as bench_agent
    participant S as llama-server

    Note over B: t0 = perf_counter()
    B->>S: POST /v1/chat/completions<br/>{stream:true, temperature:0, cache_prompt:true}

    rect rgb(116, 42, 42)
        Note over S: score slots by LCP similarity<br/>← THE DECISION UNDER TEST
        Note over S: prefill the non-cached suffix<br/>← the cost being measured
    end

    S-->>B: data: {delta:{content:"Sure"}}
    Note over B: first token → ttft_ms = (now - t0)*1000

    loop remaining tokens
        S-->>B: data: {delta:{content:"..."}}
        Note over B: out_tokens += 1
    end
    S-->>B: data: [DONE]
    Note over B: total_ms = (now - t0)*1000
```

| Setting | Value | Why |
|:--|:--|:--|
| `stream` | `true` | Without it, first-token time is unobservable — you only see the last one |
| `temperature` | `0.0` | Deterministic output, so token sequences are identical across repeats and only wall-clock varies |
| `cache_prompt` | `true` | Caching must be *on*, or there is no cache to mis-route |
| `max_tokens` | `48` | Enough to measure TTFT, short enough that decode does not dominate the run |

Prefill token counts come back byte-identical across all five repeats. That is
expected, not suspicious — temperature is 0 and the workload is fixed. It also
means the token metric carries less independent information than the timing
metric, which is stated in [methodology](methodology.md).

---

## 5. The full matrix

Six configurations, each changing **one** thing from the baseline.

```mermaid
flowchart LR
    base["solo_baseline<br/>1 tenant · -np 4 · sim 0.1"]

    base -->|"+3 tenants"| c2["4tenant_default<br/>4 · -np 4 · sim 0.1"]
    base -->|"+7 tenants"| c3["8tenant_default<br/>8 · -np 4 · sim 0.1"]
    c2 -->|"sim → 0.9"| c4["4tenant_sim09<br/>4 · -np 4 · sim 0.9"]
    c3 -->|"sim → 0.9"| c5["8tenant_sim09<br/>8 · -np 4 · sim 0.9"]
    c5 -->|"-np → 8"| c6["8tenant_sim09_np8<br/>8 · -np 8 · sim 0.9"]

    style base fill:#1a365d,color:#fff
    style c2 fill:#742a2a,color:#fff
    style c3 fill:#742a2a,color:#fff
    style c4 fill:#22543d,color:#fff
    style c5 fill:#22543d,color:#fff
    style c6 fill:#22543d,color:#fff
```

Each edge is one variable. `4tenant_default → 4tenant_sim09` changes **only** the
threshold, which is what licenses the causal claim; `8tenant_sim09 →
8tenant_sim09_np8` changes only slot count, which is what ruled out capacity as
the explanation.

```mermaid
flowchart TB
    start(["run_matrix.py --repeats 5"]) --> outer{"for each of<br/>6 configs"}
    outer --> inner{"for each of<br/>5 repeats"}
    inner --> spawn["spawn fresh server<br/>log → server_NAME_rN.log"]
    spawn --> health{"healthy<br/>within 240 s?"}
    health -->|"no"| skip["record nothing<br/>continue — a missing<br/>sample is not a zero"]
    health -->|"yes"| run["bench_agent.run(...)"]
    run --> collect["collected[name].append(result)"]
    skip --> kill
    collect --> kill["terminate + sleep 2"]
    kill -->|"next repeat"| inner
    inner -->|"next config"| outer
    outer --> write["write matrix.json<br/>schema specarm.matrix/1"]
    write --> done(["next: analyze_agent.py"])

    style skip fill:#744210,color:#fff
    style done fill:#22543d,color:#fff
```

A server that never becomes healthy contributes **no sample** rather than a
zero. A zero would drag a mean toward a conclusion; a missing sample reduces `n`,
which the adjudicator then reports as UNCERTAIN.

---

## 6. Adjudication

```mermaid
sequenceDiagram
    autonumber
    participant A as analyze_agent
    participant K as skeptic
    participant F as results/

    A->>F: read matrix.json
    A->>A: samples_for(cfg) = [warm_median_ms per repeat]
    Note over A: n = 5 repeats, NOT 96 requests

    loop each of 6 configs
        A->>K: Stat(samples_ms)
        K-->>A: mean, median, sd, hw = t_crit(n)·sd/√n, rsd
    end

    loop each of 6 pre-declared comparisons
        A->>A: as_rate(ms) = 1000/ms
        Note over A: verdict() is higher-is-better;<br/>TTFT is lower-is-better
        A->>K: verdict(base_rate, cand_rate)
        K-->>A: (VERIFIED | UNCERTAIN | REJECTED, reason)

        alt expectation == regression_expected
            Note over A: REJECTED is the FINDING here.<br/>Relabel → "REGRESSION CONFIRMED" 🔴
        else improvement or parity
            Note over A: ✅ / ⚠️ / ❌ as-is
        end
    end

    A->>F: write agent_report.md
    A->>F: write agent_analysis.latest.json
```

The comparisons and their expectations are declared in `COMPARISONS` **before**
any data is read:

| Question | Reference | Candidate | Expectation |
|:--|:--|:--|:--|
| Does adding tenants break prefix caching? | `solo_baseline` | `4tenant_default` | regression |
| Does it get worse with more tenants? | `solo_baseline` | `8tenant_default` | regression |
| Does raising the threshold fix it? | `4tenant_default` | `4tenant_sim09` | improvement |
| Does the fix hold when tenants exceed slots? | `8tenant_default` | `8tenant_sim09` | improvement |
| With the fix, is multi-tenant back at parity? | `solo_baseline` | `8tenant_sim09` | parity |
| Do extra slots add anything once the threshold is right? | `8tenant_sim09` | `8tenant_sim09_np8` | improvement |

**Why the relabel exists and why it is not cheating:** `verdict()` only ever
answers "is the candidate faster?" For a comparison whose entire purpose is to
expose a regression, a REJECTED *is* the finding. The raw `verdict` is preserved
in the JSON alongside the displayed label, so the relabel is presentational and
auditable — not a second, friendlier adjudicator.

---

## 7. Mechanism extraction from logs

Latency is the consequence. This is the cause.

```mermaid
flowchart LR
    logs[("server_*.log<br/>30 files")] --> r1["RE_LCP<br/><small>selected slot by LCP similarity,<br/>f_sim_best = X (> Y thold)</small>"]
    logs --> r2["RE_PROMPT<br/><small>prompt eval time = X ms / N tokens</small>"]

    r1 --> agg["per config:<br/>median similarity<br/>LRU fallback count"]
    r2 --> agg2["per config:<br/><b>median tokens recomputed</b><br/>% of requests over 100 tokens"]

    agg --> ev[("slot_evidence.json")]
    agg2 --> ev

    style agg2 fill:#22543d,color:#fff
```

| config | median similarity | **tokens recomputed** | >100 tok |
|:--|--:|--:|--:|
| `solo_baseline` | 0.955 | **35** | 25% |
| `4tenant_default` | 0.716 | **220** | 81% |
| `4tenant_sim09` | 0.955 | **36** | 25% |
| `8tenant_sim09_np8` | 0.955 | 36 | 25% |

Identical across all five repeats — deterministic, because temperature is 0.

**Why this is the stronger evidence:** tokens-recomputed comes from the server's
own accounting and is immune to CPU noise, thermal state and scheduling. TTFT is
what a user feels; this is what actually happened. If the two ever disagree,
this one is right.

---

## 8. Tuning for your own workload

Two modes, and the difference between them is the point.

```mermaid
flowchart TB
    start(["tune_similarity.py"]) --> mode{"--sweep?"}

    mode -->|"yes — authoritative"| sw["for each candidate threshold:<br/>spawn fresh server<br/>run the workload<br/>adjudicate via skeptic"]
    sw --> best["report the measured winner<br/><b>evidence</b>"]

    mode -->|"no — fast"| an["analytic model:<br/>window where intra-tenant similarity<br/>clears the threshold but<br/>inter-tenant does not"]
    an --> heur["report a suggestion<br/><b>labelled a heuristic</b>"]

    style best fill:#22543d,color:#fff
    style heur fill:#744210,color:#fff
```

The analytic mode is labelled a heuristic for a specific reason, not out of
modesty: its model of the similarity formula is **inferred from the flag's
documentation rather than read from source**, and it has already disagreed with
measurement once — the model said the usable window closed at 0.857, while 0.9
empirically worked. When a model contradicts a measurement, the measurement wins
and the model gets a warning label.

The window the analytic mode reasons about, and why it binds from turn 2:

```mermaid
flowchart LR
    t1["turn 1<br/>tenant holds no slot<br/><i>no constraint yet</i>"] --> t2["turn 2+<br/>tenant holds a slot"]
    t2 --> need["threshold must sit<br/>ABOVE inter-tenant similarity<br/>and BELOW intra-tenant"]
    need --> win["usable window"]

    style win fill:#22543d,color:#fff
```

---

## 9. CI on Arm

```mermaid
flowchart TB
    trig1(["push to main<br/>tools/ scripts/ .github/"]) --> sk
    trig2(["workflow_dispatch<br/>repeats, turns"]) --> sk

    subgraph sk["job: skeptic — ubuntu-latest"]
        t1["test_skeptic.py"] --> t2["test_probe_prefill.py"] --> t3["test_agent_tools.py"]
    end

    sk -->|"needs: skeptic"| bench

    subgraph bench["job: bench — ubuntu-24.04-arm · timeout 300 min"]
        b1["apt: build-essential cmake git curl"]
        b2["00_env_report.sh<br/><small>which Arm core is this?</small>"]
        b3["01_build_llama.sh<br/><small>records resolved SHA</small>"]
        b4["fetch_model.sh"]
        b5["run_matrix.py --repeats 5"]
        b6["analyze_agent.py"]
        b7["parse_slot_log.py"]
        b1 --> b2 --> b3 --> b4 --> b5 --> b6 --> b7
    end

    bench --> pub["GITHUB_STEP_SUMMARY<br/><small>report + evidence + env</small>"]
    bench --> art["upload-artifact<br/>results/ · 30 days"]

    style sk fill:#1a365d,color:#fff
    style bench fill:#22543d,color:#fff
```

**Why the gate:** if the adjudicator is broken there is no point spending an Arm
runner on measurements it would misjudge. Cheap x86 tests first; `bench`
declares `needs: skeptic`.

**Why it is free:** `ubuntu-24.04-arm` is an Arm-hosted GitHub runner — Cobalt
100, Neoverse N2 — at no cost on public repositories. Anyone can fork this repo
and re-run every number without paying for hardware, which is the only real
reason to believe a benchmark you did not run yourself.

`if: always()` on the publish and upload steps: a failed run still uploads its
logs, because a failure with no artifact is unrepairable.

---

## 10. How three hypotheses died

The flow that matters most is the one that stopped work early.

```mermaid
flowchart TB
    h1["H1: decode never reaches<br/>KleidiAI i8mm kernels"]
    h1 --> d1["KleidiAI ships an mr=1 dotprod<br/>kernel FOR batch-1 decode.<br/>Dispatch was correct."]
    d1 --> x1["💀 killed by reading<br/><i>described intended behaviour<br/>as a defect</i>"]

    h2["H2: agent servers re-prefill<br/>the system prompt every turn"]
    h2 --> d2["probe_prefill.py<br/>811 ms → 327 ms, ratio 0.054"]
    d2 --> x2["💀 REUSE_WORKS<br/><i>killed in 45 min,<br/>saved ~2 weeks</i>"]

    h3["H3: the regression is<br/>slot CAPACITY"]
    h3 --> d3["-np 4 recovered 14% of a 610%<br/>regression · 4x ctx changed nothing"]
    d3 --> x3["💀 ruled out"]

    h4["H4: slot SELECTION cannot<br/>distinguish tenants sharing<br/>a large preamble"]
    h4 --> d4["30 runs · CIs disjoint<br/>220 → 36 tokens"]
    d4 --> x4["✅ survives"]

    style x1 fill:#742a2a,color:#fff
    style x2 fill:#742a2a,color:#fff
    style x3 fill:#742a2a,color:#fff
    style x4 fill:#22543d,color:#fff
```

The gate that produced this is *measure the opportunity before building the
fix*. H2 is the case that paid for the discipline: a 45-minute probe returned
REUSE_WORKS and cancelled roughly two weeks of work on a problem that did not
exist.

Full history, including what would falsify H4, is in
[methodology.md](methodology.md).

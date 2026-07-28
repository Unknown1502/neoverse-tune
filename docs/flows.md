# Flows

End-to-end operational sequences. [architecture.md](architecture.md) covers what
the components *are*; this covers what actually happens, in order, including when
things go wrong.

- [1. Full local run](#1-full-local-run)
- [2. CI run](#2-ci-run)
- [3. Build A/B flow](#3-build-ab-flow)
- [4. Serving benchmark flow](#4-serving-benchmark-flow)
- [5. Adjudication flow](#5-adjudication-flow)
- [6. Degradation flows](#6-degradation-flows)
- [7. Decision flow: what the crux outcome means](#7-decision-flow-what-the-crux-outcome-means)

---

## 1. Full local run

```mermaid
sequenceDiagram
    autonumber
    actor U as Operator
    participant G as 00_env_report
    participant F as fetch_model
    participant B as 01_build_llama
    participant S as 02_batch_sweep
    participant K as 03_kernel_attrib
    participant V as 04_serve_bench
    participant A as analyze.py
    participant R as results/

    U->>G: run
    G->>G: uname -m must be aarch64
    G->>G: parse MIDR, /proc/cpuinfo Features
    G->>R: env.latest.json (crux_role)
    G-->>U: SUBJECT or CONTROL banner

    U->>F: run
    F->>R: model.path

    U->>B: run
    B->>B: clone/reuse llama.cpp, resolve SHA
    B->>B: build KleidiAI=ON, then =OFF
    B->>B: nm | grep -c kai_
    B->>R: build.latest.json
    Note over B,U: zero kai_ symbols -> loud warning,<br/>A/B is not trustworthy

    U->>S: run
    S->>R: env snapshot (before)
    S->>S: llama-bench -p sweep, then -n baseline
    S->>R: sweep_*.json + env snapshot (after)

    U->>K: run
    K->>K: probe perf on `true`
    alt perf usable
        K->>K: perf record at low and high batch
        K->>R: kernels.latest.json (evidence: execution)
    else perf blocked
        K->>K: nm inventory only
        K->>R: kernels.latest.json (evidence: capability_only)
    end

    U->>V: run
    V->>R: serve_*.json

    U->>A: run
    A->>R: read every artifact
    A->>R: report_*.md + analysis.latest.json
    A-->>U: adjudicated table
```

Each stage writes before the next reads. Any stage can be re-run alone against
existing artifacts — nothing is held in memory between them.

## 2. CI run

The x86 job gates the Arm job. An Arm runner is never spent on a pipeline that
could not report what it measured.

```mermaid
flowchart TD
    trig["workflow_dispatch<br/>or push to main"]
    sk["<b>job: skeptic</b> (ubuntu-latest, x86)"]
    t1["test_skeptic.py<br/>adjudicator vs known answers"]
    t2["test_pipeline.py<br/>report pipeline end to end"]
    t3["test_loadgen.py<br/>latency vs planted timings"]
    gate{"all pass?"}
    stop["stop — no Arm minutes spent"]
    cr["<b>job: crux</b> (ubuntu-24.04-arm, Neoverse N2)"]
    steps["deps → env → model → build →<br/>sweep → attrib → serve → analyze"]
    sum["$GITHUB_STEP_SUMMARY"]
    art["upload results/ (if: always)"]

    trig --> sk --> t1 & t2 & t3 --> gate
    gate -->|no| stop
    gate -->|yes| cr --> steps --> sum & art
```

`if: always()` on the summary and upload steps matters: a run that fails midway
still surfaces its environment report and partial artifacts, which is usually
exactly what you need to diagnose it.

## 3. Build A/B flow

```mermaid
flowchart TD
    s["vendor/llama.cpp @ SHA"]
    c1["cmake -DGGML_CPU_KLEIDIAI=ON"]
    c2["cmake -DGGML_CPU_KLEIDIAI=OFF"]
    b1["build-kleidi/<br/>llama-bench + llama-server"]
    b2["build-base/<br/>llama-bench + llama-server"]
    chk{"kai_* symbols<br/>in kleidi binary?"}
    good["manifest written<br/>A/B is interpretable"]
    bad["WARNING<br/>flag accepted but inert"]

    s --> c1 --> b1 --> chk
    s --> c2 --> b2
    chk -->|"> 0"| good
    chk -->|"= 0"| bad

    style bad fill:#5c1a1a,color:#fff
```

Both variants come from **one checkout, one compiler, one flag apart**. Any other
difference makes the comparison uninterpretable, so there is no code path that
builds them from separate sources.

## 4. Serving benchmark flow

```mermaid
sequenceDiagram
    autonumber
    participant S as 04_serve_bench
    participant L as llama-server
    participant G as loadgen.py

    loop variant in {kleidi, base}
        S->>L: spawn -np = peak concurrency
        activate L
        Note over S,L: trap installed — server is killed<br/>even on Ctrl-C or error
        loop c in {1,2,4,8,16}
            S->>G: --concurrency c
            G->>L: poll /health
            L-->>G: 200 (weights loaded)
            G->>G: warmup requests (discarded)
            par c concurrent clients
                G->>L: POST /v1/chat/completions stream=true
                L-->>G: SSE deltas
            end
            G->>G: TTFT = first delta − send<br/>TPOT = (last − first) / (tokens−1)
            G-->>S: serve_variant_c{c}.json
            alt any request failed
                G-->>S: exit 1 + warning
            end
        end
        S->>L: stop, wait for port release
        deactivate L
    end
```

Health-polling before load is not politeness — benchmarking while weights are
still being paged in measures disk throughput, not inference.

## 5. Adjudication flow

```mermaid
flowchart TD
    A["load sweep JSON<br/>kleidi + base"] --> B["Stat per cell:<br/>mean, median, stdev, CI"]
    B --> C{"samples exist?"}
    C -->|no| U1["UNCERTAIN<br/><i>measurement failed</i>"]
    C -->|yes| D{"n >= 5?"}
    D -->|no| U2["UNCERTAIN<br/><i>too few reps</i>"]
    D -->|yes| E{"RSD <= 10%?"}
    E -->|no| U3["UNCERTAIN<br/><i>host too noisy</i>"]
    E -->|yes| F{"effect > 0?"}
    F -->|no| R["REJECTED"]
    F -->|yes| G{"CIs disjoint?"}
    G -->|no| U4["UNCERTAIN<br/><i>overlap</i>"]
    G -->|yes| H{"effect >= 5%?"}
    H -->|no| U5["UNCERTAIN<br/><i>below floor</i>"]
    H -->|yes| V["VERIFIED"]

    style V fill:#14432a,color:#fff
    style R fill:#5c1a1a,color:#fff
```

Order is load-bearing. "No samples" is tested before "no gain", so a run that
failed can never be published as a negative result.

## 6. Degradation flows

Every failure path narrows what is claimed. None of them fabricate.

```mermaid
flowchart LR
    subgraph perf["perf unavailable"]
        p1["probe fails"] --> p2["nm inventory"] --> p3["evidence:<br/>capability_only"]
    end
    subgraph noise["host throttling"]
        n1["RSD > 10%"] --> n2["UNCERTAIN"] --> n3["published as<br/>not defensible"]
    end
    subgraph nokai["no i8mm core"]
        k1["crux_role=control"] --> k2["banner + report note"] --> k3["null result labelled<br/>meaningless, not negative"]
    end
    subgraph inert["flag inert"]
        i1["0 kai_ symbols"] --> i2["warning"] --> i3["do not report A/B"]
    end
```

## 7. Decision flow: what the crux outcome means

The experiment is designed so that every outcome routes somewhere useful. This
was committed before any data existed ([methodology.md](methodology.md) §7).

```mermaid
flowchart TD
    R{"hot kernel ISA<br/>at N=1 vs N=32"}
    A["<b>switches</b><br/>dotprod → i8mm"]
    B["<b>same at both</b>"]
    C["<b>i8mm already at N=1</b>"]
    D["<b>knee above N=32</b>"]

    R --> A & B & C & D

    A --> A1["Thesis holds.<br/>Quantify the crossover, tune<br/>draft length past it."]
    B --> B1["Thesis inverts — and this is<br/>the <i>stronger</i> finding:<br/>KleidiAI leaves i8mm unused<br/>for small-batch work.<br/>Report upstream."]
    C --> C1["Premise false.<br/>Report plainly; pivot to<br/>the serving-concurrency angle."]
    D --> D1["Real but unreachable by<br/>realistic draft lengths.<br/>Report as a negative result<br/>with the measured bound."]

    style A1 fill:#14432a,color:#fff
    style B1 fill:#1a3a5c,color:#fff
```

There is no branch labelled "quietly drop the project." That is the point of
pre-registering the falsification criteria.

## Related

- [architecture.md](architecture.md) — component design and rationale
- [metrics.md](metrics.md) — what each number means
- [reproducing.md](reproducing.md) — running these flows yourself

# Inference Platform Engineering: A Guided Deep Dive

Running a model is not the same as operating an inference platform. Production
serving must turn finite accelerator memory and compute into predictable first-token
latency, inter-token latency, throughput, availability, and cost while workloads,
sequence lengths, and model versions change.

This course builds a deterministic, offline control plane around a simulated LLM
fleet. It starts with weight and KV-cache arithmetic and ends with an auditable fleet
plan that sizes demand, chooses a parallel layout, places concrete GPU groups, protects
overload, scales on token work, and gates a canary.

The one big idea:

> **An inference platform is a memory-and-queue scheduler.**

Model execution creates value only when scheduling decisions satisfy user SLOs inside
memory, topology, reliability, and cost constraints. “The weights fit” is not a
capacity plan. “The GPU is busy” is not a scaling policy. “Four bit” is not a
performance result.

This is Chapter 22 of the AI Engineering Deep Dives. It follows
[Local Models](https://github.com/alexvervloet/local-models-deep-dive),
[AI in Production](https://github.com/alexvervloet/ai-in-production-deep-dive), and
[AI Architecture](https://github.com/alexvervloet/architecture-deep-dive). Local
Models teaches execution; this repository teaches the serving decisions around it.

## What you will build

By the end, you will be able to:

- budget weight, runtime, and sequence-dependent KV-cache memory;
- distinguish TTFT, TPOT, end-to-end latency, request rate, and token throughput;
- explain why continuous batching avoids request-level head-of-line blocking;
- scope prefix-cache reuse to exact tokens, model, tokenizer, adapter, and tenant;
- gate quantization and speculative decoding on measured evidence;
- choose tensor, pipeline, data, and expert parallel dimensions from fit and topology;
- reserve work before admission, bound queues, and shed overload explicitly;
- place replicas using residency, memory, capabilities, and link locality;
- autoscale from token work without counting warming replicas as ready;
- promote only a warmed canary that meets independent quality and SLO requirements;
- size capacity from workload mix, bursts, headroom, and supplied benchmark/price data;
  and
- produce deterministic JSON that names the reason behind every fleet decision.

## What the simulations do and do not prove

The complete course uses Python's standard library. It needs no GPU, model download,
API key, or network access. That makes control ordering, invariants, and negative paths
easy to inspect and reproduce.

The numeric inputs are fixtures, not claims about real accelerators. The examples do
not model kernels, allocator fragmentation, network contention, preemption, failures,
power, or a runtime's complete scheduler. Replace model metadata, benchmarks, prices,
inventory, and canary measurements with observations from your target stack. Retain
the decision boundaries and counterfactual tests.

## Setup

Python 3.11 or newer is required.

```bash
git clone https://github.com/alexvervloet/inference-platform-deep-dive.git
cd inference-platform-deep-dive
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python check_setup.py
```

The editable install matters. Running a file inside `examples/` places that directory,
not the repository root, first on Python's import path. Install before running labs.

## Learning path

Run the lessons in order. Read the corresponding section of
[TEXTBOOK.md](TEXTBOOK.md), predict the decision, execute the file, and then complete
the matching task in [EXERCISES.md](EXERCISES.md).

| # | Lesson | Consequential decision |
|---|---|---|
| 1 | Weight and KV memory | Does the target concurrency fit? |
| 2 | Service metrics | Does observed service meet every SLO? |
| 3 | Continuous batching | When may a freed decode lane accept work? |
| 4 | Prefix caching | Which exact prefill blocks may be reused? |
| 5 | Quantization | May a measured format advance to staging? |
| 6 | Speculative decoding | Does accepted draft work repay overhead? |
| 7 | Parallelism | Which TP/PP/DP/EP layout fits this topology? |
| 8 | Admission and shedding | Admit, queue, or shed before allocation? |
| 9 | GPU-aware scheduling | Which concrete accelerator group is eligible? |
| 10 | Autoscaling | What desired replica target follows token demand? |
| 11 | Rollouts | Promote, hold, or roll back? |
| 12 | Capacity and cost | How many replicas are required, and what binds? |

### 1. Weight and KV-cache memory

```bash
python examples/01_memory_and_kv.py
```

Inspect weight memory, KV reservation, runtime overhead, usable VRAM, and maximum
concurrency separately. The example supplies `kv_shards` explicitly because grouped-
query attention and runtime layouts do not always shard KV state like weights. Extend
the plan with observed runtime cache capacity before purchasing hardware.

### 2. TTFT, TPOT, and throughput

```bash
python examples/02_latency_and_throughput.py
```

One long prefill breaks p95 TTFT while decode spacing remains healthy. Aggregate
output-token throughput uses the shared wall-clock window; summing per-request token
rates would double-count overlapping time. Production dashboards should segment these
signals by model, workload class, prompt/output length, region, and priority.

### 3. Continuous batching

```bash
python examples/03_continuous_batching.py
```

Static request-level batching leaves the lane released by a short request idle until
the longest batch member completes. Continuous batching refills it at a token
iteration. The simulator isolates this scheduling effect; real systems also trade
prefill throughput against time-between-token stalls and KV availability.

### 4. Automatic prefix caching

```bash
python examples/04_prefix_caching.py
```

The same token prefix reuses complete cached blocks inside one execution/security
scope. The other tenant receives a miss. Never key only on visible text or a caller-
provided label: model, tokenizer, adapter, tenant, and exact token ids affect whether
the cached activation is both correct and authorized.

### 5. Quantization

```bash
python examples/05_quantization.py
```

The four-bit candidate saves weight memory and still fails because its measured
throughput regresses. Format support differs by accelerator and runtime, KV precision
may be separate from weight precision, and quality regression is workload-specific.
Passing this gate means “stage it,” not “promote it.”

### 6. Speculative decoding

```bash
python examples/06_speculative_decoding.py
```

Both rounds use identical costs. Only their actual draft/target agreement changes.
High acceptance amortizes verification; a poor draft spends extra compute to emit one
verified token. Exact speculative algorithms preserve the target distribution, but a
particular runtime/model/workload combination still has to earn a speedup.

### 7. Tensor, pipeline, data, and expert parallelism

```bash
python examples/07_parallelism.py
```

With fast intra-node collectives, the smallest fitting tensor width wins. Without
them, the teaching planner uses pipeline stages rather than pretending cross-device
collectives are free. Remaining complete layouts become data replicas. MoE expert
partitioning is reported independently because experts are a different axis.

### 8. Admission control and load shedding

```bash
python examples/08_admission_and_shedding.py
```

The first request reserves its maximum prompt-plus-output footprint. One later request
gets the bounded queue slot; the next is shed. The controller checks token bound and
deadline before allocation, mutates no state on rejection, and makes retries
idempotent by request id.

### 9. GPU-aware scheduling

```bash
python examples/09_gpu_scheduling.py
```

The scheduler chooses resident weights on an eligible same-node pair. Free memory
alone is insufficient: dtype/kernel capability and collective locality constrain a
parallel replica. Production orchestration must atomically reserve the proposed group
after rechecking inventory because a pure plan can race.

### 10. Queue-based autoscaling

```bash
python examples/10_autoscaling.py
```

Arrival tokens determine steady-state replicas; queued tokens add drain capacity. CPU
is present but intentionally has no authority. Scale-up uses a short cooldown, while
scale-down needs a full stable window. Warming replicas remain desired but contribute
zero ready serving throughput.

### 11. Safe model and runtime rollouts

```bash
python examples/11_rollouts.py
```

A passing shadow remains on hold because it does not exercise the user response path.
The equivalent warmed canary promotes. Unwarmed or undersampled candidates hold;
mature candidates with quality, latency, throughput, or error regressions roll back.

### 12. Capacity and cost planning

```bash
python examples/12_capacity_and_cost.py
```

Each workload contributes burst-adjusted prompt tokens, output tokens, and concurrent
requests. The largest capacity dimension sets replica count after headroom. Hourly
price is supplied rather than embedded because provider pricing and discounts change;
cost pressure never silently reduces the safe fleet size.

## Capstone: release an inference fleet plan

```bash
python hands_on/plan_fleet.py
python hands_on/plan_fleet.py
```

The command writes `fleet-plan.json`. The report includes required evidence, observed
evidence, all decisions, and the exact deciding reasons. It passes only when:

- both independently required workload classes are present;
- weights, KV reservation, runtime overhead, and target concurrency fit;
- the cluster can provide the capacity plan's complete parallel replicas;
- a concrete eligible GPU group can host a replica;
- benign work traverses real placement and admission paths;
- an oversized request is shed by its token bound;
- scaling covers measured token demand;
- the capacity and budget plan passes; and
- a warmed canary passes fixed quality and service gates.

The test suite removes a workload, shrinks GPU inventory, regresses canary TPOT, and
bypasses shedding. Each counterfactual removes evidence while the requirement remains.
That separation is the difference between a release gate and a demo that grades its
own labels.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q inference_platform examples hands_on tests
python check_setup.py
```

## Production measurement map

| Course input | Production source |
|---|---|
| Model/KV dimensions | immutable model config plus runtime-reported cache capacity |
| TTFT/TPOT/E2E | request timestamps and runtime histograms |
| Token throughput | prompt/generation token counters over wall time |
| Quantization/speculation | target-hardware staging benchmark and quality eval |
| GPU inventory/topology | device plugin, node labels, fabric discovery, scheduler state |
| Autoscaling signals | custom metrics for queued/running work and token rates |
| Rollout evidence | comparable shadow/canary slices and independent release policy |
| Capacity and price | workload forecast, failure headroom, measured replica profile, current quote |

## Primary references

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- vLLM, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [Quantization](https://docs.vllm.ai/en/latest/features/quantization/), [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), and [Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- Kubernetes, [Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/), [Node Autoscaling](https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/), and [Workload Management](https://kubernetes.io/docs/concepts/workloads/management/)

These references describe real mechanisms and APIs. This repository uses small,
inspectable models so you can reason about their contracts before selecting a runtime,
orchestrator, accelerator, or cloud.

## Repository map

```text
inference_platform/  pure decision modules
examples/            one isolated runnable lesson per concept
hands_on/            integrated fleet-planning capstone
tests/               unit, negative-path, and anti-vacuity evidence
TEXTBOOK.md           Chapter 22 lecture
EXERCISES.md          extensions with acceptance criteria
check_setup.py        offline environment and capstone verification
LESSONS.md            surprises encountered while building the course
```

## License

MIT

# Chapter 22: The Memory-and-Queue Scheduler

An LLM server performs matrix operations. An inference platform decides which work may
reach those operations, where it runs, how requests share finite memory, when capacity
changes, and whether a new version deserves traffic. Those decisions determine the
experience users and operators actually receive.

## 22.1 The one big idea

> An inference platform is a memory-and-queue scheduler.

Autoregressive inference has two distinct phases. Prefill processes the prompt in
parallel and creates key/value state. Decode repeatedly consumes that state to produce
one next token. Prefill is compute-heavy and strongly affected by prompt length;
decode is repeatedly memory-bandwidth-sensitive and keeps KV state live. A mixed fleet
must schedule both without letting long prefills stall active streams or long outputs
occupy every sequence slot.

That leads to a control loop:

```text
workload -> admit/queue/shed -> batch -> execute -> timestamps/counters
    |              |             |          |              |
 forecast       KV budget      fairness   GPU topology   SLO + capacity evidence
    |                                                        |
    +---- capacity plan <- autoscale <- rollout gate <--------+
```

Every arrow is a measurable claim. The course keeps requirements, stimuli, decisions,
and the final grader separate so a missing case cannot erase the requirement it was
supposed to prove.

## 22.2 Memory is the first scheduler

For a dense model, approximate weight storage as:

```text
weight bytes = parameter count × weight bits / 8
```

Then add the KV cache. For a conventional decoder layout, a useful planning bound per
live token is:

```text
KV bytes/token = 2 × layers × KV heads × head dimension × element bytes
```

The factor two represents keys and values. Multiply by live tokens across concurrent
requests and divide only by the KV sharding that the runtime actually uses. Grouped-
query attention reduces KV heads, quantized KV can reduce element bytes, and some
layouts replicate rather than shard state. Treat those as measured metadata, not
assumptions copied from weight parallelism.

Finally reserve runtime workspaces and allocator headroom:

```text
required/GPU = sharded weights + reserved KV/GPU + runtime overhead
usable/GPU   = physical VRAM × configured usable fraction
```

Weight fit is therefore necessary but insufficient. Context length, concurrency, and
output bounds can flip a model from fit to no-fit without changing one weight byte.
PagedAttention reduces fragmentation by managing KV in non-contiguous blocks; it does
not make KV free. Runtime-reported cache token capacity and an overload load test are
the evidence that should refine the arithmetic.

## 22.3 Measure the service users experience

Four metrics must not collapse into "latency":

- **TTFT**: arrival to first output token. Queueing and prefill dominate it.
- **TPOT**: average time between output tokens after the first. Decode scheduling and
  contention dominate it. Some systems call a related measure inter-token latency or
  time between tokens; define the exact calculation on your dashboard.
- **End-to-end latency**: arrival to completion. It depends heavily on output length.
- **Token throughput**: prompt or output tokens completed per wall-clock second across
  the fleet. It is not the sum of per-request rates.

Request throughput alone hides workload size. A server handling ten short responses
per second may have less work than one generating a long response. Average latency
hides a harmed tail. Always retain prompt/output length, queue time, model revision,
priority, region, and cache/speculation state as dimensions with controlled cardinality.

The course uses nearest-rank p95 for deterministic fixtures. A production release
needs enough observations, comparable workload slices, confidence appropriate to the
risk, and histograms whose bucket boundaries preserve the SLO.

## 22.4 Continuous batching changes the scheduling unit

Traditional static batching groups requests and runs the group to completion. In
generation, members have different output lengths. A short member leaves a hole while
the longest member continues; newly arrived work waits even when a lane is idle.

Iteration-level or continuous batching revisits membership between token iterations.
Completed sequences leave and new work enters. That often raises utilization and cuts
head-of-line waiting, but the policy still matters:

- aggressive prefills can improve aggregate throughput while stalling active decode;
- decode priority protects TPOT but can starve new requests and hurt TTFT;
- chunked prefill can limit individual stalls;
- memory pressure can require preemption or recomputation; and
- priorities need starvation and tenant-fairness controls.

Lesson 3 intentionally holds execution cost constant to isolate membership. The
[Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu) establishes the
iteration-level scheduling idea; production selection needs a workload replay against
the runtime and configuration you will deploy.

## 22.5 Prefix caching is a correctness boundary

A shared prefix such as system instructions, a document, or few-shot examples can
reuse its prefill KV blocks. The performance opportunity is largest when prefixes are
long, stable, and repeated. The correctness key must include every input that changes
the activation:

- exact token ids and block boundaries;
- model and tokenizer revisions;
- adapter or fine-tune revision;
- relevant multimodal inputs and preprocessing versions; and
- tenant, ACL, or security domain when cached state reflects protected content.

A text hash is insufficient because tokenizer revisions can change token ids. A
caller-provided "same prefix" label is not evidence. Cache lookup computes identity
from actual execution inputs and verifies the exact prefix. Invalidation, eviction,
and hit-rate metrics then become operational policy. A high hit rate is not inherently
good if it crosses an authorization boundary.

## 22.6 Quantization trades several things at once

Reducing weight precision usually reduces weight storage. It may also improve memory
bandwidth or permit larger batches, but kernels, dequantization, hardware support,
activation precision, and workload shape determine performance. Weight quantization
does not automatically shrink KV cache.

A useful candidate record contains:

1. immutable model, quantization method, runtime, and hardware identity;
2. weight and total runtime memory under the target context/concurrency;
3. quality scores on protected tasks and slices;
4. TTFT, TPOT, throughput, errors, and warmup behavior; and
5. comparison with a measured baseline under equivalent load.

Use absolute quality floors and maximum regression together. A weak baseline should
not authorize an unacceptable candidate, while an absolute floor alone can permit a
large regression. Passing permits a canary or staging test; it is not direct promotion.
The current vLLM [quantization matrix](https://docs.vllm.ai/en/latest/features/quantization/)
also illustrates why format support must be checked against actual hardware.

## 22.7 Speculative decoding has to pay back its verification

A draft model proposes multiple tokens cheaply. The target model verifies them in one
parallel step that covers every draft position plus the position after the last one.
It accepts the matching prefix and supplies the first correction when they diverge, or
a bonus token from that extra position when the whole draft agrees. Correct speculative
sampling preserves the target distribution; it does not merely trust the smaller model.

That extra position is why speculation can win at all. A round of `k` drafts emits
between 1 and `k + 1` tokens, so a fully accepted round returns more tokens than it
proposed. Counting only the accepted drafts understates the technique.

For one simplified round:

```text
emitted tokens   = accepted draft tokens + 1
baseline cost    = emitted tokens × target cost/token
speculative cost = drafted tokens × draft cost/token + target verification cost
speedup          = baseline cost / speculative cost
```

Acceptance depends on the draft/target pair, sampling configuration, prompt domain,
and position. A poor draft pays for several proposals and a verification to emit one
token. A good draft can amortize verification across several emitted tokens. Measure
acceptance by position and end-to-end latency with batching enabled; a microbenchmark
that omits scheduler interaction is incomplete. See
[Leviathan et al.](https://arxiv.org/abs/2211.17192) for the exact-decoding method.

## 22.8 Parallelism dimensions solve different constraints

**Tensor parallelism (TP)** splits operations inside layers. It can make one layer fit
and reduce per-GPU weight memory, but requires frequent collectives. Keep TP inside a
high-bandwidth domain when possible.

**Pipeline parallelism (PP)** assigns layer ranges to stages. It can cross nodes and
avoid some fine-grained collectives, but stage imbalance and bubbles affect latency and
throughput. It is useful when a model exceeds one node or when intra-node links do not
favor TP.

**Data parallelism (DP)** duplicates complete model layouts. It scales independent
requests only after each replica already fits. It cannot rescue a too-large replica.

**Expert parallelism (EP)** distributes MoE experts. Expert count, routing, active
experts, token imbalance, and all-to-all communication matter. Dense attention and MoE
expert layers may use different dimensions.

Start from fit, then topology, then demand:

1. Can a single GPU fit weights, KV target, and overhead?
2. If not, what TP width fits inside a fast-link group?
3. Does the model require PP across stages or nodes?
4. Does the runtime support the composition and divisibility?
5. How many complete layouts remain for DP?
6. For MoE, how are experts distributed and balanced?

The vLLM [parallel deployment guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
documents supported compositions and recommends validating cache capacity and estimated
concurrency in runtime logs. The course planner is intentionally a smaller decision
model, not a replacement for runtime validation.

## 22.9 Admission happens before allocation

Overload is inevitable; uncontrolled overload is optional. Admission should reserve a
request's worst plausible footprint before external work:

- prompt plus maximum output/KV tokens;
- queue slot and deadline budget;
- tenant/priority quota;
- speculative or parallel expansion where relevant; and
- idempotency identity for retries.

The decision order matters. Reject a request that violates its own bound before
considering shared capacity. Reject work predicted to miss its deadline rather than
letting it expire after consuming resources. If live capacity is full, use a bounded
queue. When that is full, shed with a stable reason and retry guidance.

An atomic reservation prevents two requests from both seeing the last capacity and
claiming it. Rejection must leave counters unchanged, and replay must not double
reserve. Production policies may protect interactive or paid tiers, but must also
prevent starvation and abusive priority selection.

## 22.10 GPU-aware scheduling and fragmentation

"Two free GPUs" says little. A replica may require:

- enough free memory on every device;
- a supported dtype, quantization kernel, compute capability, or partition type;
- same-node or same-fabric locality for its TP group;
- compatible driver/runtime versions;
- failure-domain and anti-affinity constraints; and
- model residency to avoid a long cold load.

Placement is a constrained group decision. Picking individual GPUs greedily can leave
stranded fragments that cannot host the next parallel replica. Packing a tight fit can
preserve larger groups; reusing resident weights can cut warmup. Those objectives can
conflict, so rank them explicitly and return the deciding facts.

A pure planner is not an allocator. Recheck inventory and reserve the entire group
atomically. Coordinate pod/workload scheduling with node autoscaling: adding a pod does
not create an accelerator, and adding a node does not make warmed model capacity ready.

## 22.11 Autoscaling is a delayed control loop

Scale steady-state demand from tokens, not CPU:

```text
rate replicas  = ceil(arrival token rate / usable service rate per replica)
queue replicas = ceil(queued tokens / (service rate × drain window))
```

Use the appropriate mix of prompt and output work for the runtime; the lesson uses one
token rate to keep the loop visible. Clamp to fleet limits. Scale up after a short
cooldown, but scale down only after a full stabilization window. Otherwise cold loads
and transient lulls create oscillation.

Track at least three states: desired, warming, and ready. Warming capacity may prevent
launching duplicate replicas, but it cannot serve the current queue. Readiness should
turn true only after model load, compilation/capture, cache initialization, and a
representative warmup request. Kubernetes HPA supports custom metrics and stabilization
behavior; node autoscaling is a separate loop with longer provisioning delay.

## 22.12 Rollouts are experiments with automatic exits

A safe sequence can be:

1. load the candidate with zero user traffic;
2. warm it and prove readiness;
3. shadow comparable requests for output/quality evidence;
4. canary a small, bounded user cohort;
5. expand through fixed gates; and
6. remove or retain the previous version only after rollback is proven.

Shadow output cannot prove the serving path's user-facing latency or error behavior.
An unwarmed canary should hold rather than be mislabeled a regression. Once the sample
is mature, regressions in protected quality, TTFT, TPOT, throughput, or error rate
should remove traffic automatically.

Declare requirements before looking at candidate results. Compare like workload slices
and control for cache state, hardware, time, and load. Averages cannot protect a tail
SLO. Preserve exact failure reasons so operators know whether to wait for warmup,
collect evidence, tune serving, or roll back the artifact.

## 22.13 Capacity and cost are scenario decisions

Capacity begins with workload classes, not one average RPS. For each class record
arrival rate, prompt/output distributions, service time, SLO tier, cache behavior,
regional/failure-domain needs, and burst scenario. The course uses means plus an
explicit burst multiplier to expose the arithmetic; production planning needs
distributions and replay.

Calculate prefill, decode, and concurrency requirements separately, discount measured
replica capacity by headroom, and take the largest. Add redundancy and maintenance
constraints appropriate to your failure model. If the required fleet exceeds cost or
quota, report the conflict. Do not shrink below safe capacity to make a spreadsheet
green.

Normalize cost carefully. Cost per million offered output tokens can compare scenarios,
but utilization, idle redundancy, input work, transfer, storage, control-plane nodes,
and discounts belong in total cost. Prices are volatile inputs; store quote source and
date rather than teaching a timeless dollar constant.

## 22.14 The capstone evidence contract

`hands_on/plan_fleet.py` declares two required workload classes and eight control
claims separately from all model, inventory, benchmark, and canary inputs. It adds
evidence only when real decision outputs satisfy those claims. The report retains both
the decision structure and a human-readable reason.

The anti-vacuity tests prove four properties:

- deleting a workload removes evidence but not its requirement;
- insufficient inventory prevents placement evidence;
- a TPOT regression removes canary evidence; and
- bypassing oversized-request shedding fails even though the rest of the plan passes.

This is still a teaching release gate. In production, sign input snapshots and output
evidence, test evaluator failures, require freshness, bind reports to immutable artifact
and configuration revisions, and execute rollback against real traffic infrastructure.

## 22.15 Senior review questions

Before approving an inference design, ask:

1. Which memory term grows with live tokens, and who reserves it?
2. Which workload slice binds TTFT, TPOT, throughput, and concurrency?
3. Can the scheduler admit new prefills without starving active decodes?
4. What exact inputs scope prefix reuse?
5. Which measured evidence enables quantization or speculation?
6. Which parallel collective crosses which physical link?
7. What happens when both KV capacity and the queue are full?
8. Can placement atomically reserve a complete eligible topology?
9. Which replicas are desired, warming, and truly ready?
10. What independent observation automatically rolls a canary back?
11. Which burst and failure scenario sets capacity?
12. Which conclusions came from measurements, and which remain assumptions?

If the answer is only a framework name, GPU count, or average benchmark, the platform
decision has not yet been made.

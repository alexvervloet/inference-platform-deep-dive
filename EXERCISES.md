# Inference Platform Engineering Exercises

These exercises turn each deterministic lesson into a design decision. Write the
invariant and a failing test before changing a module. Keep requirements separate from
stimuli, and make every counterfactual cross the intended control boundary.

## 1. Add sliding-window and KV-precision scenarios

Extend the memory model with an explicitly declared effective KV window and a KV dtype
that may differ from weight precision. Compare full-context and sliding-window plans.

Acceptance criteria:

- Weight and KV element precision remain separate inputs.
- A shorter effective window reduces KV reservation without changing weight memory.
- A model whose weights fit but target concurrency does not is rejected.
- The test shows its byte/GiB arithmetic and crosses the fit boundary visibly.

Stretch: compare the planner with cache capacity reported by a real runtime, recording
the runtime/model/config revision rather than changing the deterministic fixture.

## 2. Segment service objectives by workload class

Add interactive and batch traces with separate SLOs. Return both per-class and fleet
reports without averaging away the interactive tail.

Acceptance criteria:

- TTFT, TPOT, E2E, input tokens, and output tokens remain separate fields.
- A slow batch trace does not fail an interactive objective, and vice versa.
- Unknown or empty workload classes fail validation.
- One slow first token breaks TTFT while an unchanged decode sequence still passes TPOT.

Stretch: add deterministic histogram buckets and prove the chosen buckets can represent
each SLO threshold without interpolation ambiguity.

## 3. Add prefill/decode scheduling policy

Extend continuous batching so a prefill consumes configurable work units and can be
chunked. Compare prefill-first, decode-first, and bounded-chunk policies.

Acceptance criteria:

- Outcomes come from the simulated timeline, never a policy label on the request.
- Prefill-first improves one throughput scenario while harming an active stream's TPOT.
- Decode-first protects TPOT but can delay a new request's TTFT.
- Bounded chunks enforce an explicit maximum decode stall.

Stretch: implement starvation prevention and demonstrate it with a sustained arrival
stream rather than a hard-coded "starved" result.

## 4. Add prefix-cache eviction and invalidation

Give the cache a block budget and deterministic LRU eviction. Add explicit invalidation
for a model, tokenizer, adapter, tenant, or corpus revision.

Acceptance criteria:

- Only complete exact token blocks enter the cache.
- A changed scope cannot hit even when token ids match.
- Eviction derives from access order and budget, not a fixture's expected victim.
- Invalidating one tenant leaves another tenant's entries available.

Stretch: record saved prefill tokens and show why hit rate alone can reward many tiny,
low-value hits.

## 5. Compare two quantization candidates

Add one candidate with strong quality but unsupported hardware and another with lower
quality but good serving results. Decide which, if either, advances.

Acceptance criteria:

- Hardware support, absolute quality, quality regression, memory, TTFT, and throughput
  are independent gates.
- Bit width is never used to manufacture latency or throughput.
- Every measurement names immutable model/runtime/hardware identity.
- Passing means staging, not production promotion.

Stretch: add protected quality slices so an acceptable aggregate cannot hide one
critical task regression.

## 6. Evaluate speculation over many rounds

Aggregate draft/target sequences into acceptance-by-position, emitted tokens, total
cost, and p95 round latency. Compare two draft models.

Acceptance criteria:

- Acceptance is computed from actual token agreement.
- A fully accepted round emits one more token than it drafted.
- The baseline comparison covers the same emitted tokens.
- A low-acceptance draft loses even if its individual token cost is small.
- A verification exception fails the evaluation rather than dropping the round.

Stretch: include batch interference and show a configuration that helps isolated
latency but harms fleet throughput.

## 7. Enforce failure-domain-aware parallelism

Extend the topology with nodes, racks, per-link bandwidth classes, and runtime-supported
TP/PP/EP compositions. Reject unsupported or unsafe layouts.

Acceptance criteria:

- TP groups cannot silently cross a disallowed slow link.
- DP is allocated only after a complete replica layout fits.
- An MoE expert partition divides the declared expert count.
- Removing one GPU or capability changes the layout or yields a specific no-fit reason.

Stretch: maintain a ranked set of feasible alternatives with predicted communication
cost, then require a measured staging result before selecting among close candidates.

## 8. Add priority without starvation

Introduce interactive and batch admission classes, per-tenant quotas, and aging. Keep
worst-case reservation and bounded queues.

Acceptance criteria:

- Priority comes from trusted routing policy, not request/model text.
- Rejected or expired work mutates no live-token or queue counters.
- Replays reserve once.
- Sustained interactive arrivals cannot starve a queued batch request beyond its
  declared maximum wait.

Stretch: return retry-after guidance derived from service observations while keeping
the estimate explicitly non-binding.

## 9. Prevent GPU topology fragmentation

Schedule a sequence of one-, two-, and four-GPU replicas. Compare tight-fit packing
with a naive "most free memory first" policy.

Acceptance criteria:

- Every placed group satisfies memory, capability, and locality.
- The naive policy strands enough total GPUs but no eligible four-GPU group.
- The topology-aware policy preserves a valid group for the later request.
- Planning and atomic allocation are represented as separate operations.

Stretch: include model-load time and show when residency should lose to preserving a
scarce high-bandwidth group.

## 10. Coordinate pod and node autoscaling

Add pending replicas, node provisioning delay, model loading, readiness, and scale-down
termination. Simulate a burst longer and shorter than total warmup.

Acceptance criteria:

- CPU never becomes authoritative for token demand.
- Pending and warming replicas contribute zero ready throughput.
- The pod loop can request capacity while the node loop explains why it is pending.
- A full stabilization window is required before scale-down.

Stretch: add forecast-based prewarming and compare its idle cost with reactive SLO
violations under an independently supplied event schedule.

## 11. Build a staged rollout ladder

Implement zero-traffic warmup, shadow, 1% canary, 10% canary, and promotion. Give each
stage explicit evidence and exit criteria.

Acceptance criteria:

- Requirements are declared separately from candidate measurements.
- Shadow success cannot directly promote.
- Insufficient or unwarmed evidence holds rather than passes.
- One protected quality or p95 latency regression removes traffic and names the gate.
- Rollback itself is exercised and produces observable evidence.

Stretch: detect a Simpson's-paradox workload shift by comparing stratified and aggregate
candidate results.

## 12. Plan N+1 and regional capacity

Split workload across two failure domains, then remove the largest domain. Include
maintenance headroom and a current price supplied as an input.

Acceptance criteria:

- Prompt, decode, and concurrency requirements are calculated separately.
- A dimension that only ties after rounding is reported as a tie, not as the winner.
- The loss scenario, not mean traffic, sets fleet size.
- A budget violation reports the unsafe required fleet; it never shrinks capacity.
- Changing price changes cost but not replica demand.

Stretch: compare on-demand and committed pricing while representing commitment risk and
unused reserved capacity explicitly.

## Capstone extension: make evidence fresh and attributable

Extend `hands_on/plan_fleet.py` so every input and decision includes an immutable
revision, observed-at timestamp, and expiry. Bind the JSON report to those inputs with
a digest. Do not use the report itself as its own source of truth.

Acceptance criteria:

- Required control/workload coverage remains independent of observed evidence.
- Stale benchmark, inventory, price, or canary evidence fails release.
- Changing any bound input changes the report digest.
- Removing one workload, weakening one control, shrinking one GPU, and regressing one
  canary metric each fail for their own deciding reason.
- Benign work still traverses real admission and placement; a block-all plan fails.
- Running the unchanged capstone twice produces byte-identical JSON apart from any
  deliberately excluded generation timestamp.

Production question: which system signs the evidence, where is the trusted policy
stored, and how does deployment verify both without allowing the candidate pipeline to
rewrite its own requirements?

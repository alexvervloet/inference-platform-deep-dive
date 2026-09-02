"""Lesson 12: workload mix and bursts set capacity; supplied prices set cost."""

from inference_platform.capacity import (
    CapacityRequirements,
    ReplicaBenchmark,
    WorkloadClass,
    plan_capacity,
)


# Long retrieval prompts make prefill and decode land on the same replica count
# after rounding. The plan reports both rather than letting a sort order pick one.
workloads = (
    WorkloadClass("interactive", 1.5, 900, 80, 1.5, 1.5),
    WorkloadClass("generation", 0.5, 400, 500, 5, 2.0),
)
benchmark = ReplicaBenchmark("staging-profile", 1000, 300, 16, hourly_cost=4.25)
requirements = CapacityRequirements(2, 20, 0.25, max_hourly_cost=50)
plan = plan_capacity(workloads, benchmark, requirements)

usable = 1 - plan.reserved_headroom_fraction
unrounded = {
    "prefill tokens": plan.prompt_tokens_per_second / (benchmark.prefill_tokens_per_second * usable),
    "decode tokens": plan.output_tokens_per_second / (benchmark.output_tokens_per_second * usable),
    "request concurrency": plan.concurrent_requests / (benchmark.max_concurrent_requests * usable),
    "minimum replica floor": float(requirements.min_replicas),
}

print(f"required replicas: {plan.required_replicas}")
print("unrounded demand per dimension:")
for name, demand in sorted(unrounded.items(), key=lambda item: -item[1]):
    mark = "binds" if name in plan.binding_dimensions else "slack"
    print(f"  {name:<22} {demand:.2f} -> {mark}")
print(f"binding after rounding: {', '.join(plan.binding_dimensions)}")
print(f"named bottleneck: {plan.bottleneck} (largest unrounded demand among those)")
print(f"reserved headroom: {plan.reserved_headroom_fraction:.0%}")
print(f"hourly cost from supplied quote: ${plan.hourly_cost:.2f}")
print(f"cost / million offered output tokens: ${plan.cost_per_million_output_tokens:.2f}")
print(f"release ready: {plan.release_ready}")

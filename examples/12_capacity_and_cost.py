"""Lesson 12: workload mix and bursts set capacity; supplied prices set cost."""

from inference_platform.capacity import (
    CapacityRequirements,
    ReplicaBenchmark,
    WorkloadClass,
    plan_capacity,
)


workloads = (
    WorkloadClass("interactive", 1.5, 200, 80, 1.5, 1.5),
    WorkloadClass("generation", 0.5, 100, 500, 5, 2.0),
)
benchmark = ReplicaBenchmark("staging-profile", 1000, 300, 16, hourly_cost=4.25)
requirements = CapacityRequirements(2, 20, 0.25, max_hourly_cost=50)
plan = plan_capacity(workloads, benchmark, requirements)

print(f"required replicas: {plan.required_replicas} ({plan.bottleneck})")
print(f"reserved headroom: {plan.reserved_headroom_fraction:.0%}")
print(f"hourly cost from supplied quote: ${plan.hourly_cost:.2f}")
print(f"cost / million offered output tokens: ${plan.cost_per_million_output_tokens:.2f}")
print(f"release ready: {plan.release_ready}")

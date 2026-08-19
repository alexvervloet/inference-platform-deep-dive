"""Lesson 10: scale from token demand and backlog, not a convenient CPU graph."""

from inference_platform.autoscaling import (
    AutoscalingPolicy,
    ScaleObservation,
    ScaleState,
    recommend_replicas,
)


policy = AutoscalingPolicy(1, 10, 100, 0.8, 10, 5, 60)
observation = ScaleObservation(
    timestamp=100,
    queued_tokens=800,
    arrival_tokens_per_second=160,
    cpu_utilization=0.12,
)
state = ScaleState(desired_replicas=1, ready_replicas=1, warming_replicas=0, last_scale_at=0)
decision = recommend_replicas((observation,), state, policy, now=100)

print(f"calculated demand: {decision.calculated_demand_replicas} replicas")
print(f"target: {decision.target_replicas} replicas")
print(f"ready serving capacity: {decision.ready_capacity_tokens_per_second:.0f} tokens/s")
print(decision.reason)

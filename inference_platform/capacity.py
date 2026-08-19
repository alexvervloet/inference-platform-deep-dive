"""Size a fleet from workload mix, burst demand, headroom, and supplied prices.

Requests per second is not a portable capacity unit: long prompts stress prefill,
long outputs stress decode, and long service times consume concurrency slots. The
planner sizes all three dimensions after applying each class's burst multiplier and
an explicit headroom target, then reports whichever dimension actually binds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WorkloadClass:
    name: str
    requests_per_second: float
    prompt_tokens: int
    output_tokens: int
    mean_service_seconds: float
    burst_multiplier: float


@dataclass(frozen=True)
class ReplicaBenchmark:
    name: str
    prefill_tokens_per_second: float
    output_tokens_per_second: float
    max_concurrent_requests: int
    hourly_cost: float


@dataclass(frozen=True)
class CapacityRequirements:
    min_replicas: int
    max_replicas: int
    reserved_headroom_fraction: float
    max_hourly_cost: float


@dataclass(frozen=True)
class CapacityPlan:
    release_ready: bool
    required_replicas: int
    bottleneck: str
    prompt_tokens_per_second: float
    output_tokens_per_second: float
    concurrent_requests: float
    reserved_headroom_fraction: float
    hourly_cost: float
    cost_per_million_output_tokens: float
    violations: tuple[str, ...]


def plan_capacity(
    workloads: tuple[WorkloadClass, ...],
    benchmark: ReplicaBenchmark,
    requirements: CapacityRequirements,
) -> CapacityPlan:
    """Decide fleet size and budget fit from independent demand and supply inputs.

    Each workload class contributes burst-adjusted prompt tokens, output tokens, and
    concurrency (Little's Law using its supplied mean service time). Per-replica
    benchmark capacities are discounted by reserved headroom. The maximum of prefill,
    decode, concurrency, and minimum-replica requirements sets fleet size; the name of
    that exact constraint is returned. Budget and fleet ceilings gate release after
    sizing, so the planner never trims an unsafe plan merely to make cost pass.

    Benchmark results and hourly price are inputs on purpose: measure the intended
    model/runtime/hardware and provide a current quote. This arithmetic does not
    predict GPU performance, transfer charges, reserved-instance discounts, failure
    domains, or statistical tails beyond the declared burst and headroom assumptions.
    """

    _validate(workloads, benchmark, requirements)
    prompt_tps = sum(
        item.requests_per_second * item.prompt_tokens * item.burst_multiplier
        for item in workloads
    )
    output_tps = sum(
        item.requests_per_second * item.output_tokens * item.burst_multiplier
        for item in workloads
    )
    concurrency = sum(
        item.requests_per_second * item.mean_service_seconds * item.burst_multiplier
        for item in workloads
    )
    usable = 1 - requirements.reserved_headroom_fraction
    dimensions = {
        "minimum replica floor": requirements.min_replicas,
        "prefill tokens": math.ceil(prompt_tps / (benchmark.prefill_tokens_per_second * usable)),
        "decode tokens": math.ceil(output_tps / (benchmark.output_tokens_per_second * usable)),
        "request concurrency": math.ceil(concurrency / (benchmark.max_concurrent_requests * usable)),
    }
    bottleneck, replicas = max(dimensions.items(), key=lambda item: (item[1], item[0]))
    hourly_cost = replicas * benchmark.hourly_cost
    cost_per_million = hourly_cost / (output_tps * 3600) * 1_000_000
    violations: list[str] = []
    if replicas > requirements.max_replicas:
        violations.append("required replicas exceed fleet ceiling")
    if hourly_cost > requirements.max_hourly_cost:
        violations.append("required fleet exceeds hourly cost budget")
    return CapacityPlan(
        not violations,
        replicas,
        bottleneck,
        prompt_tps,
        output_tps,
        concurrency,
        requirements.reserved_headroom_fraction,
        hourly_cost,
        cost_per_million,
        tuple(violations),
    )


def _validate(
    workloads: tuple[WorkloadClass, ...],
    benchmark: ReplicaBenchmark,
    requirements: CapacityRequirements,
) -> None:
    if not workloads:
        raise ValueError("at least one workload class is required")
    names = [item.name for item in workloads]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("workload names must be present and unique")
    if any(
        item.requests_per_second < 0
        or min(item.prompt_tokens, item.output_tokens) < 0
        or item.mean_service_seconds <= 0
        or item.burst_multiplier < 1
        for item in workloads
    ):
        raise ValueError("workload rates are invalid")
    if sum(item.output_tokens * item.requests_per_second for item in workloads) <= 0:
        raise ValueError("positive output-token demand is required for cost normalization")
    if not benchmark.name.strip() or min(
        benchmark.prefill_tokens_per_second,
        benchmark.output_tokens_per_second,
        benchmark.max_concurrent_requests,
    ) <= 0 or benchmark.hourly_cost < 0:
        raise ValueError("benchmark capacities must be positive and cost non-negative")
    if requirements.min_replicas <= 0 or requirements.max_replicas < requirements.min_replicas:
        raise ValueError("replica requirements are invalid")
    if not 0 <= requirements.reserved_headroom_fraction < 1:
        raise ValueError("reserved headroom must be in [0, 1)")
    if requirements.max_hourly_cost < 0:
        raise ValueError("hourly budget may not be negative")

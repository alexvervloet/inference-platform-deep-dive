"""Scale from token work while respecting warmup and stabilization state.

CPU is often weak evidence for GPU-bound inference. Queue depth alone is also weak
because requests vary by token count. This controller uses arriving and queued tokens,
measured per-replica service rate, a target utilization, and a queue-drain window.
Warming replicas remain desired capacity but are never reported as ready capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScaleObservation:
    timestamp: float
    queued_tokens: int
    arrival_tokens_per_second: float
    cpu_utilization: float


@dataclass(frozen=True)
class ScaleState:
    desired_replicas: int
    ready_replicas: int
    warming_replicas: int
    last_scale_at: float


@dataclass(frozen=True)
class AutoscalingPolicy:
    min_replicas: int
    max_replicas: int
    service_tokens_per_second_per_replica: float
    target_utilization: float
    queue_drain_seconds: float
    scale_up_cooldown_seconds: float
    scale_down_stabilization_seconds: float


@dataclass(frozen=True)
class ScaleDecision:
    target_replicas: int
    calculated_demand_replicas: int
    ready_capacity_tokens_per_second: float
    reason: str


def recommend_replicas(
    observations: tuple[ScaleObservation, ...],
    state: ScaleState,
    policy: AutoscalingPolicy,
    *,
    now: float,
) -> ScaleDecision:
    """Choose a desired replica target from token demand and control-loop history.

    For each observation, rate demand is arrival tokens divided by target usable
    service rate. Queue recovery adds enough replicas to drain observed backlog in
    the configured window.

    The two terms discount differently on purpose. Steady arrivals are sized against
    the utilization-discounted rate, keeping the reserve that absorbs variance. Drain
    replicas are sized against the full measured rate, on the assumption that clearing
    a transient backlog is exactly what that reserve is for. That is the optimistic
    reading: if the backlog is not transient, the drain term under-provisions, and an
    operator who expects sustained overload should discount it the same way. The
    newest demand can scale up after a short cooldown.
    Scaling down requires a complete stabilization window whose every observation
    supports the lower target, preventing a brief lull from terminating warm models.

    `cpu_utilization` is retained in the observation to make its non-authority visible:
    it does not affect the decision. Warming replicas count toward the existing
    desired target but not the returned ready throughput. This is deterministic
    control logic, not a replacement for runtime custom metrics or node autoscaling.
    """

    _validate(observations, state, policy, now)
    ordered = tuple(sorted(observations, key=lambda item: item.timestamp))
    demands = tuple(_demand(observation, policy) for observation in ordered)
    newest_demand = demands[-1]
    target = state.desired_replicas
    reason = "current desired capacity matches calculated token demand"

    if newest_demand > state.desired_replicas:
        if now - state.last_scale_at >= policy.scale_up_cooldown_seconds:
            target = newest_demand
            reason = "arrival and queued-token demand cleared scale-up cooldown"
        else:
            reason = "scale-up demand is inside cooldown"
    elif newest_demand < state.desired_replicas:
        window_start = now - policy.scale_down_stabilization_seconds
        covers_window = ordered[0].timestamp <= window_start
        stable_demands = [
            demand
            for observation, demand in zip(ordered, demands, strict=True)
            if observation.timestamp >= window_start
        ]
        if covers_window and stable_demands and max(stable_demands) < state.desired_replicas:
            target = max(stable_demands)
            reason = "lower token demand persisted for the full stabilization window"
        else:
            reason = "scale-down held for an incomplete or unstable window"

    ready_capacity = (
        state.ready_replicas
        * policy.service_tokens_per_second_per_replica
        * policy.target_utilization
    )
    return ScaleDecision(target, newest_demand, ready_capacity, reason)


def _demand(observation: ScaleObservation, policy: AutoscalingPolicy) -> int:
    usable_rate = (
        policy.service_tokens_per_second_per_replica * policy.target_utilization
    )
    rate_replicas = math.ceil(observation.arrival_tokens_per_second / usable_rate)
    queue_replicas = math.ceil(
        observation.queued_tokens
        / (policy.service_tokens_per_second_per_replica * policy.queue_drain_seconds)
    )
    return min(policy.max_replicas, max(policy.min_replicas, rate_replicas + queue_replicas))


def _validate(
    observations: tuple[ScaleObservation, ...],
    state: ScaleState,
    policy: AutoscalingPolicy,
    now: float,
) -> None:
    if not observations:
        raise ValueError("at least one scale observation is required")
    if policy.min_replicas <= 0 or policy.max_replicas < policy.min_replicas:
        raise ValueError("replica bounds are invalid")
    if policy.service_tokens_per_second_per_replica <= 0:
        raise ValueError("service rate must be positive")
    if not 0 < policy.target_utilization <= 1:
        raise ValueError("target_utilization must be in (0, 1]")
    if min(
        policy.queue_drain_seconds,
        policy.scale_up_cooldown_seconds,
        policy.scale_down_stabilization_seconds,
    ) < 0 or policy.queue_drain_seconds == 0:
        raise ValueError("timing windows are invalid")
    if min(state.desired_replicas, state.ready_replicas, state.warming_replicas) < 0:
        raise ValueError("replica counts may not be negative")
    if state.ready_replicas + state.warming_replicas > state.desired_replicas:
        raise ValueError("ready plus warming replicas exceed desired replicas")
    if any(
        item.timestamp > now
        or item.queued_tokens < 0
        or item.arrival_tokens_per_second < 0
        for item in observations
    ):
        raise ValueError("observations contain future timestamps or negative demand")

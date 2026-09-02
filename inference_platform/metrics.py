"""Turn timestamped requests into user-latency and fleet-throughput evidence.

TTFT measures the wait until streaming begins. TPOT measures the average spacing
of the remaining output tokens, and end-to-end latency includes both. Aggregate
output-token throughput instead divides all generated tokens by the observation
window. Collapsing these into one average hides whether prefill, decode, or queueing
hurt users, so the decision below keeps them separate and gates their tails.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RequestTrace:
    identifier: str
    arrived_at: float
    first_token_at: float
    completed_at: float
    output_tokens: int


@dataclass(frozen=True)
class ServiceObjectives:
    max_p95_ttft_seconds: float
    max_p95_tpot_seconds: float
    max_p95_e2e_seconds: float
    min_output_tokens_per_second: float


@dataclass(frozen=True)
class ServiceReport:
    meets_objectives: bool
    p95_ttft_seconds: float
    p95_tpot_seconds: float
    p95_e2e_seconds: float
    output_tokens_per_second: float
    violations: tuple[str, ...]


def evaluate_service(
    traces: tuple[RequestTrace, ...], objectives: ServiceObjectives
) -> ServiceReport:
    """Decide whether observed traces meet every independently declared objective.

    Validation runs before metric calculation so impossible timestamps cannot look
    fast. The p95 uses the nearest-rank definition, which is deterministic for this
    small teaching implementation. TPOT excludes the first output token because its
    latency is already represented by TTFT; a one-token response has TPOT zero.
    Throughput covers the full interval from first arrival to last completion and
    therefore includes idle and queue time rather than summing per-request rates.
    A window of zero length is rejected: every trace would have to arrive and finish
    at one instant while emitting tokens, and reporting infinite throughput for it
    would pass any minimum a caller declared.

    This function evaluates supplied observations; it does not claim that a short
    trace is statistically representative. Production gates also need a deliberate
    load shape, a sufficient sample count, and breakdowns by workload class.
    """

    if not traces:
        raise ValueError("at least one request trace is required")
    if min(
        objectives.max_p95_ttft_seconds,
        objectives.max_p95_tpot_seconds,
        objectives.max_p95_e2e_seconds,
        objectives.min_output_tokens_per_second,
    ) < 0:
        raise ValueError("service objectives may not be negative")

    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    for trace in traces:
        if not trace.identifier.strip():
            raise ValueError("trace identifiers are required")
        if trace.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if not trace.arrived_at <= trace.first_token_at <= trace.completed_at:
            raise ValueError(f"trace {trace.identifier!r} has impossible timestamps")
        ttfts.append(trace.first_token_at - trace.arrived_at)
        e2es.append(trace.completed_at - trace.arrived_at)
        decoded_intervals = trace.output_tokens - 1
        tpots.append(
            (trace.completed_at - trace.first_token_at) / decoded_intervals
            if decoded_intervals
            else 0.0
        )

    p95_ttft = _nearest_rank(ttfts, 0.95)
    p95_tpot = _nearest_rank(tpots, 0.95)
    p95_e2e = _nearest_rank(e2es, 0.95)
    window = max(trace.completed_at for trace in traces) - min(
        trace.arrived_at for trace in traces
    )
    if window <= 0:
        raise ValueError("the observation window has no duration; throughput is undefined")
    throughput = sum(trace.output_tokens for trace in traces) / window

    violations: list[str] = []
    if p95_ttft > objectives.max_p95_ttft_seconds:
        violations.append("p95 TTFT exceeded")
    if p95_tpot > objectives.max_p95_tpot_seconds:
        violations.append("p95 TPOT exceeded")
    if p95_e2e > objectives.max_p95_e2e_seconds:
        violations.append("p95 end-to-end latency exceeded")
    if throughput < objectives.min_output_tokens_per_second:
        violations.append("output-token throughput below minimum")
    return ServiceReport(
        not violations,
        p95_ttft,
        p95_tpot,
        p95_e2e,
        throughput,
        tuple(violations),
    )


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]

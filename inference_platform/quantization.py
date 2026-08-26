"""Gate quantization with measured quality, memory, latency, and compatibility.

Bit width predicts weight storage approximately; it does not predict kernel support,
KV-cache size, end-to-end memory, quality, or speed. A format should therefore earn
staging through measurements taken on the intended hardware and workload rather than
through a claim such as "four bit means four times faster."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantizationAction(str, Enum):
    STAGE = "stage"
    REJECT = "reject"


@dataclass(frozen=True)
class QuantizationMeasurement:
    name: str
    weight_bits: int
    hardware_supported: bool
    quality_score: float
    weight_memory_gib: float
    p95_ttft_seconds: float
    output_tokens_per_second: float


@dataclass(frozen=True)
class QuantizationRequirements:
    min_quality_score: float
    max_quality_drop: float
    max_weight_memory_gib: float
    max_p95_ttft_seconds: float
    min_relative_throughput: float


@dataclass(frozen=True)
class QuantizationDecision:
    action: QuantizationAction
    quality_drop: float
    relative_throughput: float
    reasons: tuple[str, ...]


def evaluate_quantization(
    baseline: QuantizationMeasurement,
    candidate: QuantizationMeasurement,
    requirements: QuantizationRequirements,
) -> QuantizationDecision:
    """Decide whether a measured candidate is safe to advance to staging.

    Compatibility is checked first because unusable kernels make all performance
    numbers irrelevant. Quality must satisfy both an absolute floor and a maximum
    regression from the independently measured baseline. Memory, TTFT, and relative
    throughput then enforce the deployment goals. Every failed control is returned
    so a rejected format cannot masquerade as a generic benchmark loss.

    The function never estimates measurements from `weight_bits`; callers must bring
    results from the target runtime and hardware. Passing is permission to stage and
    load-test, not permission to promote directly to production.
    """

    _validate_measurement(baseline)
    _validate_measurement(candidate)
    if min(
        requirements.min_quality_score,
        requirements.max_quality_drop,
        requirements.max_weight_memory_gib,
        requirements.max_p95_ttft_seconds,
        requirements.min_relative_throughput,
    ) < 0:
        raise ValueError("quantization requirements may not be negative")

    quality_drop = baseline.quality_score - candidate.quality_score
    relative_throughput = (
        candidate.output_tokens_per_second / baseline.output_tokens_per_second
    )
    reasons: list[str] = []
    if not candidate.hardware_supported:
        reasons.append("format lacks target-hardware kernel support")
    if candidate.quality_score < requirements.min_quality_score:
        reasons.append("quality score below absolute floor")
    if quality_drop > requirements.max_quality_drop:
        reasons.append("quality regression exceeds allowance")
    if candidate.weight_memory_gib > requirements.max_weight_memory_gib:
        reasons.append("weight memory exceeds budget")
    if candidate.p95_ttft_seconds > requirements.max_p95_ttft_seconds:
        reasons.append("p95 TTFT exceeds objective")
    if relative_throughput < requirements.min_relative_throughput:
        reasons.append("relative throughput below requirement")
    return QuantizationDecision(
        QuantizationAction.REJECT if reasons else QuantizationAction.STAGE,
        quality_drop,
        relative_throughput,
        tuple(reasons) or ("all measured staging gates passed",),
    )


def _validate_measurement(measurement: QuantizationMeasurement) -> None:
    if not measurement.name.strip():
        raise ValueError("measurement name is required")
    if measurement.weight_bits <= 0 or measurement.weight_memory_gib <= 0:
        raise ValueError("weight precision and memory must be positive")
    if measurement.p95_ttft_seconds < 0 or measurement.output_tokens_per_second <= 0:
        raise ValueError("latency must be non-negative and throughput positive")

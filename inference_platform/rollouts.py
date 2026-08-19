"""Promote only a warmed canary that passes independent quality and SLO gates.

Shadow traffic can compare outputs without risking responses, but it does not prove
the user-facing routing and latency path. A canary exercises that path at limited
exposure. Neither deserves promotion until weights are warm, the sample is large
enough, and candidate measurements pass requirements declared before the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrafficMode(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"


class RolloutAction(str, Enum):
    HOLD = "hold"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ReleaseMeasurement:
    revision: str
    mode: TrafficMode
    warmed: bool
    sample_count: int
    quality_score: float
    p95_ttft_seconds: float
    p95_tpot_seconds: float
    output_tokens_per_second: float
    error_rate: float


@dataclass(frozen=True)
class ReleaseRequirements:
    min_sample_count: int
    min_quality_score: float
    max_quality_drop: float
    max_p95_ttft_seconds: float
    max_p95_tpot_seconds: float
    min_relative_throughput: float
    max_error_rate: float


@dataclass(frozen=True)
class RolloutDecision:
    action: RolloutAction
    deciding_controls: tuple[str, ...]


def evaluate_rollout(
    baseline: ReleaseMeasurement,
    candidate: ReleaseMeasurement,
    requirements: ReleaseRequirements,
) -> RolloutDecision:
    """Promote, hold, or roll back from measured evidence and fixed requirements.

    Readiness is evaluated first: an unwarmed or undersampled candidate is held, not
    punished as a regression. Once evidence is mature, quality, TTFT, TPOT, throughput,
    and errors are evaluated independently. Any failed safety/SLO gate rolls back and
    names every deciding control. A passing shadow is held because only canary traffic
    exercises the serving path; a passing canary is promoted.

    Requirements are not inferred from either measurement. The function assumes the
    baseline and candidate used equivalent workload slices; production rollout tooling
    must enforce that experimental design and automate traffic removal on rollback.
    """

    _validate_measurement(baseline)
    _validate_measurement(candidate)
    _validate_requirements(requirements)
    readiness: list[str] = []
    if not candidate.warmed:
        readiness.append("candidate has not completed model warmup")
    if candidate.sample_count < requirements.min_sample_count:
        readiness.append("candidate sample is below the independent minimum")
    if readiness:
        return RolloutDecision(RolloutAction.HOLD, tuple(readiness))

    failures: list[str] = []
    if candidate.quality_score < requirements.min_quality_score:
        failures.append("quality score below absolute floor")
    if baseline.quality_score - candidate.quality_score > requirements.max_quality_drop:
        failures.append("quality regression exceeds allowance")
    if candidate.p95_ttft_seconds > requirements.max_p95_ttft_seconds:
        failures.append("p95 TTFT exceeds rollout objective")
    if candidate.p95_tpot_seconds > requirements.max_p95_tpot_seconds:
        failures.append("p95 TPOT exceeds rollout objective")
    relative_throughput = (
        candidate.output_tokens_per_second / baseline.output_tokens_per_second
    )
    if relative_throughput < requirements.min_relative_throughput:
        failures.append("relative throughput below rollout objective")
    if candidate.error_rate > requirements.max_error_rate:
        failures.append("error rate exceeds rollout objective")
    if failures:
        return RolloutDecision(RolloutAction.ROLLBACK, tuple(failures))
    if candidate.mode is TrafficMode.SHADOW:
        return RolloutDecision(
            RolloutAction.HOLD,
            ("shadow passed, but user-facing canary evidence is still required",),
        )
    return RolloutDecision(
        RolloutAction.PROMOTE, ("warmed canary passed every independent release gate",)
    )


def _validate_measurement(measurement: ReleaseMeasurement) -> None:
    if not measurement.revision.strip() or measurement.sample_count < 0:
        raise ValueError("revision is required and sample count may not be negative")
    if measurement.output_tokens_per_second <= 0:
        raise ValueError("throughput must be positive")
    if min(
        measurement.quality_score,
        measurement.p95_ttft_seconds,
        measurement.p95_tpot_seconds,
        measurement.error_rate,
    ) < 0:
        raise ValueError("measurement values may not be negative")


def _validate_requirements(requirements: ReleaseRequirements) -> None:
    if requirements.min_sample_count <= 0:
        raise ValueError("minimum sample count must be positive")
    if min(
        requirements.min_quality_score,
        requirements.max_quality_drop,
        requirements.max_p95_ttft_seconds,
        requirements.max_p95_tpot_seconds,
        requirements.min_relative_throughput,
        requirements.max_error_rate,
    ) < 0:
        raise ValueError("release requirements may not be negative")

"""Capstone: turn measured demand and supply into auditable fleet-release evidence.

The inputs below stand in for versioned model metadata, a staging benchmark, current
accelerator inventory, an observed workload, and canary measurements. `run_plan`
passes them through the real course decisions: memory fit, capacity, parallel layout,
placement, admission, autoscaling, and rollout. The final gate compares resulting
evidence with requirements declared independently of those inputs.

Run twice to check determinism:

    python hands_on/plan_fleet.py
    python hands_on/plan_fleet.py

The command writes ``fleet-plan.json`` and exits nonzero when any required control
lacks evidence. This offline plan demonstrates reasoning; replace every supplied
benchmark, price, inventory fact, and measurement with observed production inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from inference_platform.admission import (
    AdmissionAction,
    AdmissionController,
    AdmissionPolicy,
    AdmissionRequest,
)
from inference_platform.autoscaling import (
    AutoscalingPolicy,
    ScaleObservation,
    ScaleState,
    recommend_replicas,
)
from inference_platform.capacity import (
    CapacityRequirements,
    ReplicaBenchmark,
    WorkloadClass,
    plan_capacity,
)
from inference_platform.memory import GIB, DeploymentMemory, ModelMemory, assess_memory
from inference_platform.parallelism import (
    ClusterTopology,
    ParallelModel,
    plan_parallelism,
)
from inference_platform.placement import GPU, PlacementRequest, place_replica
from inference_platform.rollouts import (
    ReleaseMeasurement,
    ReleaseRequirements,
    RolloutAction,
    TrafficMode,
    evaluate_rollout,
)


WORKLOADS = (
    WorkloadClass("interactive", 1.5, 400, 100, 2.0, 1.5),
    WorkloadClass("generation", 0.5, 200, 600, 5.0, 2.0),
)
BENCHMARK = ReplicaBenchmark("staging:model@abc:gpu-x", 1000, 350, 8, 4.25)
CAPACITY_REQUIREMENTS = CapacityRequirements(2, 8, 0.25, 40)
MODEL_MEMORY = ModelMemory(30_000_000_000, 16, 60, 8, 128)
DEPLOYMENT_MEMORY = DeploymentMemory(40, 2, 2, 4096, 6, 4, 0.9)
TOPOLOGY = ClusterTopology(2, 4, 40, 0.9, True)
INVENTORY = (
    GPU("a0", "node-a", 40, frozenset({"bf16", "fast-collective"}), frozenset({"model@abc"})),
    GPU("a1", "node-a", 40, frozenset({"bf16", "fast-collective"}), frozenset({"model@abc"})),
    GPU("a2", "node-a", 40, frozenset({"bf16", "fast-collective"})),
    GPU("a3", "node-a", 40, frozenset({"bf16", "fast-collective"})),
    GPU("b0", "node-b", 40, frozenset({"bf16", "fast-collective"})),
    GPU("b1", "node-b", 40, frozenset({"bf16", "fast-collective"})),
    GPU("b2", "node-b", 40, frozenset({"bf16", "fast-collective"})),
    GPU("b3", "node-b", 40, frozenset({"bf16", "fast-collective"})),
)
BASELINE_RELEASE = ReleaseMeasurement(
    "stable", TrafficMode.CANARY, True, 10_000, 0.90, 0.8, 0.10, 330, 0.001
)
CANDIDATE_RELEASE = ReleaseMeasurement(
    "model@abc", TrafficMode.CANARY, True, 1_000, 0.89, 0.8, 0.10, 350, 0.002
)
RELEASE_REQUIREMENTS = ReleaseRequirements(500, 0.86, 0.03, 1.0, 0.15, 0.95, 0.01)


# This is the platform claim, deliberately separate from the stimulus constants above.
# Removing a workload or bypassing a control removes evidence, not its requirement.
REQUIRED_EVIDENCE = frozenset(
    {
        "workload:interactive",
        "workload:generation",
        "memory-fit",
        "capacity-plan",
        "parallel-layout",
        "gpu-placement",
        "benign-admission",
        "oversize-shedding",
        "queue-scaling",
        "canary-gate",
    }
)


def run_plan(
    *,
    workloads: tuple[WorkloadClass, ...] = WORKLOADS,
    inventory: tuple[GPU, ...] = INVENTORY,
    candidate_release: ReleaseMeasurement = CANDIDATE_RELEASE,
) -> dict[str, object]:
    """Build a fleet decision and grade its evidence against fixed requirements.

    The observable workload is sized first. Model fit and topology then prove that
    the required replica count can exist on the cluster, after which concrete GPU
    placement, benign and adversarial admission, token-demand scaling, and a canary
    gate exercise their own boundaries. Evidence is added only when the corresponding
    decision actually succeeds and includes that decision's reason in the report.

    The three overridable inputs support counterfactual tests; none carries an expected
    result. Requirements remain module constants and the grader uses set difference,
    so deleting a case cannot delete the obligation it was meant to prove.
    """

    capacity = plan_capacity(workloads, BENCHMARK, CAPACITY_REQUIREMENTS)
    memory = assess_memory(MODEL_MEMORY, DEPLOYMENT_MEMORY)

    # Weight size is derived from the one model record rather than restated, and the
    # per-GPU reservation carries runtime overhead plus the concurrency-sized KV the
    # memory assessment just computed. Planning a layout against weights alone is the
    # "the weights fit" mistake this course exists to prevent.
    weight_gib = MODEL_MEMORY.parameters * MODEL_MEMORY.weight_bits / 8 / GIB
    parallel = plan_parallelism(
        ParallelModel(weight_gib, DEPLOYMENT_MEMORY.runtime_overhead_gib + memory.kv_gib_per_gpu),
        TOPOLOGY,
        target_replicas=capacity.required_replicas,
    )
    gpus_per_replica = (
        parallel.tensor_parallel * parallel.pipeline_parallel if parallel.feasible else 1
    )
    # The memory assessment sharded weights across a declared width. A layout that
    # spreads them differently invalidates the per-GPU number placement is about to
    # reserve, so the capstone checks the two agree instead of assuming it.
    layout_matches_memory_plan = (
        parallel.feasible and gpus_per_replica == DEPLOYMENT_MEMORY.tensor_parallel_size
    )
    placement = place_replica(
        PlacementRequest(
            "replica-1",
            "model@abc",
            gpus_per_replica,
            memory.required_gib_per_gpu if layout_matches_memory_plan else 40,
            frozenset({"bf16", "fast-collective"}),
        ),
        inventory,
    )

    admission = AdmissionController(AdmissionPolicy(8192, 4, 4096))
    benign = admission.decide(
        AdmissionRequest("interactive-request", 400, 100, deadline_at=5),
        now=0,
        estimated_queue_seconds=0.5,
    )
    oversize = admission.decide(
        AdmissionRequest("unbounded-request", 4500, 4500, deadline_at=30),
        now=0,
        estimated_queue_seconds=0,
    )

    scaling = recommend_replicas(
        (
            ScaleObservation(
                100,
                queued_tokens=0,
                arrival_tokens_per_second=capacity.output_tokens_per_second,
                cpu_utilization=0.15,
            ),
        ),
        ScaleState(capacity.required_replicas, capacity.required_replicas, 0, 0),
        AutoscalingPolicy(2, 8, 350, 0.75, 10, 5, 60),
        now=100,
    )
    rollout = evaluate_rollout(
        BASELINE_RELEASE, candidate_release, RELEASE_REQUIREMENTS
    )

    evidence = {f"workload:{workload.name}" for workload in workloads}
    reasons: dict[str, str] = {
        "memory-fit": memory.reason,
        "capacity-plan": (
            f"{capacity.bottleneck} sets {capacity.required_replicas} replicas; "
            f"binding dimensions: {', '.join(capacity.binding_dimensions)}"
        ),
        "parallel-layout": parallel.reason,
        "gpu-placement": placement.reason,
        "benign-admission": benign.reason,
        "oversize-shedding": oversize.reason,
        "queue-scaling": scaling.reason,
        "canary-gate": "; ".join(rollout.deciding_controls),
    }
    if memory.fits:
        evidence.add("memory-fit")
    if capacity.release_ready:
        evidence.add("capacity-plan")
    if layout_matches_memory_plan and parallel.data_parallel >= capacity.required_replicas:
        evidence.add("parallel-layout")
    if placement.placed:
        evidence.add("gpu-placement")
    if benign.action is AdmissionAction.ADMIT:
        evidence.add("benign-admission")
    if oversize.action is AdmissionAction.SHED and "token bound" in oversize.reason:
        evidence.add("oversize-shedding")
    if scaling.target_replicas >= capacity.required_replicas:
        evidence.add("queue-scaling")
    if rollout.action is RolloutAction.PROMOTE:
        evidence.add("canary-gate")

    missing = sorted(REQUIRED_EVIDENCE - evidence)
    violations = list(capacity.violations)
    if missing:
        violations.append(f"missing required evidence: {missing}")
    report: dict[str, object] = {
        "release_ready": not violations,
        "required_evidence": sorted(REQUIRED_EVIDENCE),
        "observed_evidence": sorted(evidence),
        "violations": violations,
        "decisions": {
            "memory": asdict(memory),
            "capacity": asdict(capacity),
            "parallelism": asdict(parallel),
            "placement": asdict(placement),
            "benign_admission": asdict(benign),
            "oversize_admission": asdict(oversize),
            "autoscaling": asdict(scaling),
            "rollout": asdict(rollout),
        },
        "deciding_reasons": reasons,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fleet-plan.json"))
    args = parser.parse_args(argv)
    report = run_plan()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

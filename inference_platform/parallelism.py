"""Choose parallel dimensions from model fit, topology, and replica demand.

Tensor parallelism splits layer operations and pays frequent collective-communication
cost. Pipeline parallelism splits layer ranges and pays stage bubbles. Data parallelism
duplicates a complete replica for concurrency. Expert parallelism distributes MoE
experts. These axes solve different constraints and should not be treated as a single
“use more GPUs” switch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ParallelModel:
    weight_memory_gib: float
    runtime_memory_gib_per_gpu: float
    experts: int = 1


@dataclass(frozen=True)
class ClusterTopology:
    nodes: int
    gpus_per_node: int
    gpu_memory_gib: float
    usable_fraction: float
    fast_intra_node_collectives: bool


@dataclass(frozen=True)
class ParallelPlan:
    feasible: bool
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    expert_parallel: int
    memory_gib_per_gpu: float
    reason: str


def plan_parallelism(
    model: ParallelModel, topology: ClusterTopology, target_replicas: int
) -> ParallelPlan:
    """Select the smallest feasible replica layout, then allocate data replicas.

    The planner enumerates tensor widths that stay within a node and divide its GPU
    count. Without a fast collective link it refuses tensor widths above one and
    grows pipeline stages instead. For each layout it checks per-GPU weight plus
    runtime memory against usable VRAM. It prefers fewer total GPUs; ties prefer
    tensor parallelism on fast links and pipeline parallelism otherwise. Remaining
    devices become complete data-parallel replicas up to the declared target.

    MoE expert parallelism is reported as the largest divisor of the expert count
    that fits inside one replica. This simplified placement model teaches the axes;
    a real plan must be validated against the runtime's supported layouts, layer
    divisibility, network bandwidth, and measured communication profile.
    """

    _validate(model, topology, target_replicas)
    total_gpus = topology.nodes * topology.gpus_per_node
    usable_gib = topology.gpu_memory_gib * topology.usable_fraction
    tensor_widths = [
        width
        for width in range(1, topology.gpus_per_node + 1)
        if topology.gpus_per_node % width == 0
        and (width == 1 or topology.fast_intra_node_collectives)
    ]
    candidates: list[tuple[int, int, float]] = []
    for tp in tensor_widths:
        for pp in range(1, total_gpus // tp + 1):
            memory = model.weight_memory_gib / (tp * pp)
            memory += model.runtime_memory_gib_per_gpu
            if memory <= usable_gib:
                candidates.append((tp, pp, memory))
    if not candidates:
        return ParallelPlan(
            False,
            0,
            0,
            0,
            0,
            math.inf,
            "no supported tensor/pipeline layout fits usable GPU memory",
        )

    def score(candidate: tuple[int, int, float]) -> tuple[int, int]:
        tp, pp, _ = candidate
        preference = pp if topology.fast_intra_node_collectives else tp
        return (tp * pp, preference)

    tp, pp, memory = min(candidates, key=score)
    gpus_per_replica = tp * pp
    available_replicas = total_gpus // gpus_per_replica
    dp = min(target_replicas, available_replicas)
    ep = _largest_divisor_at_most(model.experts, gpus_per_replica)
    if dp < target_replicas:
        reason = (
            f"layout fits, but cluster supplies {dp}/{target_replicas} requested replicas"
        )
    elif pp > 1 and tp == 1:
        reason = "pipeline split chosen because tensor collectives are unavailable or unnecessary"
    elif tp > 1:
        reason = "intra-node tensor split is the smallest fast-link layout that fits"
    else:
        reason = "one GPU per replica fits; remaining GPUs provide data parallelism"
    return ParallelPlan(True, tp, pp, dp, ep, memory, reason)


def _largest_divisor_at_most(value: int, maximum: int) -> int:
    return max(divisor for divisor in range(1, maximum + 1) if value % divisor == 0)


def _validate(model: ParallelModel, topology: ClusterTopology, target_replicas: int) -> None:
    if model.weight_memory_gib <= 0 or model.runtime_memory_gib_per_gpu < 0:
        raise ValueError("model memory must be positive with non-negative runtime memory")
    if model.experts <= 0:
        raise ValueError("expert count must be positive")
    if min(topology.nodes, topology.gpus_per_node, topology.gpu_memory_gib) <= 0:
        raise ValueError("cluster capacity must be positive")
    if not 0 < topology.usable_fraction <= 1:
        raise ValueError("usable_fraction must be in (0, 1]")
    if target_replicas <= 0:
        raise ValueError("target_replicas must be positive")

"""Place replicas on concrete accelerator groups, not an abstract GPU count.

A feasible group must satisfy per-device memory, accelerator capabilities, and the
communication locality required by the chosen parallel layout. Among feasible groups,
reusing resident weights avoids a cold load and packing onto the tightest fit preserves
roomier devices for later work. These priorities are explicit so fragmentation and
cold-start trade-offs can be changed deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class GPU:
    identifier: str
    node: str
    free_memory_gib: float
    capabilities: frozenset[str]
    resident_models: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PlacementRequest:
    replica: str
    model_revision: str
    gpus_required: int
    memory_gib_per_gpu: float
    required_capabilities: frozenset[str]
    require_same_node: bool = True


@dataclass(frozen=True)
class PlacementDecision:
    placed: bool
    gpu_ids: tuple[str, ...]
    reason: str


def place_replica(
    request: PlacementRequest, inventory: tuple[GPU, ...]
) -> PlacementDecision:
    """Choose the best eligible GPU group and report the deciding placement facts.

    Eligibility is evaluated from actual inventory: every GPU must have the requested
    memory and capabilities, and multi-GPU groups must share a node when locality is
    required. Feasible groups are ranked by most resident model copies, then least
    post-placement free memory, then stable GPU ids. That order makes cold-start
    avoidance the primary optimization while still packing capacity deterministically.

    The function is intentionally pure: orchestration must re-check inventory and
    reserve the chosen devices atomically because another scheduler may race between
    this plan and allocation. It does not infer bandwidth from a node name; callers
    should express requirements such as `fast-collective` as capabilities.
    """

    if not request.replica.strip() or not request.model_revision.strip():
        raise ValueError("replica and model revision are required")
    if request.gpus_required <= 0 or request.memory_gib_per_gpu <= 0:
        raise ValueError("GPU count and memory requirement must be positive")
    identifiers = [gpu.identifier for gpu in inventory]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("GPU identifiers must be unique")
    if any(gpu.free_memory_gib < 0 for gpu in inventory):
        raise ValueError("free GPU memory may not be negative")

    eligible = [
        gpu
        for gpu in inventory
        if gpu.free_memory_gib >= request.memory_gib_per_gpu
        and request.required_capabilities <= gpu.capabilities
    ]
    groups = []
    for group in combinations(eligible, request.gpus_required):
        if request.require_same_node and len({gpu.node for gpu in group}) != 1:
            continue
        resident = sum(
            request.model_revision in gpu.resident_models for gpu in group
        )
        remaining = sum(
            gpu.free_memory_gib - request.memory_gib_per_gpu for gpu in group
        )
        ids = tuple(sorted(gpu.identifier for gpu in group))
        groups.append((resident, remaining, ids))
    if not groups:
        return PlacementDecision(
            False,
            (),
            "no GPU group satisfies memory, capability, and locality constraints",
        )
    resident, remaining, ids = min(
        groups, key=lambda group: (-group[0], group[1], group[2])
    )
    locality = "same-node" if request.require_same_node else "cross-node permitted"
    reason = (
        f"selected {locality} group with {resident}/{request.gpus_required} "
        f"resident model copies and {remaining:.2f} GiB remaining"
    )
    return PlacementDecision(True, ids, reason)

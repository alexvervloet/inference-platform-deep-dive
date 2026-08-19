"""Account for weights, KV cache, and headroom before claiming a model fits.

Weight memory is mostly fixed for a chosen format, while KV-cache memory grows
with live tokens and concurrency. A useful fit check therefore reserves both,
plus explicit runtime headroom, against only the configured usable fraction of
each accelerator. The arithmetic is a planning bound, not a hardware benchmark:
real runtimes may add workspaces, allocator fragmentation, or replicated tensors.
"""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True)
class ModelMemory:
    parameters: int
    weight_bits: int
    layers: int
    kv_heads: int
    head_dim: int
    kv_element_bytes: int = 2


@dataclass(frozen=True)
class DeploymentMemory:
    gpu_memory_gib: float
    tensor_parallel_size: int
    kv_shards: int
    max_live_tokens_per_request: int
    target_concurrency: int
    runtime_overhead_gib: float
    usable_fraction: float = 0.90


@dataclass(frozen=True)
class MemoryAssessment:
    fits: bool
    weight_gib_per_gpu: float
    kv_gib_per_gpu: float
    overhead_gib_per_gpu: float
    required_gib_per_gpu: float
    usable_gib_per_gpu: float
    max_concurrency: int
    reason: str


def assess_memory(model: ModelMemory, deployment: DeploymentMemory) -> MemoryAssessment:
    """Decide whether weights and the target KV reservation fit on every GPU.

    The function first shards weight bytes by tensor parallelism, then shards KV
    bytes by the separately declared KV layout. It reserves the target's maximum
    live tokens for every concurrent request before adding runtime overhead. This
    order prevents weight fit from being mistaken for service fit.

    `kv_shards` is explicit because grouped-query attention and runtime layouts do
    not always shard KV state exactly like weights. The result is a conservative
    planner only when its inputs are conservative; confirm it with the runtime's
    observed cache capacity and a staging load test.
    """

    _validate(model, deployment)
    weight_bytes = model.parameters * model.weight_bits / 8
    weight_gib = weight_bytes / deployment.tensor_parallel_size / GIB
    kv_bytes_per_token = (
        2
        * model.layers
        * model.kv_heads
        * model.head_dim
        * model.kv_element_bytes
    )
    kv_gib_per_request = kv_bytes_per_token / deployment.kv_shards / GIB
    kv_gib_per_request *= deployment.max_live_tokens_per_request
    usable_gib = deployment.gpu_memory_gib * deployment.usable_fraction
    fixed_gib = weight_gib + deployment.runtime_overhead_gib
    available_for_kv = max(0.0, usable_gib - fixed_gib)
    max_concurrency = int(available_for_kv // kv_gib_per_request)
    kv_gib = kv_gib_per_request * deployment.target_concurrency
    required_gib = fixed_gib + kv_gib
    fits = required_gib <= usable_gib
    reason = (
        f"fits with {usable_gib - required_gib:.2f} GiB headroom per GPU"
        if fits
        else f"requires {required_gib - usable_gib:.2f} GiB more per GPU"
    )
    return MemoryAssessment(
        fits,
        weight_gib,
        kv_gib,
        deployment.runtime_overhead_gib,
        required_gib,
        usable_gib,
        max_concurrency,
        reason,
    )


def _validate(model: ModelMemory, deployment: DeploymentMemory) -> None:
    if min(
        model.parameters,
        model.weight_bits,
        model.layers,
        model.kv_heads,
        model.head_dim,
        model.kv_element_bytes,
    ) <= 0:
        raise ValueError("all model memory fields must be positive")
    if min(
        deployment.gpu_memory_gib,
        deployment.tensor_parallel_size,
        deployment.kv_shards,
        deployment.max_live_tokens_per_request,
        deployment.target_concurrency,
    ) <= 0:
        raise ValueError("deployment capacity fields must be positive")
    if deployment.runtime_overhead_gib < 0:
        raise ValueError("runtime overhead may not be negative")
    if not 0 < deployment.usable_fraction <= 1:
        raise ValueError("usable_fraction must be in (0, 1]")

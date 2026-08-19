"""Lesson 1: model fit includes a concurrency-sized KV-cache reservation."""

from inference_platform.memory import DeploymentMemory, ModelMemory, assess_memory


model = ModelMemory(
    parameters=7_000_000_000,
    weight_bits=4,
    layers=32,
    kv_heads=8,
    head_dim=128,
)
deployment = DeploymentMemory(
    gpu_memory_gib=24,
    tensor_parallel_size=1,
    kv_shards=1,
    max_live_tokens_per_request=8192,
    target_concurrency=8,
    runtime_overhead_gib=2,
)
assessment = assess_memory(model, deployment)

print(f"weights/GPU:  {assessment.weight_gib_per_gpu:.2f} GiB")
print(f"KV cache/GPU: {assessment.kv_gib_per_gpu:.2f} GiB")
print(f"required/GPU: {assessment.required_gib_per_gpu:.2f} GiB")
print(f"maximum concurrency under this bound: {assessment.max_concurrency}")
print(f"decision: {'FIT' if assessment.fits else 'NO FIT'} — {assessment.reason}")

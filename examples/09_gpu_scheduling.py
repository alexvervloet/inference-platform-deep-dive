"""Lesson 9: residency, capabilities, memory, and locality decide GPU placement."""

from inference_platform.placement import GPU, PlacementRequest, place_replica


capabilities = frozenset({"bf16", "fast-collective"})
inventory = (
    GPU("a0", "node-a", 40, capabilities),
    GPU("a1", "node-a", 40, capabilities),
    GPU("b0", "node-b", 40, capabilities, frozenset({"model@abc"})),
    GPU("b1", "node-b", 40, capabilities, frozenset({"model@abc"})),
)
request = PlacementRequest(
    "replica-2", "model@abc", 2, 30, capabilities, require_same_node=True
)
decision = place_replica(request, inventory)

print(f"placed={decision.placed}, GPUs={','.join(decision.gpu_ids) or 'none'}")
print(decision.reason)

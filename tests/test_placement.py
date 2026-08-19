from __future__ import annotations

import unittest

from inference_platform.placement import GPU, PlacementRequest, place_replica


REQUEST = PlacementRequest(
    "replica-1", "model@abc", 2, 30, frozenset({"bf16", "fast-collective"})
)


class PlacementTests(unittest.TestCase):
    def test_prefers_an_eligible_group_with_resident_weights(self) -> None:
        inventory = (
            GPU("a0", "node-a", 40, REQUEST.required_capabilities),
            GPU("a1", "node-a", 40, REQUEST.required_capabilities),
            GPU("b0", "node-b", 40, REQUEST.required_capabilities, frozenset({"model@abc"})),
            GPU("b1", "node-b", 40, REQUEST.required_capabilities, frozenset({"model@abc"})),
        )
        decision = place_replica(REQUEST, inventory)
        self.assertTrue(decision.placed)
        self.assertEqual(decision.gpu_ids, ("b0", "b1"))
        self.assertIn("2/2 resident", decision.reason)

    def test_free_memory_without_required_capability_is_not_eligible(self) -> None:
        inventory = (
            GPU("a0", "node-a", 80, frozenset({"bf16"})),
            GPU("a1", "node-a", 80, frozenset({"bf16"})),
        )
        decision = place_replica(REQUEST, inventory)
        self.assertFalse(decision.placed)
        self.assertIn("capability", decision.reason)

    def test_same_node_constraint_rejects_scattered_free_gpus(self) -> None:
        inventory = (
            GPU("a0", "node-a", 40, REQUEST.required_capabilities),
            GPU("b0", "node-b", 40, REQUEST.required_capabilities),
        )
        self.assertFalse(place_replica(REQUEST, inventory).placed)

    def test_tightest_fit_preserves_larger_eligible_devices(self) -> None:
        single = PlacementRequest("r", "m", 1, 30, frozenset({"bf16"}))
        inventory = (
            GPU("large", "n", 80, frozenset({"bf16"})),
            GPU("tight", "n", 40, frozenset({"bf16"})),
        )
        self.assertEqual(place_replica(single, inventory).gpu_ids, ("tight",))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from inference_platform.memory import (
    GIB,
    DeploymentMemory,
    ModelMemory,
    assess_memory,
)


class MemoryAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ModelMemory(
            parameters=GIB,
            weight_bits=8,
            layers=1,
            kv_heads=1,
            head_dim=1,
            kv_element_bytes=1,
        )

    def test_accounts_for_weight_kv_and_runtime_memory_separately(self) -> None:
        result = assess_memory(
            self.model,
            DeploymentMemory(2, 1, 1, GIB // 4, 1, 0.25, usable_fraction=1),
        )
        self.assertEqual(result.weight_gib_per_gpu, 1)
        self.assertEqual(result.kv_gib_per_gpu, 0.5)
        self.assertEqual(result.required_gib_per_gpu, 1.75)
        self.assertTrue(result.fits)
        self.assertIn("headroom", result.reason)

    def test_context_growth_can_flip_weight_fit_to_service_no_fit(self) -> None:
        short = assess_memory(
            self.model,
            DeploymentMemory(2, 1, 1, GIB // 8, 1, 0.25, usable_fraction=1),
        )
        long = assess_memory(
            self.model,
            DeploymentMemory(2, 1, 1, GIB // 2, 1, 0.25, usable_fraction=1),
        )
        self.assertTrue(short.fits)
        self.assertFalse(long.fits)
        self.assertIn("more per GPU", long.reason)

    def test_tensor_and_kv_sharding_are_independent_inputs(self) -> None:
        unsharded_kv = assess_memory(
            self.model,
            DeploymentMemory(1, 2, 1, GIB // 4, 2, 0, usable_fraction=1),
        )
        sharded_kv = assess_memory(
            self.model,
            DeploymentMemory(1, 2, 2, GIB // 4, 2, 0, usable_fraction=1),
        )
        self.assertFalse(unsharded_kv.fits)
        self.assertTrue(sharded_kv.fits)

    def test_rejects_zero_or_negative_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity fields"):
            assess_memory(
                self.model,
                DeploymentMemory(0, 1, 1, 1, 1, 0),
            )


if __name__ == "__main__":
    unittest.main()

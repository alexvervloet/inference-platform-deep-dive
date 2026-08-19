from __future__ import annotations

import unittest

from inference_platform.parallelism import (
    ClusterTopology,
    ParallelModel,
    plan_parallelism,
)


class ParallelismTests(unittest.TestCase):
    def test_fast_intra_node_link_prefers_tensor_parallel_fit(self) -> None:
        plan = plan_parallelism(
            ParallelModel(60, 4),
            ClusterTopology(2, 4, 40, 0.9, True),
            target_replicas=2,
        )
        self.assertTrue(plan.feasible)
        self.assertEqual((plan.tensor_parallel, plan.pipeline_parallel), (2, 1))
        self.assertEqual(plan.data_parallel, 2)
        self.assertIn("fast-link", plan.reason)

    def test_without_fast_collectives_pipeline_parallelism_carries_the_split(self) -> None:
        plan = plan_parallelism(
            ParallelModel(60, 4),
            ClusterTopology(1, 4, 40, 0.9, False),
            target_replicas=1,
        )
        self.assertEqual((plan.tensor_parallel, plan.pipeline_parallel), (1, 2))
        self.assertIn("pipeline split", plan.reason)

    def test_moe_plan_reports_an_expert_partition_that_divides_experts(self) -> None:
        plan = plan_parallelism(
            ParallelModel(100, 4, experts=8),
            ClusterTopology(1, 4, 40, 0.9, True),
            target_replicas=1,
        )
        self.assertEqual(plan.tensor_parallel, 4)
        self.assertEqual(plan.expert_parallel, 4)

    def test_reports_replica_shortfall_instead_of_claiming_target_capacity(self) -> None:
        plan = plan_parallelism(
            ParallelModel(60, 4),
            ClusterTopology(1, 2, 40, 0.9, True),
            target_replicas=2,
        )
        self.assertEqual(plan.data_parallel, 1)
        self.assertIn("1/2", plan.reason)


if __name__ == "__main__":
    unittest.main()

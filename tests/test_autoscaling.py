from __future__ import annotations

import unittest

from inference_platform.autoscaling import (
    AutoscalingPolicy,
    ScaleObservation,
    ScaleState,
    recommend_replicas,
)


POLICY = AutoscalingPolicy(1, 10, 100, 0.8, 10, 5, 60)


class AutoscalingTests(unittest.TestCase):
    def test_arrival_rate_and_backlog_both_contribute_to_scale_up(self) -> None:
        decision = recommend_replicas(
            (ScaleObservation(100, 800, 160, 0.1),),
            ScaleState(1, 1, 0, 0),
            POLICY,
            now=100,
        )
        self.assertEqual(decision.calculated_demand_replicas, 3)
        self.assertEqual(decision.target_replicas, 3)
        self.assertIn("scale-up", decision.reason)

    def test_cpu_signal_does_not_manufacture_token_demand(self) -> None:
        decision = recommend_replicas(
            (ScaleObservation(100, 0, 0, 1.0),),
            ScaleState(1, 1, 0, 0),
            POLICY,
            now=100,
        )
        self.assertEqual(decision.target_replicas, 1)

    def test_warming_replicas_are_not_counted_as_ready_throughput(self) -> None:
        decision = recommend_replicas(
            (ScaleObservation(100, 0, 160, 0.5),),
            ScaleState(2, 1, 1, 0),
            POLICY,
            now=100,
        )
        self.assertEqual(decision.target_replicas, 2)
        self.assertEqual(decision.ready_capacity_tokens_per_second, 80)

    def test_scale_down_requires_a_complete_stable_window(self) -> None:
        complete = recommend_replicas(
            (ScaleObservation(0, 0, 20, 0.1), ScaleObservation(60, 0, 20, 0.1)),
            ScaleState(3, 3, 0, 0),
            POLICY,
            now=60,
        )
        incomplete = recommend_replicas(
            (ScaleObservation(30, 0, 20, 0.1), ScaleObservation(60, 0, 20, 0.1)),
            ScaleState(3, 3, 0, 0),
            POLICY,
            now=60,
        )
        self.assertEqual(complete.target_replicas, 1)
        self.assertEqual(incomplete.target_replicas, 3)


if __name__ == "__main__":
    unittest.main()

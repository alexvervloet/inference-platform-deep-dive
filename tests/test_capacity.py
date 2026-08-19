from __future__ import annotations

from dataclasses import replace
import unittest

from inference_platform.capacity import (
    CapacityRequirements,
    ReplicaBenchmark,
    WorkloadClass,
    plan_capacity,
)


BENCHMARK = ReplicaBenchmark("measured-gpu", 200, 200, 10, 10)
REQUIREMENTS = CapacityRequirements(1, 10, 0.2, 100)


class CapacityPlanningTests(unittest.TestCase):
    def test_burst_multiplier_changes_required_capacity(self) -> None:
        average = WorkloadClass("chat", 1, 100, 100, 1, 1)
        burst = replace(average, burst_multiplier=2)
        self.assertEqual(plan_capacity((average,), BENCHMARK, REQUIREMENTS).required_replicas, 1)
        self.assertEqual(plan_capacity((burst,), BENCHMARK, REQUIREMENTS).required_replicas, 2)

    def test_rounding_tie_reports_every_binding_dimension(self) -> None:
        plan = plan_capacity(
            (WorkloadClass("chat", 1, 100, 100, 1, 1),),
            BENCHMARK,
            REQUIREMENTS,
        )
        self.assertEqual(plan.required_replicas, 1)
        self.assertEqual(
            plan.binding_dimensions,
            ("decode tokens", "minimum replica floor", "prefill tokens", "request concurrency"),
        )

    def test_replica_floor_is_named_when_no_workload_dimension_exceeds_it(self) -> None:
        plan = plan_capacity(
            (WorkloadClass("trickle", 0.1, 10, 10, 0.5, 1),),
            BENCHMARK,
            replace(REQUIREMENTS, min_replicas=3),
        )
        self.assertEqual(plan.required_replicas, 3)
        self.assertEqual(plan.bottleneck, "minimum replica floor")
        self.assertEqual(plan.binding_dimensions, ("minimum replica floor",))

    def test_workload_mix_identifies_decode_as_the_actual_bottleneck(self) -> None:
        plan = plan_capacity(
            (
                WorkloadClass("short-answer", 1, 100, 10, 1, 1),
                WorkloadClass("generation", 1, 10, 500, 3, 1),
            ),
            BENCHMARK,
            REQUIREMENTS,
        )
        self.assertEqual(plan.bottleneck, "decode tokens")
        self.assertEqual(plan.binding_dimensions, ("decode tokens",))
        self.assertEqual(plan.required_replicas, 4)

    def test_budget_failure_does_not_shrink_the_required_fleet(self) -> None:
        workload = WorkloadClass("chat", 1, 100, 100, 1, 2)
        plan = plan_capacity(
            (workload,), BENCHMARK, replace(REQUIREMENTS, max_hourly_cost=15)
        )
        self.assertEqual(plan.required_replicas, 2)
        self.assertEqual(plan.hourly_cost, 20)
        self.assertFalse(plan.release_ready)
        self.assertIn("hourly cost budget", plan.violations[0])

    def test_supplied_price_changes_cost_but_not_capacity(self) -> None:
        workload = WorkloadClass("chat", 1, 100, 100, 1, 1)
        cheap = plan_capacity((workload,), BENCHMARK, REQUIREMENTS)
        expensive = plan_capacity(
            (workload,), replace(BENCHMARK, hourly_cost=20), REQUIREMENTS
        )
        self.assertEqual(cheap.required_replicas, expensive.required_replicas)
        self.assertEqual(expensive.hourly_cost, 2 * cheap.hourly_cost)


if __name__ == "__main__":
    unittest.main()

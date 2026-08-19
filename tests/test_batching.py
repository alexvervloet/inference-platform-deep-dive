from __future__ import annotations

import unittest

from inference_platform.batching import (
    BatchMode,
    BatchRequest,
    simulate_batching,
)


WORKLOAD = (
    BatchRequest("short", 0, 1),
    BatchRequest("long", 0, 3),
    BatchRequest("late", 1, 1),
)


class BatchingTests(unittest.TestCase):
    def test_continuous_batching_refills_the_lane_freed_by_short_work(self) -> None:
        run = simulate_batching(WORKLOAD, 2, BatchMode.CONTINUOUS)
        completions = {item.identifier: item for item in run.completions}
        self.assertEqual(completions["late"].first_service_step, 1)
        self.assertEqual(completions["late"].completion_step, 2)
        self.assertEqual(run.total_wait_steps, 0)

    def test_static_batching_exhibits_head_of_line_blocking(self) -> None:
        run = simulate_batching(WORKLOAD, 2, BatchMode.STATIC)
        completions = {item.identifier: item for item in run.completions}
        self.assertEqual(completions["late"].first_service_step, 3)
        self.assertEqual(completions["late"].completion_step, 4)
        self.assertEqual(run.total_wait_steps, 2)

    def test_same_stimulus_produces_a_shorter_continuous_makespan(self) -> None:
        static = simulate_batching(WORKLOAD, 2, BatchMode.STATIC)
        continuous = simulate_batching(WORKLOAD, 2, BatchMode.CONTINUOUS)
        self.assertLess(continuous.makespan_steps, static.makespan_steps)

    def test_makespan_is_a_span_not_an_absolute_step(self) -> None:
        shifted = tuple(
            BatchRequest(item.identifier, item.arrival_step + 100, item.output_tokens)
            for item in WORKLOAD
        )
        for mode in BatchMode:
            with self.subTest(mode=mode):
                self.assertEqual(
                    simulate_batching(shifted, 2, mode).makespan_steps,
                    simulate_batching(WORKLOAD, 2, mode).makespan_steps,
                )

    def test_rejects_duplicate_identifiers_that_would_overwrite_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_batching(
                (BatchRequest("same", 0, 1), BatchRequest("same", 1, 1)),
                1,
                BatchMode.CONTINUOUS,
            )


if __name__ == "__main__":
    unittest.main()

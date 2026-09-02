from __future__ import annotations

import unittest

from inference_platform.metrics import (
    RequestTrace,
    ServiceObjectives,
    evaluate_service,
)


OBJECTIVES = ServiceObjectives(1.0, 0.2, 3.0, 2.0)


class ServiceMetricTests(unittest.TestCase):
    def test_known_trace_separates_ttft_tpot_e2e_and_throughput(self) -> None:
        report = evaluate_service(
            (RequestTrace("r1", 0.0, 1.0, 3.0, 11),),
            ServiceObjectives(1.0, 0.2, 3.0, 11 / 3),
        )
        self.assertEqual(report.p95_ttft_seconds, 1.0)
        self.assertEqual(report.p95_tpot_seconds, 0.2)
        self.assertEqual(report.p95_e2e_seconds, 3.0)
        self.assertAlmostEqual(report.output_tokens_per_second, 11 / 3)
        self.assertTrue(report.meets_objectives)

    def test_slow_first_token_breaks_ttft_without_breaking_tpot(self) -> None:
        report = evaluate_service(
            (RequestTrace("r1", 0.0, 1.5, 2.4, 10),), OBJECTIVES
        )
        self.assertEqual(report.violations, ("p95 TTFT exceeded",))
        self.assertLessEqual(report.p95_tpot_seconds, OBJECTIVES.max_p95_tpot_seconds)

    def test_throughput_uses_the_shared_observation_window(self) -> None:
        report = evaluate_service(
            (
                RequestTrace("r1", 0, 0.2, 1, 5),
                RequestTrace("r2", 1, 1.2, 2, 5),
            ),
            ServiceObjectives(1, 1, 3, 6),
        )
        self.assertEqual(report.output_tokens_per_second, 5)
        self.assertIn("output-token throughput below minimum", report.violations)

    def test_rejects_impossible_timestamps_instead_of_emitting_negative_latency(self) -> None:
        with self.assertRaisesRegex(ValueError, "impossible timestamps"):
            evaluate_service((RequestTrace("r1", 1, 0, 2, 2),), OBJECTIVES)

    def test_zero_length_window_is_rejected_rather_than_passing_every_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "no duration"):
            evaluate_service(
                (RequestTrace("r1", 5.0, 5.0, 5.0, 3),),
                ServiceObjectives(1, 1, 1, 1_000_000),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import replace
import unittest

from inference_platform.rollouts import (
    ReleaseMeasurement,
    ReleaseRequirements,
    RolloutAction,
    TrafficMode,
    evaluate_rollout,
)


BASELINE = ReleaseMeasurement("stable", TrafficMode.CANARY, True, 10_000, 0.90, 0.8, 0.1, 100, 0.001)
CANDIDATE = ReleaseMeasurement("candidate", TrafficMode.CANARY, True, 1_000, 0.89, 0.8, 0.1, 105, 0.002)
REQUIREMENTS = ReleaseRequirements(500, 0.85, 0.03, 1.0, 0.15, 0.95, 0.01)


class RolloutTests(unittest.TestCase):
    def test_warmed_canary_promotes_after_all_independent_gates_pass(self) -> None:
        decision = evaluate_rollout(BASELINE, CANDIDATE, REQUIREMENTS)
        self.assertEqual(decision.action, RolloutAction.PROMOTE)
        self.assertIn("every independent release gate", decision.deciding_controls[0])

    def test_unwarmed_candidate_holds_before_regression_evaluation(self) -> None:
        candidate = replace(CANDIDATE, warmed=False, quality_score=0.1)
        decision = evaluate_rollout(BASELINE, candidate, REQUIREMENTS)
        self.assertEqual(decision.action, RolloutAction.HOLD)
        self.assertEqual(decision.deciding_controls, ("candidate has not completed model warmup",))

    def test_passing_shadow_cannot_prove_the_user_traffic_path(self) -> None:
        decision = evaluate_rollout(
            BASELINE, replace(CANDIDATE, mode=TrafficMode.SHADOW), REQUIREMENTS
        )
        self.assertEqual(decision.action, RolloutAction.HOLD)
        self.assertIn("canary evidence", decision.deciding_controls[0])

    def test_one_tail_latency_regression_flips_the_rollout(self) -> None:
        decision = evaluate_rollout(
            BASELINE, replace(CANDIDATE, p95_tpot_seconds=0.2), REQUIREMENTS
        )
        self.assertEqual(decision.action, RolloutAction.ROLLBACK)
        self.assertIn("p95 TPOT exceeds rollout objective", decision.deciding_controls)

    def test_threshold_does_not_come_from_candidate_measurement(self) -> None:
        strict = replace(REQUIREMENTS, max_error_rate=0.001)
        decision = evaluate_rollout(BASELINE, CANDIDATE, strict)
        self.assertEqual(decision.action, RolloutAction.ROLLBACK)
        self.assertIn("error rate exceeds rollout objective", decision.deciding_controls)


if __name__ == "__main__":
    unittest.main()

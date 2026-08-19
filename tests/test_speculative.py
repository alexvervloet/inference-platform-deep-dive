from __future__ import annotations

import unittest

from inference_platform.speculative import SpeculationCosts, evaluate_speculation


COSTS = SpeculationCosts(
    baseline_seconds_per_token=1.0,
    draft_seconds_per_token=0.1,
    verification_seconds_per_round=1.0,
)


class SpeculativeDecodingTests(unittest.TestCase):
    def test_high_actual_acceptance_can_repay_a_verification_round(self) -> None:
        decision = evaluate_speculation((1, 2, 3, 4), (1, 2, 3, 4), COSTS, 2.0)
        self.assertTrue(decision.enable)
        self.assertEqual(decision.accepted_tokens, 4)
        self.assertAlmostEqual(decision.estimated_speedup, 5 / 1.4)

    def test_fully_accepted_round_emits_the_extra_verified_position(self) -> None:
        decision = evaluate_speculation((1, 2, 3, 4), (1, 2, 3, 4), COSTS)
        self.assertEqual(decision.accepted_tokens, 4)
        self.assertEqual(decision.drafted_tokens, 4)
        self.assertEqual(decision.emitted_tokens, 5)
        self.assertEqual(decision.acceptance_rate, 1.0)

    def test_low_acceptance_can_make_speculation_slower(self) -> None:
        decision = evaluate_speculation((9, 8, 7, 6), (1, 2, 3, 4), COSTS)
        self.assertFalse(decision.enable)
        self.assertEqual(decision.accepted_tokens, 0)
        self.assertEqual(decision.emitted_tokens, 1)
        self.assertLess(decision.estimated_speedup, 1)

    def test_first_mismatch_emits_verified_target_token_after_accepted_prefix(self) -> None:
        decision = evaluate_speculation((1, 2, 9, 9), (1, 2, 3, 4), COSTS)
        self.assertEqual(decision.accepted_tokens, 2)
        self.assertEqual(decision.emitted_tokens, 3)
        self.assertEqual(decision.acceptance_rate, 0.5)

    def test_speedup_gate_is_independent_of_token_inputs(self) -> None:
        lenient = evaluate_speculation((1, 2), (1, 2), COSTS, minimum_speedup=2.0)
        strict = evaluate_speculation((1, 2), (1, 2), COSTS, minimum_speedup=3.0)
        self.assertAlmostEqual(lenient.estimated_speedup, strict.estimated_speedup)
        self.assertTrue(lenient.enable)
        self.assertFalse(strict.enable)
        self.assertIn("misses 3.00x", strict.reason)


if __name__ == "__main__":
    unittest.main()

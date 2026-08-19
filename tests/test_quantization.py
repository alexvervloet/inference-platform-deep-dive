from __future__ import annotations

import unittest

from inference_platform.quantization import (
    QuantizationAction,
    QuantizationMeasurement,
    QuantizationRequirements,
    evaluate_quantization,
)


BASELINE = QuantizationMeasurement("bf16", 16, True, 0.90, 28, 0.8, 100)
REQUIREMENTS = QuantizationRequirements(0.85, 0.03, 16, 0.9, 1.0)


class QuantizationTests(unittest.TestCase):
    def test_measured_candidate_can_advance_to_staging(self) -> None:
        candidate = QuantizationMeasurement("int8", 8, True, 0.88, 15, 0.7, 120)
        decision = evaluate_quantization(BASELINE, candidate, REQUIREMENTS)
        self.assertEqual(decision.action, QuantizationAction.STAGE)
        self.assertAlmostEqual(decision.quality_drop, 0.02)
        self.assertEqual(decision.relative_throughput, 1.2)

    def test_low_bit_width_does_not_manufacture_a_speedup(self) -> None:
        candidate = QuantizationMeasurement("int4", 4, True, 0.88, 8, 0.7, 70)
        decision = evaluate_quantization(BASELINE, candidate, REQUIREMENTS)
        self.assertEqual(decision.action, QuantizationAction.REJECT)
        self.assertIn("relative throughput below requirement", decision.reasons)
        self.assertEqual(decision.relative_throughput, 0.7)

    def test_hardware_support_is_an_independent_gate(self) -> None:
        candidate = QuantizationMeasurement("format-x", 4, False, 0.90, 8, 0.5, 150)
        decision = evaluate_quantization(BASELINE, candidate, REQUIREMENTS)
        self.assertIn("target-hardware kernel support", decision.reasons[0])

    def test_quality_regression_is_compared_with_a_separate_baseline(self) -> None:
        candidate = QuantizationMeasurement("int8", 8, True, 0.86, 15, 0.7, 120)
        decision = evaluate_quantization(BASELINE, candidate, REQUIREMENTS)
        self.assertIn("quality regression exceeds allowance", decision.reasons)


if __name__ == "__main__":
    unittest.main()

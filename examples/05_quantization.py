"""Lesson 5: a smaller format advances only on measured quality and serving evidence."""

from inference_platform.quantization import (
    QuantizationMeasurement,
    QuantizationRequirements,
    evaluate_quantization,
)


baseline = QuantizationMeasurement("bf16", 16, True, 0.91, 28, 0.9, 100)
candidate = QuantizationMeasurement("int4", 4, True, 0.88, 8, 0.8, 85)
requirements = QuantizationRequirements(
    min_quality_score=0.86,
    max_quality_drop=0.04,
    max_weight_memory_gib=12,
    max_p95_ttft_seconds=1.0,
    min_relative_throughput=1.0,
)
decision = evaluate_quantization(baseline, candidate, requirements)

print(f"candidate: {candidate.name} ({candidate.weight_bits}-bit)")
print(f"measured relative throughput: {decision.relative_throughput:.2f}x")
print(f"decision: {decision.action.value}")
for reason in decision.reasons:
    print(f"  - {reason}")

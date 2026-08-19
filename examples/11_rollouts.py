"""Lesson 11: shadow evidence can hold; only a warmed, passing canary promotes."""

from dataclasses import replace

from inference_platform.rollouts import (
    ReleaseMeasurement,
    ReleaseRequirements,
    TrafficMode,
    evaluate_rollout,
)


baseline = ReleaseMeasurement("stable", TrafficMode.CANARY, True, 10_000, 0.90, 0.8, 0.1, 100, 0.001)
candidate = ReleaseMeasurement("candidate", TrafficMode.SHADOW, True, 1_000, 0.89, 0.8, 0.1, 105, 0.002)
requirements = ReleaseRequirements(500, 0.85, 0.03, 1.0, 0.15, 0.95, 0.01)

for measurement in (candidate, replace(candidate, mode=TrafficMode.CANARY)):
    decision = evaluate_rollout(baseline, measurement, requirements)
    print(f"{measurement.mode.value}: {decision.action.value}")
    for control in decision.deciding_controls:
        print(f"  - {control}")

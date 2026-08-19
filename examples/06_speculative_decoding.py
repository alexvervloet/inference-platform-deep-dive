"""Lesson 6: speculation is valuable only when real draft acceptance repays its work."""

from inference_platform.speculative import SpeculationCosts, evaluate_speculation


costs = SpeculationCosts(
    baseline_seconds_per_token=1.0,
    draft_seconds_per_token=0.1,
    verification_seconds_per_round=1.0,
)
rounds = {
    "good draft": ((1, 2, 3, 4), (1, 2, 3, 4)),
    "poor draft": ((9, 8, 7, 6), (1, 2, 3, 4)),
}

for name, (draft, target) in rounds.items():
    decision = evaluate_speculation(draft, target, costs, minimum_speedup=1.2)
    print(
        f"{name}: acceptance={decision.acceptance_rate:.0%}, "
        f"speedup={decision.estimated_speedup:.2f}x, enable={decision.enable}"
    )

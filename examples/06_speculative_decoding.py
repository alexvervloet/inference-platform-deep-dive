"""Lesson 6: speculation is valuable only when real draft acceptance repays its work."""

from inference_platform.speculative import SpeculationCosts, evaluate_speculation


costs = SpeculationCosts(
    baseline_seconds_per_token=1.0,
    draft_seconds_per_token=0.1,
    verification_seconds_per_round=1.0,
)
# Each target sequence runs one position past the draft. That fifth entry is what
# the verification pass covers after the last proposal, and a fully accepted round
# emits it as a bonus token.
rounds = {
    "good draft": ((1, 2, 3, 4), (1, 2, 3, 4, 5)),
    "poor draft": ((9, 8, 7, 6), (1, 2, 3, 4, 5)),
}

for name, (draft, target) in rounds.items():
    decision = evaluate_speculation(draft, target, costs, minimum_speedup=1.2)
    print(
        f"{name}: acceptance={decision.acceptance_rate:.0%}, "
        f"drafted={decision.drafted_tokens}, emitted={decision.emitted_tokens}, "
        f"speedup={decision.estimated_speedup:.2f}x, enable={decision.enable}"
    )

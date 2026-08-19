"""Measure whether draft-token acceptance repays speculative-decoding overhead.

Speculative decoding proposes several cheap draft tokens, verifies them with the
target model in parallel, accepts the matching prefix, and uses the target token at
the first mismatch. Exact algorithms preserve the target distribution; performance
still depends on acceptance and the relative cost of drafting and verification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculationCosts:
    baseline_seconds_per_token: float
    draft_seconds_per_token: float
    verification_seconds_per_round: float


@dataclass(frozen=True)
class SpeculationDecision:
    enable: bool
    accepted_tokens: int
    drafted_tokens: int
    emitted_tokens: int
    acceptance_rate: float
    estimated_speedup: float
    reason: str


def evaluate_speculation(
    draft_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
    costs: SpeculationCosts,
    minimum_speedup: float = 1.0,
) -> SpeculationDecision:
    """Decide from actual agreement and measured costs whether speculation helps.

    Agreement is computed token by token until the first mismatch; inputs cannot
    carry an expected acceptance label. A mismatch still emits the target model's
    token, so one verification round emits the accepted prefix plus one correction.
    Full agreement emits the entire draft. The baseline cost covers the same number
    of emitted tokens, while speculative cost includes every draft token and one
    verification round.

    The cost model isolates a single round and ignores batching interactions. Enable
    only after measuring these costs and the acceptance distribution for the actual
    draft/target pair and workload in staging.
    """

    if not draft_tokens or not target_tokens:
        raise ValueError("draft and target token sequences are required")
    if len(draft_tokens) > len(target_tokens):
        raise ValueError("target sequence must cover every draft position")
    if min(
        costs.baseline_seconds_per_token,
        costs.draft_seconds_per_token,
        costs.verification_seconds_per_round,
        minimum_speedup,
    ) <= 0:
        raise ValueError("costs and minimum_speedup must be positive")

    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens, strict=False):
        if draft != target:
            break
        accepted += 1
    emitted = accepted if accepted == len(draft_tokens) else accepted + 1
    baseline_cost = emitted * costs.baseline_seconds_per_token
    speculative_cost = (
        len(draft_tokens) * costs.draft_seconds_per_token
        + costs.verification_seconds_per_round
    )
    speedup = baseline_cost / speculative_cost
    enable = speedup >= minimum_speedup
    reason = (
        f"measured round clears {minimum_speedup:.2f}x speedup gate"
        if enable
        else f"measured round misses {minimum_speedup:.2f}x speedup gate"
    )
    return SpeculationDecision(
        enable,
        accepted,
        len(draft_tokens),
        emitted,
        accepted / len(draft_tokens),
        speedup,
        reason,
    )

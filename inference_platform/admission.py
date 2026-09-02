"""Reserve bounded work before admission and shed overload with named reasons.

An unbounded queue converts a traffic spike into expired requests and KV pressure.
Admission control uses the request's worst-case live-token demand and deadline before
allocating work. When immediate capacity is gone, it allows only a bounded queue;
everything else is deliberately shed rather than accepted and silently abandoned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdmissionAction(str, Enum):
    ADMIT = "admit"
    QUEUE = "queue"
    SHED = "shed"


@dataclass(frozen=True)
class AdmissionRequest:
    identifier: str
    prompt_tokens: int
    max_output_tokens: int
    deadline_at: float

    @property
    def maximum_live_tokens(self) -> int:
        return self.prompt_tokens + self.max_output_tokens


@dataclass(frozen=True)
class AdmissionPolicy:
    max_live_tokens: int
    max_queued_requests: int
    max_tokens_per_request: int


@dataclass(frozen=True)
class AdmissionDecision:
    action: AdmissionAction
    reserved_tokens: int
    reason: str


class AdmissionController:
    def __init__(self, policy: AdmissionPolicy) -> None:
        if min(
            policy.max_live_tokens,
            policy.max_tokens_per_request,
        ) <= 0 or policy.max_queued_requests < 0:
            raise ValueError("admission capacities must be positive or zero where bounded")
        self.policy = policy
        self.live_tokens = 0
        self.queued_requests = 0
        self._decisions: dict[str, AdmissionDecision] = {}

    def decide(
        self,
        request: AdmissionRequest,
        *,
        now: float,
        estimated_queue_seconds: float,
    ) -> AdmissionDecision:
        """Admit, queue, or shed one request without partial reservation.

        Replays return the original decision before touching counters. New work is
        checked in this order: input validity and per-request bound, predicted
        deadline, immediate live-token capacity, then bounded queue capacity. An
        admitted request atomically reserves its worst-case prompt-plus-output token
        footprint; a queued request reserves a queue slot; a shed request changes no
        counters. The precise boundary that decided the action is returned.

        `estimated_queue_seconds` is an observed/modelled scheduling input, not an
        SLO guarantee. Production controllers should segment estimates by workload,
        account for preemption, and expose retry/backoff behavior to clients.

        Decisions are retained by request id until `release`, which refuses to release
        a shed one, so a rejection is permanent and its entry is never reclaimed. That
        keeps a retry storm from re-rolling the dice on a request already refused, but
        a real controller needs the opposite behavior too: a bounded time-to-live so a
        client that backs off and retries can be judged against current capacity, and
        eviction so the table cannot grow with every rejected request.
        """

        if request.identifier in self._decisions:
            return self._decisions[request.identifier]
        if not request.identifier.strip():
            raise ValueError("request identifier is required")
        if min(request.prompt_tokens, request.max_output_tokens) < 0:
            raise ValueError("token counts may not be negative")
        if request.maximum_live_tokens <= 0:
            raise ValueError("request must reserve at least one live token")
        if estimated_queue_seconds < 0:
            raise ValueError("queue estimate may not be negative")

        if request.maximum_live_tokens > self.policy.max_tokens_per_request:
            decision = AdmissionDecision(
                AdmissionAction.SHED, 0, "request token bound exceeds policy"
            )
        elif now + estimated_queue_seconds > request.deadline_at:
            decision = AdmissionDecision(
                AdmissionAction.SHED, 0, "predicted queue wait misses request deadline"
            )
        elif self.live_tokens + request.maximum_live_tokens <= self.policy.max_live_tokens:
            self.live_tokens += request.maximum_live_tokens
            decision = AdmissionDecision(
                AdmissionAction.ADMIT,
                request.maximum_live_tokens,
                "worst-case live-token reservation fits",
            )
        elif self.queued_requests < self.policy.max_queued_requests:
            self.queued_requests += 1
            decision = AdmissionDecision(
                AdmissionAction.QUEUE, 0, "live-token budget full; bounded queue slot reserved"
            )
        else:
            decision = AdmissionDecision(
                AdmissionAction.SHED, 0, "live-token budget and bounded queue are full"
            )
        self._decisions[request.identifier] = decision
        return decision

    def release(self, request_identifier: str) -> None:
        """Release the reservation associated with a completed or cancelled request.

        A shed request holds no reservation, so releasing one would free nothing and
        would drop the record that makes its rejection permanent. The next retry of
        that id would then be judged again against whatever capacity happened to
        exist, which is the retry storm `decide` retains the decision to prevent.
        Refusing here keeps that invariant in the code rather than in a comment.
        """

        decision = self._decisions.get(request_identifier)
        if decision is None:
            raise KeyError(request_identifier)
        if decision.action is AdmissionAction.SHED:
            raise ValueError(
                f"request {request_identifier!r} was shed; a shed decision is permanent"
            )
        del self._decisions[request_identifier]
        if decision.action is AdmissionAction.ADMIT:
            self.live_tokens -= decision.reserved_tokens
        elif decision.action is AdmissionAction.QUEUE:
            self.queued_requests -= 1

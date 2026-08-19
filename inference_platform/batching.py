"""Expose head-of-line blocking with a token-iteration scheduling simulation.

A request-level (static) batch admits work only when the current batch drains.
Continuous batching can refill a free sequence slot after each decode iteration.
The simulator makes that difference visible in service timestamps; it deliberately
omits kernel timing, prefill cost, and memory paging, which require real load tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BatchMode(str, Enum):
    STATIC = "static"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class BatchRequest:
    identifier: str
    arrival_step: int
    output_tokens: int


@dataclass(frozen=True)
class Completion:
    identifier: str
    first_service_step: int
    completion_step: int


@dataclass(frozen=True)
class BatchRun:
    mode: BatchMode
    completions: tuple[Completion, ...]
    makespan_steps: int
    total_wait_steps: int


def simulate_batching(
    requests: tuple[BatchRequest, ...], capacity: int, mode: BatchMode
) -> BatchRun:
    """Run static or continuous batching and return observed scheduling evidence.

    At each integer step, arrived requests join a FIFO queue. Static mode fills an
    empty batch and leaves freed lanes idle until every member finishes. Continuous
    mode refills each freed lane before the next token iteration. Every active
    request emits exactly one token per step, so completion times arise from the
    scheduler and request lengths rather than an expected result attached to input.

    The returned wait total counts arrival-to-first-service delay. This is the
    queueing component of TTFT, not a complete TTFT model; prefill execution is
    intentionally outside this isolated scheduling lesson.
    """

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not requests:
        raise ValueError("at least one request is required")
    identifiers = [request.identifier for request in requests]
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError("request identifiers are required")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("request identifiers must be unique")
    if any(request.arrival_step < 0 or request.output_tokens <= 0 for request in requests):
        raise ValueError("arrivals must be non-negative and output tokens positive")

    pending = sorted(requests, key=lambda request: (request.arrival_step, request.identifier))
    waiting: list[BatchRequest] = []
    active: dict[str, tuple[BatchRequest, int]] = {}
    first_service: dict[str, int] = {}
    completed: list[Completion] = []
    step = pending[0].arrival_step

    while pending or waiting or active:
        while pending and pending[0].arrival_step <= step:
            waiting.append(pending.pop(0))

        may_fill = mode is BatchMode.CONTINUOUS or not active
        if may_fill:
            while waiting and len(active) < capacity:
                request = waiting.pop(0)
                active[request.identifier] = (request, request.output_tokens)
                first_service[request.identifier] = step

        if not active:
            step = pending[0].arrival_step
            continue

        finished: list[str] = []
        for identifier, (request, remaining) in tuple(active.items()):
            remaining -= 1
            if remaining == 0:
                completed.append(
                    Completion(identifier, first_service[identifier], step + 1)
                )
                finished.append(identifier)
            else:
                active[identifier] = (request, remaining)
        for identifier in finished:
            del active[identifier]
        step += 1

    ordered = tuple(sorted(completed, key=lambda item: item.identifier))
    waits = sum(
        first_service[request.identifier] - request.arrival_step for request in requests
    )
    return BatchRun(mode, ordered, step, waits)

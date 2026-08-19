from __future__ import annotations

import unittest

from inference_platform.admission import (
    AdmissionAction,
    AdmissionController,
    AdmissionPolicy,
    AdmissionRequest,
)


class AdmissionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = AdmissionController(AdmissionPolicy(100, 1, 80))

    def test_benign_request_traverses_the_real_reservation_path(self) -> None:
        decision = self.controller.decide(
            AdmissionRequest("normal", 20, 30, deadline_at=10),
            now=0,
            estimated_queue_seconds=1,
        )
        self.assertEqual(decision.action, AdmissionAction.ADMIT)
        self.assertEqual(decision.reserved_tokens, 50)
        self.assertEqual(self.controller.live_tokens, 50)

    def test_replayed_request_cannot_double_reserve_capacity(self) -> None:
        request = AdmissionRequest("retry", 20, 30, deadline_at=10)
        first = self.controller.decide(request, now=0, estimated_queue_seconds=1)
        second = self.controller.decide(request, now=0, estimated_queue_seconds=1)
        self.assertEqual(first, second)
        self.assertEqual(self.controller.live_tokens, 50)

    def test_oversize_request_is_shed_without_mutating_state(self) -> None:
        before = (self.controller.live_tokens, self.controller.queued_requests)
        decision = self.controller.decide(
            AdmissionRequest("huge", 50, 50, deadline_at=10),
            now=0,
            estimated_queue_seconds=0,
        )
        self.assertEqual(decision.action, AdmissionAction.SHED)
        self.assertEqual((self.controller.live_tokens, self.controller.queued_requests), before)

    def test_overload_queues_once_then_sheds_instead_of_growing_unbounded(self) -> None:
        active = self.controller.decide(
            AdmissionRequest("active", 40, 40, 10), now=0, estimated_queue_seconds=0
        )
        queued = self.controller.decide(
            AdmissionRequest("queued", 30, 20, 10), now=0, estimated_queue_seconds=1
        )
        shed = self.controller.decide(
            AdmissionRequest("shed", 30, 20, 10), now=0, estimated_queue_seconds=1
        )
        self.assertEqual(active.action, AdmissionAction.ADMIT)
        self.assertEqual(queued.action, AdmissionAction.QUEUE)
        self.assertEqual(shed.action, AdmissionAction.SHED)
        self.assertIn("bounded queue", shed.reason)

    def test_predicted_deadline_failure_is_decided_before_allocation(self) -> None:
        decision = self.controller.decide(
            AdmissionRequest("late", 10, 10, deadline_at=2),
            now=1,
            estimated_queue_seconds=2,
        )
        self.assertEqual(decision.action, AdmissionAction.SHED)
        self.assertIn("deadline", decision.reason)
        self.assertEqual(self.controller.live_tokens, 0)


if __name__ == "__main__":
    unittest.main()

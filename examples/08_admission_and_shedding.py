"""Lesson 8: reserve worst-case live tokens, bound the queue, then shed explicitly."""

from inference_platform.admission import (
    AdmissionController,
    AdmissionPolicy,
    AdmissionRequest,
)


controller = AdmissionController(AdmissionPolicy(100, 1, 80))
requests = (
    AdmissionRequest("interactive", 30, 50, deadline_at=10),
    AdmissionRequest("queued", 20, 30, deadline_at=10),
    AdmissionRequest("overload", 20, 30, deadline_at=10),
)

for request in requests:
    decision = controller.decide(request, now=0, estimated_queue_seconds=1)
    print(f"{request.identifier}: {decision.action.value}: {decision.reason}")
print(f"reserved live tokens={controller.live_tokens}, queued={controller.queued_requests}")

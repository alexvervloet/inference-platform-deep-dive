"""Lesson 2: TTFT, TPOT, end-to-end latency, and throughput answer different questions."""

from inference_platform.metrics import RequestTrace, ServiceObjectives, evaluate_service


traces = (
    RequestTrace("interactive-1", 0.0, 0.4, 1.3, 10),
    RequestTrace("interactive-2", 0.2, 0.7, 1.6, 10),
    RequestTrace("long-prefill", 0.4, 1.6, 2.5, 10),
)
objectives = ServiceObjectives(
    max_p95_ttft_seconds=1.0,
    max_p95_tpot_seconds=0.15,
    max_p95_e2e_seconds=2.5,
    min_output_tokens_per_second=10,
)
report = evaluate_service(traces, objectives)

print(f"p95 TTFT: {report.p95_ttft_seconds:.2f}s")
print(f"p95 TPOT: {report.p95_tpot_seconds:.2f}s")
print(f"p95 E2E:  {report.p95_e2e_seconds:.2f}s")
print(f"output throughput: {report.output_tokens_per_second:.2f} tokens/s")
print(f"decision: {'PASS' if report.meets_objectives else 'FAIL'}")
for violation in report.violations:
    print(f"  - {violation}")

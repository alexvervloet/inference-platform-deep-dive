"""Lesson 3: refill freed decode lanes instead of waiting for the longest request."""

from inference_platform.batching import BatchMode, BatchRequest, simulate_batching


workload = (
    BatchRequest("short", arrival_step=0, output_tokens=1),
    BatchRequest("long", arrival_step=0, output_tokens=5),
    BatchRequest("arrives-later", arrival_step=1, output_tokens=1),
)

for mode in BatchMode:
    run = simulate_batching(workload, capacity=2, mode=mode)
    print(f"{mode.value}: makespan={run.makespan_steps}, wait={run.total_wait_steps}")
    for completion in run.completions:
        print(
            f"  {completion.identifier}: first={completion.first_service_step}, "
            f"done={completion.completion_step}"
        )

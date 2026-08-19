"""Lesson 7: fit and interconnect determine parallel layout; spare replicas provide DP."""

from inference_platform.parallelism import ClusterTopology, ParallelModel, plan_parallelism


model = ParallelModel(weight_memory_gib=60, runtime_memory_gib_per_gpu=4, experts=8)
for fast_link in (True, False):
    topology = ClusterTopology(
        nodes=2,
        gpus_per_node=4,
        gpu_memory_gib=40,
        usable_fraction=0.9,
        fast_intra_node_collectives=fast_link,
    )
    plan = plan_parallelism(model, topology, target_replicas=2)
    print(f"fast tensor collectives={fast_link}")
    print(
        f"  TP={plan.tensor_parallel} PP={plan.pipeline_parallel} "
        f"DP={plan.data_parallel} EP={plan.expert_parallel}"
    )
    print(f"  {plan.reason}")

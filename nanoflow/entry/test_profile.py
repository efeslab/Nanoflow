import argparse
import time
import torch.multiprocessing as mp

from nanoflow.entry.common import (
    CliArgs,
    setup_model_and_configs,
    ensure_weights,
    create_pipelines,
    create_shared_variables,
    start_workers,
    world_info,
    step_barrier,
)

from nanoflow.entry.worker_entry import worker_entry


def main():
    mp.set_start_method("spawn")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    arg_parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=1,
        help="Expert parallel size",
    )
    arg_parser.add_argument(
        "--model",
        default="8B",
        help="Pick which Pipeline to instantiate",
    )
    arg_parser.add_argument(
        "--kvcache_type",
        choices=["none", "torch", "flashinfer"],
        default="flashinfer",
        help="Pick which KVCache to use",
    )
    arg_parser.add_argument(
        "--network_type",
        choices=["allreduce", "allgather"],
        default="allreduce",
        help="Pick which network type to use",
    )
    arg_parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Enable CUDA graph",
    )
    arg_parser.add_argument(
        "--use_auto_search",
        action="store_true",
        help="Enable auto search",
    )
    arg_parser.add_argument(
        "--use_nanosplit",
        action="store_true",
        help="Enable nanosplit",
    )
    arg_parser.add_argument(
        "--affinity_module_path",
        type=str,
        default=None,
        help="Affinity module path",
    )
    args = arg_parser.parse_args()

    args = CliArgs(
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        test="profile",
        model=args.model,
        kvcache_type=args.kvcache_type,
        network_type=args.network_type,
        use_cuda_graph=args.use_cuda_graph,
        use_auto_search=args.use_auto_search,
        use_nanosplit=args.use_nanosplit,
        affinity_module_path=args.affinity_module_path,
    )

    world_size = world_info()
    arts = setup_model_and_configs(args)

    ensure_weights(arts.cfgs, arts.Pipeline, arts.weight_map)
    pipeline_list = create_pipelines(arts.cfgs, arts.Pipeline)
    command, shared_decode_bts, use_auto_search, use_nanosplit, use_cuda_graph, barrier = create_shared_variables(world_size)

    # Settings
    processes = start_workers(
        0.0,
        world_size,
        args.affinity_module_path,
        None,
        shared_decode_bts,
        None,
        barrier,
        pipeline_list,
        use_auto_search,
        arts.auto_search_path,
        use_nanosplit,
        use_cuda_graph,
        command,
        worker_entry,
    )
    command.value = b"Profile"
    step_barrier(barrier)

    command.value = b"Terminate"
    step_barrier(barrier)
    
    for p in processes:
        p.join()

    print("All processes have finished.")


if __name__ == "__main__":
    main()

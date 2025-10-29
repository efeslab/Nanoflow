import argparse
import time
import torch
import torch.multiprocessing as mp

from nanoflow.entry.common import (
    CliArgs,
    setup_model_and_configs,
    ensure_weights,
    create_pipelines,
    create_shared_variables,
    prefill_context,
    start_workers,
    world_info,
    step_barrier,
)

from nanoflow.entry.worker_entry import worker_entry


def main():
    mp.set_start_method("spawn")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Data parallel size",
    )
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
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        test="prefill_only",
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
    command, shared_decode_bts, use_auto_search, use_nanosplit, use_cuda_graph, barrier = create_shared_variables(
        world_size)

    # Settings
    seq_len = 512
    num_prefill_reqs = 128

    prefill_context_ids = arts.tokenizer.encode(prefill_context)
    print("len(prefill_context_ids): ", len(
        prefill_context_ids), "seq_len: ", seq_len)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]

    request_queues = [mp.Queue(maxsize=1000) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=1000)

    prefill_inputs = []
    decode_inputs = []
    output_strings = {}
    processes = start_workers(
        0.0,
        world_size,
        args.affinity_module_path,
        request_queues,
        shared_decode_bts,
        result_queue,
        barrier,
        pipeline_list,
        use_auto_search,
        arts.auto_search_path,
        use_nanosplit,
        use_cuda_graph,
        command,
        worker_entry,
    )
    command.value = b"Execute"
    shared_decode_bts.value = 0
    use_auto_search.value = args.use_auto_search
    use_nanosplit.value = args.use_nanosplit
    use_cuda_graph.value = args.use_cuda_graph

    group_prefill_size = 16  # might encounter the illegal memory access issue when group_prefill_size is too large, like group_prefill_size* seq_len == 16384
    cycles = (num_prefill_reqs + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
        if i == 1:
            torch.cuda.cudart().cudaProfilerStart()
        print(f"Cycle {i + 1}/{cycles}")
        prefill_inputs = []
        if i == cycles - 1:
            for j in range(i * group_prefill_size, num_prefill_reqs):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                output_strings[j] = prefill_input_ids.copy()
        else:
            for j in range(i * group_prefill_size, (i + 1) * group_prefill_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                output_strings[j] = prefill_input_ids.copy()
        for queue in request_queues:
            queue.put_nowait(prefill_inputs)

        step_barrier(barrier)

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        # print("new_tokens: ", new_tokens)

    torch.cuda.cudart().cudaProfilerStop()
    command.value = b"Terminate"
    step_barrier(barrier)
    
    for p in processes:
        p.join()

    print("All processes have finished.")

    output_text = arts.tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)


if __name__ == "__main__":
    main()

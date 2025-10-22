import time
import torch
import torch.multiprocessing as mp


def worker(T0, rank, affinity_module_path, *rest):
    # --- Set CPU affinity (import here so parent stays light) ---
    try:
        _aff_mod = __import__(affinity_module_path, fromlist=[
                              "set_affinity_for_rank", "tune_threads_like", "AFFINITY"])
        _set_affinity_for_rank = getattr(_aff_mod, "set_affinity_for_rank")
        _tune_threads_like = getattr(_aff_mod, "tune_threads_like")
        _AFFINITY = getattr(_aff_mod, "AFFINITY")
        _cores = _AFFINITY.get(rank, [])
        if _cores:
            _set_affinity_for_rank(rank, _cores)
            _tune_threads_like(_cores)
        print(f"[rank {rank}] CPU affinity set to cores: {_cores}", flush=True)
    except Exception as _e:
        # Affinity is best-effort; don't crash the worker if unavailable
        print(
            f"[rank {rank}] CPU affinity setup skipped or failed: {_e}", flush=True)

    from nanoflow.core.worker import worker as real_worker

    return real_worker(T0, rank, *rest)


def test_correctness():
    # Spawn one worker per GPU (or per unit of parallelism).
    input_string = "Hi, who are you?"
    input_ids = tokenizer.encode(input_string)
    input0 = [(i, input_ids.copy()) for i in range(2)]
    input1 = [(i, input_ids.copy()) for i in range(2, 4)]
    request_queues = [mp.Queue(maxsize=100) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=100)

    output_strings = {}
    for idx in range(4):
        output_strings[idx] = input_ids.copy()
    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (
            T0,
            rank,
            AFFINITY_MODULE_PATH,
            request_queues[rank],
            shared_decode_bts,
            result_queue,
            barrier,
            pipeline_list[rank],
            use_auto_search,
            None,
            use_nanosplit,
            use_cuda_graph,
            command,
        )
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")

    command.value = b"Execute"
    for queue in request_queues:
        queue.put(input0)
    shared_decode_bts.value = 0
    barrier.wait()
    barrier.wait()

    new_tokens = result_queue.get()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)

    new_tokens.extend(input1)
    for queue in request_queues:
        queue.put(new_tokens)
    shared_decode_bts.value = 2
    iterations = 20
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        # Set the shared task value.
        barrier.wait()
        barrier.wait()
        new_tokens = result_queue.get()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        for queue in request_queues:
            queue.put(new_tokens)
        shared_decode_bts.value = 4

    command.value = b"Terminate"
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.

    print("Waiting for all processes to finish... ", time.perf_counter() - T0)
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

    print("All processes have finished.")

    output_text = tokenizer.batch_decode(
        list(output_strings.values()), skip_special_tokens=True
    )

    print(output_text)

def test_prefill_only():
    seq_len = 512
    num_prefill_reqs = 128
    prefill_context_ids = tokenizer.encode(prefill_context)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]
    request_queues = [mp.Queue(maxsize=1000) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=1000)

    prefill_inputs = []
    output_strings = {}
    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (
            T0,
            rank,
            AFFINITY_MODULE_PATH,
            request_queues[rank],
            shared_decode_bts,
            result_queue,
            barrier,
            pipeline_list[rank],
            use_auto_search,
            auto_search_path,
            use_nanosplit,
            use_cuda_graph,
            command,
        )
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")
    command.value = b"Execute"
    shared_decode_bts.value = 0
    use_auto_search.value = USE_AUTO_SEARCH
    use_nanosplit.value = USE_NANOSPLIT
    use_cuda_graph.value = USE_CUDA_GRAPH

    group_prefill_size = 16
    cycles = (num_prefill_reqs + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
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

        barrier.wait()
        barrier.wait()

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)

    command.value = b"Terminate"
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.

    print("Waiting for all processes to finish... ", time.perf_counter() - T0)
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

    print("All processes have finished.")

    output_text = tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)

def test_performance():
    seq_len = 1024
    # seq_len = 2048
    # global_batch_size = 1024
    global_batch_size = 2048
    # global_batch_size = 3072
    # decode_batch_size = 128
    decode_batch_size = 640
    # decode_batch_size = 1280
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = tokenizer.encode(prefill_context)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]
    request_queues = [mp.Queue(maxsize=1000) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=1000)

    prefill_inputs = []
    decode_inputs = []
    output_strings = {}
    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (
            T0,
            rank,
            AFFINITY_MODULE_PATH,
            request_queues[rank],
            shared_decode_bts,
            result_queue,
            barrier,
            pipeline_list[rank],
            use_auto_search,
            auto_search_path,
            use_nanosplit,
            use_cuda_graph,
            command,
        )
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")
    command.value = b"Execute"
    shared_decode_bts.value = 0
    use_auto_search.value = 0
    use_nanosplit.value = 0
    use_cuda_graph.value = 0

    group_prefill_size = 8
    cycles = (decode_batch_size + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}")
        prefill_inputs = []
        if i == cycles - 1:
            for j in range(i * group_prefill_size, decode_batch_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                output_strings[j] = prefill_input_ids.copy()
        else:
            for j in range(i * group_prefill_size, (i + 1) * group_prefill_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                output_strings[j] = prefill_input_ids.copy()
        for queue in request_queues:
            queue.put_nowait(prefill_inputs)

        barrier.wait()
        barrier.wait()

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_inputs.extend(new_tokens)
        # print("new_tokens: ", new_tokens)

    # prepare for the testing configuration
    assert prefill_batch_size <= len(
        prefill_context_ids), f"prefill_batch_size {prefill_batch_size} should be less than {len(prefill_context_ids)}"
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend(
        [(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())]
    )
    for queue in request_queues:
        queue.put_nowait(decode_inputs)
    shared_decode_bts.value = decode_batch_size
    use_auto_search.value = 0
    use_nanosplit.value = 0
    use_cuda_graph.value = 0

    for i in range(decode_batch_size, decode_batch_size + 20):
        print("Cycle: ", i - decode_batch_size)
        next_prefill_idx = i + 1
        # Set the shared task value.
        barrier.wait()
        barrier.wait()
        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)

        new_tokens = new_tokens[:-1]
        # print("new_tokens: ", new_tokens)
        assert len(new_tokens) == decode_batch_size

        output_strings[next_prefill_idx] = prefill_context_ids[
            :prefill_batch_size
        ].copy()

        new_tokens.extend(
            [(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())]
        )

        for queue in request_queues:
            queue.put_nowait(new_tokens)

    print("Start to terminate")

    command.value = b"Terminate"
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.

    print("Waiting for all processes to finish... ", time.perf_counter() - T0)
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

    print("All processes have finished.")

    output_text = tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)


def profile():
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (
            T0,
            rank,
            AFFINITY_MODULE_PATH,
            None,
            None,
            None,
            barrier,
            pipeline_list[rank],
            0,
            None,
            0,
            0,
            command,
        )
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")

    command.value = b"Profile"
    barrier.wait()
    barrier.wait()

    command.value = b"Terminate"
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.

    print("Waiting for all processes to finish... ", time.perf_counter() - T0)
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

    print("All processes have finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    import argparse

    from transformers import AutoTokenizer

    from nanoflow.utils.util_functions import prepare_weight
    from nanoflow.utils.input_test import prefill_context
    from nanoflow.pybind.build.bind_all_reduce import NCCLWrapper

    AFFINITY_MODULE_PATH = None
    MULTI_GPU_MODE = True
    # AFFINITY_MODULE_PATH = "utils.affinity_utils"
    T0 = time.perf_counter()

    print("import modules, ", time.perf_counter() - T0)

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
        "--test",
        default="correctness",
        help="Which test to run",
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
        default=False,
        help="Enable CUDA graph",
    )
    arg_parser.add_argument(
        "--use_auto_search",
        action="store_true",
        default=False,
        help="Enable auto search",
    )
    arg_parser.add_argument(
        "--use_nanosplit",
        action="store_true",
        default=False,
        help="Enable nanosplit",
    )
    args = arg_parser.parse_args()

    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    TP_size: int = args.tensor_parallel_size
    EP_size: int = args.expert_parallel_size
    PP_size: int = 1
    DP_size: int = 1
    USE_CUDA_GRAPH: bool = args.use_cuda_graph
    USE_AUTO_SEARCH: bool = args.use_auto_search
    USE_NANOSPLIT: bool = args.use_nanosplit

    unique_nccl_ids = [NCCLWrapper.get_nccl_unique_id() for _ in range(10)]

    if args.model == "70B":
        MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
        weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
        from nanoflow.models.llama3_70B.config_llama3_70B import Llama3_70B_Config as Config

        if args.kvcache_type == "flashinfer":
            if args.network_type == "allreduce":
                from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache_allreduce import Pipeline
            elif args.network_type == "allgather":
                from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache_allgather import Pipeline
            else:
                raise NotImplementedError(
                    f"Network type {args.network_type} not implemented yet.")
        elif args.kvcache_type == "torch":
            if args.network_type == "allreduce":
                from nanoflow.models.llama3_70B.llama3_70B_KVCacheTorch_allreduce import Pipeline
            elif args.network_type == "allgather":
                from nanoflow.models.llama3_70B.llama3_70B_KVCacheTorch_allgather import Pipeline
            else:
                raise NotImplementedError(
                    f"Network type {args.network_type} not implemented yet.")
        else:
            raise NotImplementedError(
                f"KVCache type {args.kvcache_type} not implemented yet.")


        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=TP_size,
            tp_rank=i,
            kv_cache_type=args.kvcache_type,
            network_type=args.network_type,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = "/code/Nanoflow-python/nanoflow/auto_search/result_json/prefill_only_search_result_stage1.json"
        # auto_search_path = None

    elif args.model == "8B":
        weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
        from nanoflow.models.llama3_8B.llama3_8B_FlashinferKVCache_allreduce import Pipeline
        from nanoflow.models.llama3_8B.config_llama3_8B import Llama3_8B_Config as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=TP_size,
            tp_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")
        auto_search_path = "../auto_search/search_result_json/8B_allreduce_search_result.json"

    elif args.model == "Qwen1.5-MoE-A2.7B-EP":
        MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
        weight_map = "/code/hf/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
        from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline
        from nanoflow.models.qwen2_moe.config_qwen2_moe import Qwen2MoEConfig as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            ep_size=EP_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None
    
    elif args.model == "Qwen2-57B-A14B-Instruct-EP":
        MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
        weight_map = "/code/hf/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0"
        from nanoflow.models.qwen2_moe_57B.qwen2_moe_57B_ep import Pipeline
        from nanoflow.models.qwen2_moe_57B.config_qwen2_moe_57B import Qwen2MoEConfig as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            ep_size=EP_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None

    elif args.model == "Qwen2-57B-A14B-Instruct-TP-EP":
        MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
        weight_map = "/code/hf/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0"
        from nanoflow.models.qwen2_moe_57B.qwen2_moe_57B_tp_ep import Pipeline
        from nanoflow.models.qwen2_moe_57B.config_qwen2_moe_57B import Qwen2MoEConfig as Config
        assert world_size == TP_size == EP_size, "world_size should be equal to TP_size and EP_size"
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=TP_size,
            tp_rank=i,
            ep_size=EP_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None
    else:
        # from models.llama3_8B_KVCacheFA_TP2 import Pipeline
        raise ValueError("Unsupported model")

    # mkdir for profiler
    if args.test == "profile":
        print("Do Profile")
        profile_data_path = cfgs[0].profile_data_path()
        import os
        os.makedirs(profile_data_path, exist_ok=True)

    print("--------------------------------")
    print("MODEL_ID: ", MODEL_ID)
    print("args: ", args)
    print("--------------------------------")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # process weights
    HAS_CACHED_WEIGHT = cfgs[0].has_cached_weight()
    print("HAS_CACHED_WEIGHT: ", HAS_CACHED_WEIGHT)
    if not HAS_CACHED_WEIGHT:
        pipeline_weight_list = [
            Pipeline(cfg=cfgs[i])
            for i in range(world_size)
        ]
        prepare_weight(pipeline_weight_list, weight_map)

    # create pipeline instances
    pipeline_list = [
        Pipeline(cfg=cfgs[i])
        for i in range(world_size)
    ]

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    # A character array to hold the command string.
    command = mp.Array("c", 32)
    shared_decode_bts = mp.Value("i", 0)
    use_auto_search = mp.Value("i", 0)
    use_nanosplit = mp.Value("i", 0)
    use_cuda_graph = mp.Value("i", 0)

    # Create a Barrier for world_size workers plus the main process.
    barrier = mp.Barrier(world_size + 1)

    print("create shared variables, ", time.perf_counter() - T0)

    if args.test == "correctness":
        # optionally: set a global used by test_correctness
        test_correctness()
    elif args.test == "performance":
        test_performance()
    elif args.test == "prefill_only":
        test_prefill_only()
    elif args.test == "profile":
        profile()
    else:
        raise NotImplementedError(f"Unsupported test: {args.test}")
import time
import torch

def test_correctness_new():
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
        args = (T0, rank, request_queues[rank], shared_decode_bts, result_queue, barrier, pipeline_list[rank], use_auto_search, use_nanosplit, use_cuda_graph, command)
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

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)

    print(output_text)

def test_performance():
    seq_len = 1024
    # global_batch_size = 1024
    global_batch_size = 2048
    # decode_batch_size = 128
    decode_batch_size = 640
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = tokenizer.encode(prefill_context)  # which length is 1912.
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
        args = (T0, rank, request_queues[rank], shared_decode_bts, result_queue, barrier, pipeline_list[rank], use_auto_search, use_nanosplit, use_cuda_graph, command)
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")
    command.value = b"Execute"
    shared_decode_bts.value = 0
    use_auto_search.value = 0
    use_nanosplit.value = 0
    use_cuda_graph.value = 0

    group_prefill_size = 16
    cycles = (decode_batch_size + group_prefill_size - 1) // group_prefill_size


    for i in range(cycles):
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
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend([(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())])
    for queue in request_queues:
        queue.put_nowait(decode_inputs)
    shared_decode_bts.value = decode_batch_size
    use_auto_search.value = 1
    use_nanosplit.value = 1
    use_cuda_graph.value = 1

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

        output_strings[next_prefill_idx] = prefill_context_ids[:prefill_batch_size].copy()

        new_tokens.extend([(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())])

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

    output_text = tokenizer.batch_decode(list(output_strings.values())[:2], skip_special_tokens=True)
    print(output_text)

def test_profile():
    # Spawn one worker per GPU (or per unit of parallelism).
    prefill_context_ids = tokenizer.encode(prefill_context)  # which length is 1912.
    processes = []
    request_queues = [mp.Queue(maxsize=1000) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=1000)
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (T0, rank, request_queues[rank], shared_decode_bts, result_queue, barrier, pipeline_list[rank], 0, 0, 0, command)
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")
        request_queues[rank].put_nowait(prefill_context_ids)
    
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


if __name__ == '__main__':
    T0 = time.perf_counter()
    import torch.multiprocessing as mp
    import sys
    import argparse

    sys.path.append("../")
    sys.path.append('../pybind/build')

    from core.worker import worker
    from utils.util_functions import prepare_weight
    from transformers import AutoTokenizer
    from input_test import prefill_context

    print("import modules1, ", time.perf_counter() - T0)
    # from models.llama3_70B_KVCacheTorch_allgather import Pipeline
    # from models.llama3_70B_FlashinferKVCache_allgather import Pipeline
    # from models.llama3_70B_KVCacheTorch_allreduce import Pipeline
    from models.llama3_70B_FlashinferKVCache_allreduce import Pipeline
    # from models.llama3_8B_KVCacheFA_TP2 import Pipeline

    print("import modules, ", time.perf_counter() - T0)
    mp.set_start_method('spawn')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--load_hf_weight", action="store_true", help="Load weights from huggingface")

    args = arg_parser.parse_args()

    # print("initializing the modules and start mode setting, ", time.perf_counter() - T0)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

    # print("tokenize the inputs, initialize the output dict, ", time.perf_counter() - T0)

    weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"

    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    TP_size = 4
    PP_size = 1
    DP_size = 1

    import bind_all_reduce
    unique_id_1 = bind_all_reduce.get_nccl_unique_id()
    unique_id_2 = bind_all_reduce.get_nccl_unique_id()
    unique_id_3 = bind_all_reduce.get_nccl_unique_id()
    unique_id_4 = bind_all_reduce.get_nccl_unique_id()

    unique_nccl_ids = [unique_id_1, unique_id_2, unique_id_3, unique_id_4]
    assert world_size == TP_size * PP_size * DP_size, f"world size {world_size} is not equal to TP size {TP_size} * PP size {PP_size} * DP size {DP_size}"

    if args.load_hf_weight:
        pipeline_weight_list = [
                (i, f"cuda:{i}", Pipeline(
                    TP_idx=i,
                    TP_size=TP_size,
                )) for i in range(world_size)
            ]
        prepare_weight(pipeline_weight_list, weight_map_wzr)

    # print("finish update pipeline")

    pipeline_list = [ Pipeline(
        TP_idx=i,
        TP_size=TP_size,
        unique_nccl_ids=unique_nccl_ids) for i in range(world_size) ]
    
    # print(f"Number of GPUs: {world_size}")

    # print("create pipeline instance, ", time.perf_counter() - T0)
    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    command = mp.Array('c', 32)  # A character array to hold the command string.
    shared_decode_bts = mp.Value('i', 0)
    use_auto_search = mp.Value('i', 0)
    use_nanosplit = mp.Value('i', 0)
    use_cuda_graph = mp.Value('i', 0)

    # Create a Barrier for world_size workers plus the main process.
    barrier = mp.Barrier(world_size + 1)
    
    print("create shared variables, ", time.perf_counter() - T0)
    
    test_correctness_new()
    # test_performance()
    # test_profile()
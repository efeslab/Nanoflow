import time
import torch

if __name__ == '__main__':
    T0 = time.perf_counter()
    import torch.multiprocessing as mp
    import sys, os
    import argparse

    sys.path.append("../")
    sys.path.append('../pybind/build')
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    from core.worker import worker
    from utils.util_functions import prepare_weight
    from multiprocessing import Value, Array, Barrier
    from transformers import AutoTokenizer
    print("import modules1, ", time.perf_counter() - T0)
    # from models.llama3_70B_KVCacheTorch_allgather import Pipeline
    # from models.llama3_70B_FlashinferKVCache_allgather import Pipeline
    # from models.llama3_70B_KVCacheTorch_allreduce import Pipeline
    from models.llama3_70B_FlashinferKVCache_allreduce import Pipeline

    print("import modules, ", time.perf_counter() - T0)
    mp.set_start_method('spawn')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--load_hf_weight", type=bool, default=False, help="Load weights from huggingface")

    args = arg_parser.parse_args()

    # print("initializing the modules and start mode setting, ", time.perf_counter() - T0)
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    input_strings = [ "Hi, who are you?" for _ in range(4)]
    input_ids = [(idx, tokenizer.encode(s)) for idx, s in enumerate(input_strings)]

    output_strings = {}
    for idx, ids in input_ids:
        output_strings[idx] = ids
    # print("tokenize the inputs, initialize the output dict, ", time.perf_counter() - T0)

    weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"

    
    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    TP_size = 2
    PP_size = 1
    DP_size = 1

    assert world_size == TP_size * PP_size * DP_size, f"world size {world_size} is not equal to TP size {TP_size} * PP size {PP_size} * DP size {DP_size}"

    if args.load_hf_weight:
        pipeline_dict = dict(
            [
                (f"cuda:{i}", Pipeline(
                    TP_idx=i,
                    TP_size=TP_size,
                )) for i in range(world_size)
            ]
        )
        prepare_weight(pipeline_dict, weight_map_wzr)

    # print("finish update pipeline")

    pipeline_list = [ Pipeline(
        TP_idx=i,
        TP_size=TP_size,) for i in range(world_size) ]
    
    # print(f"Number of GPUs: {world_size}")

    # print("create pipeline instance, ", time.perf_counter() - T0)
    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    command = Value('i', 1)    # 'i' stands for a signed integer.
    shared_batch_size = Value('i', 0)    # 'i' stands for a signed integer.
    shared_array = Array('i', 4)  # An array of integers with length equal to world_size.

    # Create a Barrier for world_size workers plus the main process.
    barrier = Barrier(world_size + 1)
    
    print("create shared variables, ", time.perf_counter() - T0)
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (T0, rank, world_size, shared_batch_size, shared_array, barrier, pipeline_list, command, input_ids)
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)
        # print(f"Process {rank} started on GPU {rank} in {time.perf_counter() - start_time:.2f} seconds")
    
    print("Waiting for all processes to start... ", time.perf_counter() - T0)
    command.value = 1
    barrier.wait()
    barrier.wait()

    for i in range(2):
        output_strings[i].append(shared_array[i])

    command.value = 2
    iterations = 20
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        # Set the shared task value.
        barrier.wait()

        barrier.wait()
        for i in range(4):
            output_strings[i].append(shared_array[i])
    
    command.value = -1
    
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.
    
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

    print("All processes have finished.")

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)
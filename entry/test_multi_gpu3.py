import sys, os
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import torch
import nvtx
import torch.multiprocessing as mp
from multiprocessing import Value, Array, Barrier

from transformers import AutoTokenizer
from models.llama3_FlashinferKVCache import Pipeline

def worker(rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, temp_out, shared_command):
    """
    Worker process: wait for a new task value from the main process,
    then compute (rank + shared_int) and store the result in shared_array[rank].
    """
    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        match shared_command.value:
            case 0:
                # pipeline.init_external_data()
                pipeline.batched_kv_cache.pre_allocate(0,5)
                pipeline.batched_kv_cache._pool.k_data[0].fill_(1)
                print("in worker: ", pipeline.batched_kv_cache.get(0,0))
                print("in worker shape: ", pipeline.batched_kv_cache.get(0,0)[0].shape)
                # print(pipeline.batched_kv_cache.get_whole_kv_data_all_layers())
                print("init_external_data")
            case 1:
                # temp_out = torch.zeros(pipeline.batch_size, dtype=torch.int32, device='cuda')
                # new_tokens = pipeline.run(temp_out)
                new_tokens = pipeline.run()
                with nvtx.annotate("post_run_stage"):
                    for i, item in enumerate(new_tokens):
                        output_strings[i].append(item[0])
                    # pipeline.update(output_strings)
                with nvtx.annotate("update_stage"):
                    pipeline.update(new_tokens, decode_flag=True)
                # Read the shared integer.
                with shared_int.get_lock():
                    task_val = shared_int.value

                flattened = [item for sublist in new_tokens for item in sublist]
                shared_batch_size.value = len(flattened)
                # Check for termination signal.
                if task_val == -1:
                    # Call barrier a second time to keep the barrier count consistent.
                    barrier.wait()
                    break

                # Compute the result and write to the worker’s assigned index.
                shared_array[rank] = rank + task_val
                # print("output_strings: ", output_strings)
        # Second barrier: wait until all workers finish computation.
        barrier.wait()
    
    # Worker exits gracefully.
    
if __name__ == '__main__':
    
    # print available GPUs
    # print("Available GPUs: ", torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # input_strings = ["Hi, who are you?"]
    # input_strings = ["Hi, who are you?", "What's the weather today?"]
    input_strings = [ "Hi, who are you?" for _ in range(1024)]
    input_ids = [tokenizer.encode(s) for s in input_strings]
    # print(input_ids)
    pipeline = Pipeline()
    pipeline.init_external_data()
    pipeline.batched_kv_cache.pre_allocate(0,5)
    # pipeline.init_operations()
    # pipeline.init_dependency()
    # pipeline.init_executor()
    # pipeline.init_set_shape()
    # pipeline.init_set_weight("/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
    # pipeline.update(input_ids)

    # mp.set_start_method('fork', force=True)
    mp.set_start_method('spawn')
    
    # Use the number of available GPUs (or set a fixed number). Here we use torch.cuda.device_count()
    # if you have GPUs; otherwise, you could simply set world_size = 4.
    world_size = 1
    
    iterations = 20

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    shared_command = Value('i', 1)    # 'i' stands for a signed integer.
    shared_int = Value('i', 0)    # 'i' stands for a signed integer.
    shared_batch_size = Value('i', pipeline.batch_size)    # 'i' stands for a signed integer.
    shared_array = Array('i', world_size)  # An array of integers with length equal to world_size.
    
    temp_out = torch.zeros(shared_batch_size.value, dtype=torch.int32, device='cuda')
    # Create a Barrier for world_size workers plus the main process.
    barrier = Barrier(world_size + 1)
    
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, temp_out, shared_command))
        p.start()
        processes.append(p)
    
    shared_command.value = 0
    barrier.wait()
    barrier.wait()

    # start_time = time.time()
    print(pipeline.batched_kv_cache.get(0,0))
    print("in main: ", pipeline.batched_kv_cache.get(0,0)[0].shape)
    pipeline.

    # torch.cuda.empty_cache()
    # device = torch.cuda.current_device()
    # reserved_memory = torch.cuda.memory_reserved(device)
    # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
    # pipeline.config()

    output_strings = []
    for i in input_ids:
        output_strings.append(i)
    #     print("input_ids: ", i)
    # For each iteration, update the shared integer, synchronize with the workers,
    # and let them compute and write their results.
    # for i in range(iterations):
    #     # Set the shared task value.
    #     with shared_int.get_lock():
    #         shared_int.value = i
        
    #     # First barrier: signal to all workers that a new task is available.
    #     barrier.wait()
        
    #     # Second barrier: wait for all workers to finish processing.
    #     barrier.wait()

    #     temp_out = torch.zeros(shared_batch_size.value, dtype=torch.int32, device='cuda')
        
        # (Optional) Main process could inspect shared_array here if desired.
        # For example, expected value at index j is: j + i.
        # results = list(shared_array)
    
    # Signal termination: write -1 into the shared integer.
    with shared_int.get_lock():
        shared_int.value = -1
    
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.
    
    # total_time = time.time() - start_time
    # print("Total time for {} iterations: {:.4f} seconds".format(iterations, total_time))
    
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

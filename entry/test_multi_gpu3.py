import torch.multiprocessing as mp

import sys, os
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
import time
import torch
import torch.distributed as dist
import nvtx

from multiprocessing import Value, Array, Barrier

from transformers import AutoTokenizer
from models.llama3_FlashinferKVCache_TP2 import Pipeline
# from models.llama3_KVCacheTorch_TP2 import Pipeline

def worker(rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, temp_out, shared_command, input_ids):
    """
    Worker process: wait for a new task value from the main process,
    then compute (rank + shared_int) and store the result in shared_array[rank].
    """
    
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    pipeline.init_streams()
    pipeline.config_streams()
    pipeline.config_network(rank)
    pipeline.update_network_ops()
    output_strings = {}
    for idx, tensor in input_ids:
        output_strings[idx] = tensor
    
    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        match shared_command.value:
            case 1:
                input0 = input_ids[0:2]
                pipeline.update(input0, 0, device_id=rank)
                new_tokens = pipeline.run(rank=rank, file_name=f"tp_test_flashinfer_{rank}", filefolder_name=f"tp_test_flashinfer_{rank}_folder")
                for req_idx, new_token in new_tokens:
                    output_strings[req_idx].extend(new_token)
                decode_batchsize = len(new_tokens)
                assert decode_batchsize == 2

                new_tokens.extend(input_ids[2:4])

                output_length=20
                for i in range(output_length):
                    print("Cycle: ", i)
                    pipeline.update(new_tokens, decode_batchsize, device_id=rank)

                    new_tokens = pipeline.run(rank=rank, file_name=f"tp_test_flashinfer_{rank}", filefolder_name=f"tp_test_flashinfer_{rank}_folder")
                    for req_idx, new_token in new_tokens:
                        output_strings[req_idx].extend(new_token)
                    decode_batchsize = len(new_tokens)

                # flattened = [item for sublist in new_tokens for item in sublist]
                # shared_batch_size.value = len(flattened)
                # Check for termination signal.
                output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
                print(output_text)


                
            case -1:
                # Termination signal received.
                pipeline.terminate()
                barrier.wait()
                break
        # Second barrier: wait until all workers finish computation.
        barrier.wait()
    
    # Worker exits gracefully.
    
if __name__ == '__main__':
    # print("main Current start method:", mp.get_start_method(allow_none=True))
    mp.set_start_method('spawn')
    # print("main Current start method:", mp.get_start_method(allow_none=True))
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    input_strings = [ "Hi, who are you?" for _ in range(16)]
    input_ids = [(idx, tokenizer.encode(s)) for idx, s in enumerate(input_strings)]

    pipeline = Pipeline()
    pipeline.init_external_data()
    pipeline.init_operations()
    pipeline.init_dependency()
    pipeline.init_set_shape()
    print("finish init shape")
    pipeline.init_set_weight("/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a", cached=True)

    print("finish update pipeline")

    world_size = pipeline.num_devices
    print(f"Number of GPUs: {world_size}")
    
    iterations = 1

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    shared_command = Value('i', 1)    # 'i' stands for a signed integer.
    shared_int = Value('i', 0)    # 'i' stands for a signed integer.
    shared_batch_size = Value('i', 0)    # 'i' stands for a signed integer.
    shared_array = Array('i', world_size)  # An array of integers with length equal to world_size.
    
    temp_out = torch.zeros(shared_batch_size.value, dtype=torch.int32, device='cuda')
    # Create a Barrier for world_size workers plus the main process.
    barrier = Barrier(world_size + 1)
    
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        args = (rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, temp_out, shared_command, input_ids)
        p = mp.Process(target=worker, args=args)

        print(f"Starting process {rank} on GPU {rank}")
        p.start()
        processes.append(p)
    
    print("Waiting for all processes to start...")
    shared_command.value = 1
    barrier.wait()
    barrier.wait()

    # start_time = time.time()


    # pipeline.

    # torch.cuda.empty_cache()
    # device = torch.cuda.current_device()
    # reserved_memory = torch.cuda.memory_reserved(device)
    # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
    # pipeline.config()


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
    with shared_command.get_lock():
        shared_command.value = -1
    
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.
    
    # total_time = time.time() - start_time
    # print("Total time for {} iterations: {:.4f} seconds".format(iterations, total_time))
    
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()

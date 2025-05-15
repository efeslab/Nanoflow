import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def worker(rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, command, input_ids):
    """
    Worker process: wait for a new task value from the main process,
    then compute (rank + shared_int) and store the result in shared_array[rank].
    """
    from torch.cuda import set_device
    set_device(rank)
    pipeline.kv_cache.device_id = rank
    pipeline.init_streams()
    pipeline.config_streams()
    pipeline.config_network(rank)
    pipeline.update_network_ops()
    
    new_tokens = None

    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        if command.value == 1:
                input0 = input_ids[0:4]
                logging.info(f"Worker {rank} input0: {input0}")
                pipeline.update(input0, 0, device_id=rank)
                new_tokens = pipeline.run(rank=rank, file_name=f"./test_data/70B_test_flashinfer_{rank}", filefolder_name=f"./test_data/70B_test_flashinfer_{rank}_folder")
                decode_batchsize = len(new_tokens)
                print("new_tokens: ", new_tokens)
                if rank == 0:
                    for req_idx, new_token in new_tokens:
                        shared_array[req_idx] = new_token[0]
                # new_tokens.extend(input_ids[2:4])
        elif command.value == 2:
                logging.info(f"Worker {rank} new_tokens: {new_tokens}")
                pipeline.update(new_tokens, decode_batchsize=4, device_id=rank)
                new_tokens = pipeline.run(rank=rank, file_name=f"70B_test_flashinfer_{rank}", filefolder_name=f"70B_test_flashinfer_{rank}_folder")
                print("new_tokens: ", new_tokens)
                if rank == 0:
                    for req_idx, new_token in new_tokens:
                        shared_array[req_idx] = new_token[0]
        elif command.value == -1:
                # Termination signal received.
                pipeline.terminate()
                barrier.wait()
                break
        # Second barrier: wait until all workers finish computation.
        barrier.wait()
    
    # Worker exits gracefully.

if __name__ == '__main__':
    import torch.multiprocessing as mp

    import time
    import sys, os
    sys.path.append("../")
    # os.environ["HF_HOME"] = "/code/hf"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    from multiprocessing import Value, Array, Barrier
    from transformers import AutoTokenizer
    # from models.llama3_FlashinferKVCache_TP2 import Pipeline
    # from models.llama3_KVCacheTorch_TP2 import Pipeline
    # from models.llama3_70B_KVCacheTorch_TP8 import Pipeline
    from models.llama3_70B_KVCacheFA_TP8 import Pipeline
    
    mp.set_start_method('spawn')
    
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    input_strings = [ "Hi, who are you?" for _ in range(4)]
    input_ids = [(idx, tokenizer.encode(s)) for idx, s in enumerate(input_strings)]

    output_strings = {}
    for idx, ids in input_ids:
        output_strings[idx] = ids

    pipeline = Pipeline()
    pipeline.init_external_data()
    pipeline.init_operations()
    pipeline.init_dependency()
    pipeline.init_set_shape()
    print("finish init shape")
    # weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
    weigth_map_amd = "/work1/kasikci/kanzhu/models/llama3-70b"
    pipeline.init_set_weight(weigth_map_amd, cached=True)

    print("finish update pipeline")

    world_size = pipeline.num_devices
    print(f"Number of GPUs: {world_size}")

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    command = Value('i', 1)    # 'i' stands for a signed integer.
    shared_int = Value('i', 0)    # 'i' stands for a signed integer.
    shared_batch_size = Value('i', 0)    # 'i' stands for a signed integer.
    shared_array = Array('i', 4)  # An array of integers with length equal to world_size.

    # Create a Barrier for world_size workers plus the main process.
    barrier = Barrier(world_size + 1)
    
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        start_time = time.time()
        args = (rank, world_size, shared_int, shared_batch_size, shared_array, barrier, pipeline, command, input_ids)
        p = mp.Process(target=worker, args=args)

        print(f"Starting process {rank} on GPU {rank}")
        p.start()
        processes.append(p)
        print(f"Process {rank} started on GPU {rank} in {time.time() - start_time:.2f} seconds")
    
    print("Waiting for all processes to start...")
    command.value = 1
    barrier.wait()
    barrier.wait()

    # get the shared array values
    print(f"Shared array values: {list(shared_array)}")
    for i in range(2):
        output_strings[i].append(shared_array[i])

    iterations = 20
    #     print("input_ids: ", i)
    # For each iteration, update the shared integer, synchronize with the workers,
    # and let them compute and write their results.
    command.value = 2
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        # Set the shared task value.
        barrier.wait()

        barrier.wait()
        for i in range(4):
            output_strings[i].append(shared_array[i])
    
    # Signal termination: write -1 into the shared integer.
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

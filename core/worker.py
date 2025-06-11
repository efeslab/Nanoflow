import time
import torch
import os

def worker(start_time, rank, world_size, shared_batch_size, shared_array, barrier, pipeline_list, command, input_ids):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    pipeline = pipeline_list[rank]

    pipeline.set_device(device)
    pipeline.init_external_data()
    pipeline.init_operations()
    pipeline.init_dependency()
    pipeline.init_set_shape()
    print("finish init shape")
    # weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
    pipeline.init_set_weight(weight_map_wzr, cached=True)

    pipeline.init_streams()
    pipeline.config_network(rank)
    pipeline.update_network_ops()
    
    new_tokens = None
    cycle_count = 0

    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        match command.value:
            case 1:
                input0 = input_ids[0:2]
                pipeline.update(input0, decode_batch_size=0)
                # new_tokens = pipeline.run(file_name=f"./test_data/70B_test_torch_with_allreduce_{rank}", filefolder_name=f"./test_data/70B_test_torch_with_allreduce_{rank}_folder")
                new_tokens = pipeline.run(file_name=f"./test_data/70B_test_flashinfer_with_allreduce_{rank}", filefolder_name=f"./test_data/70B_test_flashinfer_with_allreduce_{rank}_folder")
                assert len(new_tokens) == 2, f"Expected 2 new tokens, got {len(new_tokens)}"
                print("new_tokens: ", new_tokens, "ttft: ", time.perf_counter() - start_time)
                if rank == 0:
                    for req_idx, new_token in new_tokens:
                        shared_array[req_idx] = new_token[0]
                new_tokens.extend(input_ids[2:4])
                pipeline.update(new_tokens, decode_batch_size=2)

            case 2:
                # new_tokens = pipeline.run(file_name=f"./test_data/70B_test_torch_with_allreduce_{rank}", filefolder_name=f"./test_data/70B_test_torch_with_allreduce_{rank}_folder")
                new_tokens = pipeline.run(file_name=f"./test_data/70B_test_flashinfer_with_allreduce_{rank}", filefolder_name=f"./test_data/70B_test_flashinfer_with_allreduce_{rank}_folder")
                assert len(new_tokens) == 4, f"Expected 4 new tokens, got {len(new_tokens)}"
                print("new_tokens: ", new_tokens)
                # save self.kv_cache.get(0,0) to a file for debugging
                # if device == "cuda:0":
                #     os.makedirs("./kv_cache_testing", exist_ok=True)
                #     torch.save(pipeline.kv_cache.get(0, 0)[0].cpu(), f"./kv_cache_testing/kvcache_0_0_{cycle_count}.pt")
                pipeline.update(new_tokens, decode_batch_size=4)
                
                if rank == 0:
                    for req_idx, new_token in new_tokens:
                        shared_array[req_idx] = new_token[0]
                
            case -1:
                # Termination signal received.
                pipeline.terminate()
                barrier.wait()
                break
        # Second barrier: wait until all workers finish computation.
        cycle_count += 1
        barrier.wait()
    
    # Worker exits gracefully.
import time
import torch
import torch.multiprocessing as mp
from nanoflow.utils.prof_marker import prof_marker

def worker(start_time, rank, request_queue: mp.Queue, shared_decode_bts, result_queue: mp.Queue, barrier, work_pipeline, use_auto_search, profile_result_path, use_nanosplit, use_cuda_graph, command):
    torch.cuda.set_device(rank)
    pipeline = work_pipeline
    pipeline.init(None, cached=True)

    new_tokens = None
    cycle_count = 0

    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        # cmd = ''.join(command[:]).strip()
        cmd = command.value.decode()
        match cmd:
            case "Execute":
                with prof_marker(f"Worker {rank} Execute S1", color="blue"):
                    input = request_queue.get(timeout=1)
                    decode_bts = shared_decode_bts.value
                with prof_marker(f"Worker {rank} Execute S2", color="blue"):
                    pipeline.update(input, decode_batch_size=decode_bts, profile_result_path=profile_result_path, use_auto_search=use_auto_search.value, use_nano_split=use_nanosplit.value, use_cuda_graph=use_cuda_graph.value)
                with prof_marker(f"Worker {rank} Execute S3", color="blue"):
                    new_tokens = pipeline.run()
                # print("new_tokens: ", new_tokens, "ttft: ", time.perf_counter() - start_time)
                with prof_marker(f"Worker {rank} Execute S4", color="blue"):
                    if rank == 0:
                        result_queue.put_nowait(new_tokens)

            case "Profile":
                input_ids = torch.randint(
                    0, 100000, (8192,), device=torch.device(rank)
                )
                pipeline.init_profile_data()

                # stream_names = ["TEST_TOTAL"]
                stream_names = [ f"TEST_{sm_count}" for sm_count in pipeline.sm_counts ]
                for stream_name in stream_names:
                    pipeline.reset()
                    print(f"Stream: {stream_name}")

                    # test for prefill
                    # total_batch_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
                    # total_batch_sizes = [640, 1408, 2048]
                    total_batch_sizes = [1280, 1792, 3072]
                    for idx, total_batch_size in enumerate(total_batch_sizes):
                        input = [(idx, input_ids[:total_batch_size].copy())]

                        pipeline.update(input, is_profile=True, stream_name=stream_name)
                        pipeline.profile_run()
                
                    # test for decode
                    # total_batch_sizes = [128, 256, 384]
                    # total_batch_sizes = [128, 256, 384, 512, 640]
                    # total_batch_sizes = [384]
                    total_batch_sizes = [1280]
                    # prepare the decode inputs for a special input_length
                    input_length = 1024
                    # output_length = 0
                    prefill_input_ids = input_ids[:input_length]

                    for total_batch_size in total_batch_sizes:
                        decode_inputs = []
                        pipeline.reset()
                        # initialize the reqs for first {total_batch_size} requests
                        for i in range(total_batch_size):
                            input = [(i, prefill_input_ids.copy())]
                            pipeline.update(input)
                            new_tokens = pipeline.run()
                            decode_inputs.extend(new_tokens)
                            print("new_tokens: ", new_tokens)
                            print("Stream_name: ", stream_name)
                            print("total_batch_size: ", total_batch_size)

                        pipeline.update(decode_inputs, total_batch_size, is_profile=True, stream_name=stream_name)
                        pipeline.profile_run()

            case "Terminate":
                # Termination signal received.
                # pipeline.terminate()
                barrier.wait()
                break
        # Second barrier: wait until all workers finish computation.
        cycle_count += 1
        barrier.wait()
    
    # Worker exits gracefully.
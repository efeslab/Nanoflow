from contextlib import nullcontext
import logging

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) if rank == 0 else nullcontext() as prof:
        while True:
            # First barrier: wait until the main process writes a new task.
            barrier.wait()
            if command.value == 1:
                input0 = input_ids[0:len(input_ids) // 2]
                logging.info(f"Worker {rank} input0: {input0}")
                pipeline.update(input0, 0, device_id=rank)
                new_tokens = pipeline.run(rank=rank, file_name=f"./test_data/70B_test_flashinfer_{rank}", filefolder_name=f"./test_data/70B_test_flashinfer_{rank}_folder")
                decode_batchsize = len(new_tokens)
                print("new_tokens: ", new_tokens)
                if rank == 0:
                    for req_idx, new_token in new_tokens:
                        shared_array[req_idx] = new_token[0]
                new_tokens.extend(input_ids[len(input_ids) // 2:])
            elif command.value == 2:
                logging.info(f"Worker {rank} new_tokens: {new_tokens}")
                decode_batchsize = 0
                for sublist in new_tokens:
                    decode_batchsize += 1 if len(sublist[1]) == 1 else 0
                pipeline.update(new_tokens, decode_batchsize=decode_batchsize, device_id=rank)
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
    
    if rank == 0:
        prof.export_chrome_trace("trace_mi300_torch.json")
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
    from models.llama3_70B_KVCacheTorch_TP8 import Pipeline
    # from models.llama3_8B_KVCacheFA_TP8 import Pipeline
    # from models.llama3_70B_KVCacheFA_TP8 import Pipeline
    
    mp.set_start_method('spawn')
    
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    batch_size = 512
    input_string = "Large Language Models (LLMs) have rapidly evolved from simple sequence-to-sequence systems into sophisticated architectures capable of human-level language understanding and generation. The architecture underpinning most of these models is the Transformer, introduced in the seminal paper “Attention is All You Need” by Vaswani et al. in 2017. Since then, the Transformer has served as the foundation for models such as GPT-3, PaLM, LLaMA, and Claude. These models, especially those with tens or hundreds of billions of parameters, rely on innovations in model scaling, parallelism, training data curation, and optimization techniques. At the heart of the Transformer is the attention mechanism. This allows the model to weigh and focus on different parts of the input sequence dynamically. Self-attention, in particular, enables the model to capture contextual relationships between words irrespective of their distance in the sequence. This is a dramatic departure from earlier RNNs and LSTMs, which processed sequences linearly and struggled with long-range dependencies. Self-attention operates in parallel, which is not only more computationally efficient but also critical for scaling up models. As models scaled, new challenges emerged—most notably, the explosion in compute requirements. Training GPT-3, for example, required hundreds of petaflop-days of compute and massive datasets scraped from the internet. To handle such computational demands, researchers employed techniques like model parallelism, data parallelism, pipeline parallelism, and optimizer sharding. Frameworks such as DeepSpeed and Megatron-LM became crucial in managing training across thousands of GPUs. However, bigger isn’t always better. While adding more parameters generally improves model capacity, it also introduces inefficiencies. This led to the rise of Mixture of Experts (MoE) models, where only a subset of the model’s parameters are active at a time. Instead of activating all parts of a monolithic network, an MoE model uses a gating mechanism to route inputs to a few selected “experts.” Each expert is a small feedforward network, and typically only 2–4 experts out of dozens or hundreds are used per token. This dramatically reduces the compute cost while keeping the overall model capacity high. Google's GLaM and Switch Transformer are notable implementations of this approach. Training such models also brings challenges like expert imbalance, where some experts are overused while others are underutilized. This is addressed using auxiliary losses that encourage balanced routing. Another complication is that the gating mechanism can become a bottleneck if it is not efficiently implemented, especially at scale. Another key component of LLM development is data. Large-scale training data must be diverse, representative, and relatively clean. Curating such datasets involves filtering out low-quality content, removing duplicate web pages, decontaminating evaluation sets, and ensuring that personal or harmful information is minimized. While some models are trained on curated corpora like The Pile or Common Crawl, others use proprietary datasets augmented with internal or human-labeled content. Instruct-tuning and reinforcement learning from human feedback (RLHF) further refine these models to align their outputs with human preferences. The emergence of RLHF as a fine-tuning method is particularly transformative. After a base model is trained to predict the next token, it’s further fine-tuned using rankings or preferences from human annotators. This process helps the model generate more helpful, harmless, and honest responses. OpenAI's ChatGPT and Anthropic's Claude use similar RLHF-based training pipelines. But alignment isn’t just a technical problem—it’s also philosophical. How should a model behave in ambiguous moral situations? Who decides what is “helpful” or “safe”? These are open questions that the AI research community, ethicists, and policymakers must grapple with. And as models become capable of multimodal reasoning—processing not just text but also images, audio, and video—the alignment problem becomes even more complex. Deployment of LLMs also poses engineering and infrastructure challenges. Serving a model like GPT-4 or Claude at scale requires careful attention to latency, cost-efficiency, caching strategies, and prompt engineering. Techniques like speculative decoding, retrieval-augmented generation (RAG), prompt compression, and system message templating are employed to make inference faster and more relevant. The future of LLMs appears to be moving in several directions simultaneously: larger models, more efficient smaller models, multimodal models, and models that can interact with tools and APIs. Tool-use in LLMs allows them to perform tasks like math, code execution, web search, or database queries by invoking external systems—essentially combining reasoning with action. This trend bridges the gap between static language modeling and interactive AI agents. In summary, the evolution of large language models is not just a story of bigger models and faster GPUs. It's a story of clever architectural decisions like attention and MoE, complex training and alignment strategies like RLHF, and ongoing work in model interpretability, safety, and deployment. The scale of the engineering challenge is immense, but so is the potential. With proper safeguards and continued research, LLMs have the capacity to transform industries—from education and healthcare to law and scientific discovery—while also raising new ethical and societal questions that we must answer with care."
    # input_string = "Hi, who are you?"
    input_ids = [(idx, tokenizer.encode(input_string)[:128]) for idx in range(batch_size)]


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
    weight_map_amd = "/work1/kasikci/kanzhu/models/llama3-70b"
    pipeline.init_set_weight(weight_map_amd, cached=True)

    print("finish update pipeline")

    world_size = pipeline.num_devices
    print(f"Number of GPUs: {world_size}")

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    command = Value('i', 1)    # 'i' stands for a signed integer.
    shared_int = Value('i', 0)    # 'i' stands for a signed integer.
    shared_batch_size = Value('i', 0)    # 'i' stands for a signed integer.
    shared_array = Array('i', batch_size)  # An array of integers with length equal to world_size.

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
    for i in range(batch_size // 2):
        output_strings[i].append(shared_array[i])

    iterations = 3
    #     print("input_ids: ", i)
    # For each iteration, update the shared integer, synchronize with the workers,
    # and let them compute and write their results.
    command.value = 2
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        # Set the shared task value.
        barrier.wait()

        barrier.wait()
        for i in range(batch_size):
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

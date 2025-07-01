import sys, os
import torch
import argparse
sys.path.append("../")
sys.path.append("../utils")
sys.path.append('../pybind/build')

from utils.prof_marker import prof_marker
from utils.frontend import requestManager
from utils.util_functions import prepare_weight
from transformers import AutoTokenizer
from input_test import prefill_context

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# from models.llama3_KVCacheTorch import Pipeline
from models.llama3_KVCacheFA import Pipeline

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-l", "--load_hf_weight", action="store_true", help="Load weights from huggingface")

args = arg_parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# input_strings = ["Hi, who are you?"]
# input_strings = ["Hi, who are you?", "What's the weather today?"]
input_strings = [ "Hi, who are you?" for _ in range(384)]
# input_strings = [ "The university of washington is located in" for _ in range(16)]
input_ids = [tokenizer.encode(s) for s in input_strings]
# print(input_ids)

# request_queue = []
# request_manager = requestManager(args.trace_path, "meta-llama/Meta-Llama-3-8B-Instruct")
# request_manager.read_request()
# request_manager.release_request()
# # print(request_manager.available_request_queue)

global_batch_size = 1024
decode_batch_size = 384
# new_input_ids = []
# for req in request_manager.available_request_queue:
#     print("req.idx: ", req.req_idx)
#     print("req.prompt: ", req.prompt)
#     print("req.output_len: ", req.output_len)
#     new_input_ids.append((req.req_idx, req.prompt))
# print("new_input_ids: ", new_input_ids)


decode_inputs_ids = [(i, input_ids[i]) for i in range(decode_batch_size)]

prefill_context = "Large Language Models (LLMs) have rapidly evolved from simple sequence-to-sequence systems into sophisticated architectures capable of human-level language understanding and generation. The architecture underpinning most of these models is the Transformer, introduced in the seminal paper “Attention is All You Need” by Vaswani et al. in 2017. Since then, the Transformer has served as the foundation for models such as GPT-3, PaLM, LLaMA, and Claude. These models, especially those with tens or hundreds of billions of parameters, rely on innovations in model scaling, parallelism, training data curation, and optimization techniques. At the heart of the Transformer is the attention mechanism. This allows the model to weigh and focus on different parts of the input sequence dynamically. Self-attention, in particular, enables the model to capture contextual relationships between words irrespective of their distance in the sequence. This is a dramatic departure from earlier RNNs and LSTMs, which processed sequences linearly and struggled with long-range dependencies. Self-attention operates in parallel, which is not only more computationally efficient but also critical for scaling up models. As models scaled, new challenges emerged—most notably, the explosion in compute requirements. Training GPT-3, for example, required hundreds of petaflop-days of compute and massive datasets scraped from the internet. To handle such computational demands, researchers employed techniques like model parallelism, data parallelism, pipeline parallelism, and optimizer sharding. Frameworks such as DeepSpeed and Megatron-LM became crucial in managing training across thousands of GPUs. However, bigger isn’t always better. While adding more parameters generally improves model capacity, it also introduces inefficiencies. This led to the rise of Mixture of Experts (MoE) models, where only a subset of the model’s parameters are active at a time. Instead of activating all parts of a monolithic network, an MoE model uses a gating mechanism to route inputs to a few selected “experts.” Each expert is a small feedforward network, and typically only 2–4 experts out of dozens or hundreds are used per token. This dramatically reduces the compute cost while keeping the overall model capacity high. Google's GLaM and Switch Transformer are notable implementations of this approach. Training such models also brings challenges like expert imbalance, where some experts are overused while others are underutilized. This is addressed using auxiliary losses that encourage balanced routing. Another complication is that the gating mechanism can become a bottleneck if it is not efficiently implemented, especially at scale. Another key component of LLM development is data. Large-scale training data must be diverse, representative, and relatively clean. Curating such datasets involves filtering out low-quality content, removing duplicate web pages, decontaminating evaluation sets, and ensuring that personal or harmful information is minimized. While some models are trained on curated corpora like The Pile or Common Crawl, others use proprietary datasets augmented with internal or human-labeled content. Instruct-tuning and reinforcement learning from human feedback (RLHF) further refine these models to align their outputs with human preferences. The emergence of RLHF as a fine-tuning method is particularly transformative. After a base model is trained to predict the next token, it’s further fine-tuned using rankings or preferences from human annotators. This process helps the model generate more helpful, harmless, and honest responses. OpenAI's ChatGPT and Anthropic's Claude use similar RLHF-based training pipelines. But alignment isn’t just a technical problem—it’s also philosophical. How should a model behave in ambiguous moral situations? Who decides what is “helpful” or “safe”? These are open questions that the AI research community, ethicists, and policymakers must grapple with. And as models become capable of multimodal reasoning—processing not just text but also images, audio, and video—the alignment problem becomes even more complex. Deployment of LLMs also poses engineering and infrastructure challenges. Serving a model like GPT-4 or Claude at scale requires careful attention to latency, cost-efficiency, caching strategies, and prompt engineering. Techniques like speculative decoding, retrieval-augmented generation (RAG), prompt compression, and system message templating are employed to make inference faster and more relevant. The future of LLMs appears to be moving in several directions simultaneously: larger models, more efficient smaller models, multimodal models, and models that can interact with tools and APIs. Tool-use in LLMs allows them to perform tasks like math, code execution, web search, or database queries by invoking external systems—essentially combining reasoning with action. This trend bridges the gap between static language modeling and interactive AI agents. In summary, the evolution of large language models is not just a story of bigger models and faster GPUs. It's a story of clever architectural decisions like attention and MoE, complex training and alignment strategies like RLHF, and ongoing work in model interpretability, safety, and deployment. The scale of the engineering challenge is immense, but so is the potential. With proper safeguards and continued research, LLMs have the capacity to transform industries—from education and healthcare to law and scientific discovery—while also raising new ethical and societal questions that we must answer with care."

prefill_context_ids = tokenizer.encode(prefill_context * 10) # which length is 1066.

# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
weight_map_amd_kan = "/app/models/llama-8b"
if args.load_hf_weight:
    pipeline_dict = {
        "cuda:0" : Pipeline()
    }
    prepare_weight(pipeline_dict, weight_map_amd_kan)

pipeline = Pipeline()
if args.load_hf_weight:
    pipeline_dict = {
        "cuda:0" : Pipeline()
    }
    pipeline.init(weight_map_amd_kan, cached=False)

pipeline.init(weight_map_amd_kan, cached=True)

# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# reserved_memory = torch.cuda.memory_reserved(device)
# print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
# pipeline.config()
def test_performance():
    input_length = 1024
    prefill_input_ids = [prefill_context_ids[:input_length] for _ in range(1000)]
    input_length = 1024
    prefill_input_ids = [prefill_context_ids[:input_length] for _ in range(1000)]
    output_strings = {}
    # initialize the reqs for first 384 requests
    decode_inputs = []
    for i in range(decode_batch_size):
        input = [(i, prefill_input_ids[i])]
        pipeline.update(input)
        output_strings[i] = prefill_input_ids[i]
        new_tokens = pipeline.run()
        for _, new_token in new_tokens:
            output_strings[i].extend(new_token)
        decode_inputs.extend(new_tokens)
        print("new_tokens: ", new_tokens)

    output_strings[decode_batch_size] = prefill_input_ids[decode_batch_size]
    decode_inputs.extend([(decode_batch_size, prefill_input_ids[decode_batch_size])])
    pipeline.update(decode_inputs, decode_batch_size)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
      for i in range(decode_batch_size, decode_batch_size + 5):
        print("Cycle: ", i - decode_batch_size)
        next_prefill_idx = i + 1
        new_tokens = pipeline.run()
        with prof_marker(f"after_execute_step_4"):
            for req_idx, new_token in new_tokens:
                output_strings[req_idx].extend(new_token)
        # print("new_tokens: ", new_tokens)
        with prof_marker(f"after_execute_step_5"):
            new_tokens = new_tokens[:-1]
            decode_batchsize = len(new_tokens)
            assert decode_batchsize == decode_batch_size
        with prof_marker(f"after_execute_step_6"):
            output_strings[next_prefill_idx] = prefill_input_ids[next_prefill_idx]
        with prof_marker(f"after_execute_step_7"):
            new_tokens.extend([(next_prefill_idx, prefill_input_ids[next_prefill_idx])])
        with prof_marker(f"after_execute_step_8"):
            pipeline.update(new_tokens, decode_batchsize)

    output_text = tokenizer.batch_decode(list(output_strings.values())[:1], skip_special_tokens=True)
    print(output_text)
    prof.export_chrome_trace("llama3_8b_mi300_384x1024_overlap_mask_256_48_288_16.json")

def test_correctness(use_kv_cache=True):
    special_inputs_0 = [(0, input_ids[0]), (1, input_ids[1])]
    special_inputs_1 = [(2, input_ids[2]), (3, input_ids[3])]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    for idx, tensor in special_inputs_1:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 2
    
    # print("new_tokens: ", new_tokens)
    if use_kv_cache:
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(0, output_strings[0]), (1, output_strings[1])]
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, 0)

    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 4
    # print("new_tokens: ", new_tokens)

    if use_kv_cache:
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(i, output_strings[i]) for i in range(4)]
        pipeline.update(new_tokens, 0)

    for i in range(20):
        print("Cycle: ", i)
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_batchsize = len(new_tokens)
        assert decode_batchsize == 4
        # print("new_tokens: ", new_tokens)
        if use_kv_cache:
            pipeline.update(new_tokens, decode_batchsize)
        else:
            new_tokens = [(i, output_strings[i]) for i in range(4)]
            pipeline.update(new_tokens, 0)

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def test_one_cycle():
    special_inputs_0 = [(0, input_ids[0]), (1, input_ids[1])]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    
    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def profile_one_cycle():
    pipeline.init_profile_data()

    stream_names = [ f"TEST_{i}" for i in range(1, 11) ]
    for stream_name in stream_names:
        print(f"Stream: {stream_name}")

        pipeline.batch_size = None
        # test for prefill
        total_batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        # total_batch_sizes = [1024]
        for idx, total_batch_size in enumerate(total_batch_sizes):
            input = [(decode_batch_size + idx, prefill_context_ids[:total_batch_size])]

            pipeline.update(input, is_profile=True, stream_name=stream_name)
            pipeline.profile_run()

        # test for decode
        total_batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384]
        # total_batch_sizes = [384]
        # prepare the decode inputs for a special input_length
        input_length = 1024
        output_length = 512
        prefill_input_ids = [prefill_context_ids[:input_length] for _ in range(1000)]

        pipeline.batch_size = None
        for total_batch_size in total_batch_sizes:
            decode_inputs = []
            pipeline.reset_kv_cache()
            # pipeline.config_algorithm()
            # initialize the reqs for first {total_batch_size} requests
            for i in range(total_batch_size):
                input = [(i, prefill_input_ids[i])]
                pipeline.update(input)
                new_tokens = pipeline.run()
                decode_inputs.extend(new_tokens)
                print("new_tokens: ", new_tokens)
                print("total_batch_size: ", total_batch_size)

            pipeline.batch_size = None
            # decode profiling from input_length to input_length + output_length
            for i in range(output_length + 1):
                print("Cycle: ", i)
                pipeline.update(decode_inputs, total_batch_size, is_profile=True, stream_name=stream_name)
                if i % 128 == 0:
                    pipeline.profile_run()

    # pipeline.profile_print()

# test_correctness()
# test_correctness(use_kv_cache=False)
test_performance()
# test_one_cycle()
# profile_one_cycle()
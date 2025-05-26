import sys, os
import torch
import argparse
sys.path.append("../")
sys.path.append("../utils")
sys.path.append('../pybind/build')

# os.environ["HF_HOME"] = "/storage/ziren/framework-test/hf"
from utils.prof_marker import prof_marker
from utils.frontend import requestManager
from transformers import AutoTokenizer

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# from models.llama3_NoKVCacheTorch import Pipeline
# from models.llama3_KVCacheTorch import Pipeline
from models.llama3_KVCacheFA import Pipeline
# from models.llama3_FlashinferKVCache import Pipeline

# arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("-t", "--trace_path", type=str, required=True, help="Request trace to read from")

# args = arg_parser.parse_args()

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

batch_size = 256
prefill_length = 128
decode_length = 16

# new_input_ids = []
# for req in request_manager.available_request_queue:
#     print("req.idx: ", req.req_idx)
#     print("req.prompt: ", req.prompt)
#     print("req.output_len: ", req.output_len)
#     new_input_ids.append((req.req_idx, req.prompt))
# print("new_input_ids: ", new_input_ids)

prefill_context = "Large Language Models (LLMs) have rapidly evolved from simple sequence-to-sequence systems into sophisticated architectures capable of human-level language understanding and generation. The architecture underpinning most of these models is the Transformer, introduced in the seminal paper “Attention is All You Need” by Vaswani et al. in 2017. Since then, the Transformer has served as the foundation for models such as GPT-3, PaLM, LLaMA, and Claude. These models, especially those with tens or hundreds of billions of parameters, rely on innovations in model scaling, parallelism, training data curation, and optimization techniques. At the heart of the Transformer is the attention mechanism. This allows the model to weigh and focus on different parts of the input sequence dynamically. Self-attention, in particular, enables the model to capture contextual relationships between words irrespective of their distance in the sequence. This is a dramatic departure from earlier RNNs and LSTMs, which processed sequences linearly and struggled with long-range dependencies. Self-attention operates in parallel, which is not only more computationally efficient but also critical for scaling up models. As models scaled, new challenges emerged—most notably, the explosion in compute requirements. Training GPT-3, for example, required hundreds of petaflop-days of compute and massive datasets scraped from the internet. To handle such computational demands, researchers employed techniques like model parallelism, data parallelism, pipeline parallelism, and optimizer sharding. Frameworks such as DeepSpeed and Megatron-LM became crucial in managing training across thousands of GPUs. However, bigger isn’t always better. While adding more parameters generally improves model capacity, it also introduces inefficiencies. This led to the rise of Mixture of Experts (MoE) models, where only a subset of the model’s parameters are active at a time. Instead of activating all parts of a monolithic network, an MoE model uses a gating mechanism to route inputs to a few selected “experts.” Each expert is a small feedforward network, and typically only 2–4 experts out of dozens or hundreds are used per token. This dramatically reduces the compute cost while keeping the overall model capacity high. Google's GLaM and Switch Transformer are notable implementations of this approach. Training such models also brings challenges like expert imbalance, where some experts are overused while others are underutilized. This is addressed using auxiliary losses that encourage balanced routing. Another complication is that the gating mechanism can become a bottleneck if it is not efficiently implemented, especially at scale. Another key component of LLM development is data. Large-scale training data must be diverse, representative, and relatively clean. Curating such datasets involves filtering out low-quality content, removing duplicate web pages, decontaminating evaluation sets, and ensuring that personal or harmful information is minimized. While some models are trained on curated corpora like The Pile or Common Crawl, others use proprietary datasets augmented with internal or human-labeled content. Instruct-tuning and reinforcement learning from human feedback (RLHF) further refine these models to align their outputs with human preferences. The emergence of RLHF as a fine-tuning method is particularly transformative. After a base model is trained to predict the next token, it’s further fine-tuned using rankings or preferences from human annotators. This process helps the model generate more helpful, harmless, and honest responses. OpenAI's ChatGPT and Anthropic's Claude use similar RLHF-based training pipelines. But alignment isn’t just a technical problem—it’s also philosophical. How should a model behave in ambiguous moral situations? Who decides what is “helpful” or “safe”? These are open questions that the AI research community, ethicists, and policymakers must grapple with. And as models become capable of multimodal reasoning—processing not just text but also images, audio, and video—the alignment problem becomes even more complex. Deployment of LLMs also poses engineering and infrastructure challenges. Serving a model like GPT-4 or Claude at scale requires careful attention to latency, cost-efficiency, caching strategies, and prompt engineering. Techniques like speculative decoding, retrieval-augmented generation (RAG), prompt compression, and system message templating are employed to make inference faster and more relevant. The future of LLMs appears to be moving in several directions simultaneously: larger models, more efficient smaller models, multimodal models, and models that can interact with tools and APIs. Tool-use in LLMs allows them to perform tasks like math, code execution, web search, or database queries by invoking external systems—essentially combining reasoning with action. This trend bridges the gap between static language modeling and interactive AI agents. In summary, the evolution of large language models is not just a story of bigger models and faster GPUs. It's a story of clever architectural decisions like attention and MoE, complex training and alignment strategies like RLHF, and ongoing work in model interpretability, safety, and deployment. The scale of the engineering challenge is immense, but so is the potential. With proper safeguards and continued research, LLMs have the capacity to transform industries—from education and healthcare to law and scientific discovery—while also raising new ethical and societal questions that we must answer with care."

prefill_input_ids = [tokenizer.encode(prefill_context)[:prefill_length] for _ in range(batch_size)]

weight_map_wzr = "/storage/ziren/framework-test/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
weight_map_amd_kan = "/work1/kasikci/kanzhu/models/llama3-8b"
# weight_map_yi = "/root/llama3-8b"

pipeline = Pipeline(max_seq_len=150)
pipeline.init(weight_map_wzr, cached=True)

def main():
    special_inputs = [(i, prefill_input_ids[i]) for i in range(batch_size)]
    output_strings = {}
    for idx, tensor in special_inputs:
        output_strings[idx] = tensor

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
    ) as prof:
        pipeline.update(special_inputs)
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_batchsize = len(new_tokens)
        assert decode_batchsize == batch_size

        pipeline.update(new_tokens, decode_batchsize)

        for i in range(decode_length):
            print("Cycle: ", i)
            new_tokens = pipeline.run()
            for req_idx, new_token in new_tokens:
                output_strings[req_idx].extend(new_token)
            decode_batchsize = len(new_tokens)
            assert decode_batchsize == batch_size
            pipeline.update(new_tokens, decode_batchsize)

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)
    prof.export_chrome_trace("llama3_8b_mi300_trace.json")

if __name__ == "__main__":
    main()

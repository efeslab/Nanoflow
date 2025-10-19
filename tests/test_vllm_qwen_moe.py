"""Benchmark offline inference throughput for vLLM.

Example Usage:
  export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
  CUDA_VISIBLE_DEVICES=0,1 python test_vllm_qwen_moe.py 
"""

import transformers
import vllm
from vllm import LLM, SamplingParams

from vllm.inputs import TokensPrompt


def main():

    from nanoflow.utils.input_test import prefill_context

    seq_len = 1024
    # seq_len = 2048
    # global_batch_size = 1024
    global_batch_size = 2048
    # global_batch_size = 3072
    # decode_batch_size = 128
    decode_batch_size = 640
    # decode_batch_size = 1280
    prefill_batch_size = global_batch_size - decode_batch_size

    model_name = "Qwen/Qwen2-57B-A14B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=2, enable_expert_parallel=True)

    prefill_context_ids = tokenizer.encode(prefill_context)
    assert seq_len <= len(
            prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]


    prompts = [TokensPrompt(prompt_token_ids=prefill_input_ids) for _ in range(prefill_batch_size)]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=20))

    print(outs[0].outputs[0].text)

if __name__ == "__main__":
    main()
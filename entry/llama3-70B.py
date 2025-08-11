#!/usr/bin/env python
# pip install "transformers>=4.41.0" accelerate sentencepiece \
#             --extra-index-url https://download.pytorch.org/whl/cu121
# For 4‑bit:  pip install bitsandbytes

import torch
import os

os.environ["HF_HOME"] = "/code/hf"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GenerationConfig,
)

from input_test import prefill_context

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # requires agreeing to Meta’s license
# MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"  # requires agreeing to Meta’s license

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Load model (BF16/FP16, spread across all visible GPUs; falls back to CPU)
model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,   # or torch.float16 if GPUs lack BF16
    device_map="auto",            # split layers across GPUs automatically
    use_safetensors=True,
)
gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
gen_cfg.do_sample = False
gen_cfg.max_new_tokens = 50

# print(model)

# 3) Encode the prompt
# prompt = "Hi, who are you?"
prompt = prefill_context
seq_len = 1024
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs["input_ids"] = inputs["input_ids"][:, :seq_len]  # truncate to seq_len

# 4) Generate 20 new tokens
outputs = model.generate(
    **inputs,
    generation_config=gen_cfg,
)

# 5) Decode & print
print("\n=== Model reply ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# test_tensor = torch.load("./test_data/70B_test_flashinfer_0_folder/LayerNormAttn_4_input")

# layer = model.model.layers[4]

# input_layernorm = layer.input_layernorm
# print("test_result")
# print(input_layernorm(test_tensor))
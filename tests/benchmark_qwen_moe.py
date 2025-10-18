#!/usr/bin/env python
# pip install "transformers>=4.41.0" accelerate sentencepiece \
#             --extra-index-url https://download.pytorch.org/whl/cu121
# For 4‑bit:  pip install bitsandbytes

import torch
import os
import sys

sys.path.append("../")
os.environ["HF_HOME"] = "/code/hf"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2MoeForCausalLM,
    GenerationConfig,
)

from nanoflow.utils.input_test import prefill_context

# MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Load model (BF16/FP16, spread across all visible GPUs; falls back to CPU)
model = Qwen2MoeForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,  # or torch.float16 if GPUs lack BF16
    device_map="auto",  # split layers across GPUs automatically
    use_safetensors=True,
)

gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
gen_cfg.do_sample = False
gen_cfg.max_new_tokens = 1
print("gen_cfg:", gen_cfg)

print(model)

# 3) Encode the prompt
prompt = "Hi, who are you?"
# prompt = prefill_context
# prompts = [
#     "Hi, who are you?",
#     prefill_context
# ]
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

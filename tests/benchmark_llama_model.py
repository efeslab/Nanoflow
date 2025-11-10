#!/usr/bin/env python
# pip install "transformers>=4.41.0" accelerate sentencepiece \
#             --extra-index-url https://download.pytorch.org/whl/cu121
# For 4‑bit:  pip install bitsandbytes

import torch
import sys

# sys.path.append("../")
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GenerationConfig,
)

# from utils.input_test import prefill_context

# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # requires agreeing to Meta’s license
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"  # requires agreeing to Meta’s license

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Load model (BF16/FP16, spread across all visible GPUs; falls back to CPU)
model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,  # or torch.float16 if GPUs lack BF16
    device_map="auto",  # split layers across GPUs automatically
    use_safetensors=True,
)
gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
gen_cfg.do_sample = False
gen_cfg.max_new_tokens = 20

print(model)

# 3) Encode the prompt
prompt = "Hi, who are you?"
# prompt = prefill_context
seq_len = 1024
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs["input_ids"] = inputs["input_ids"][:, :seq_len]  # truncate to seq_len

# 4) Generate 20 new tokens
outputs = model.generate(
    **inputs,
    generation_config=gen_cfg,
    output_scores=True,
    return_dict_in_generate=True,
)

sequences = outputs.sequences
scores = outputs.scores
full_ids = sequences[0]
print("shape of scores:", len(scores))

# 5) Decode & print
print("\n=== Model reply ===")
print(tokenizer.decode(full_ids, skip_special_tokens=True))

last_logits = scores[-1][0]                 # shape: [vocab_size]
last_probs = torch.softmax(last_logits, dim=-1)
print("shape of last_probs:", last_probs.shape)
torch.save(last_probs, "gt_llama_last_probs_seq_idx_20.pt")

# 6) Actual final token id and its probability
final_token_id = full_ids[-1].item()
final_token = tokenizer.decode([final_token_id])
final_prob = last_probs[final_token_id].item()

print("\n=== Final token info ===")
print("Final token id:      ", final_token_id)
print("Final token decoded: ", repr(final_token))
print("Probability of final token under last-step dist: {:.6f}".format(final_prob))

import os

# Set the HF_HOME environment variable
os.environ['HF_HOME'] = '/code/hf'
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaForCausalLM
# import torch



model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# Create a partial model with a fraction of layers, e.g., the first 12 layers


# Prepare input
input_text = "Hi, "
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']
input_ids = input_ids

output_text = tokenizer.decode([1, 29915, 29885, 1811], skip_special_tokens=True)
print(output_text)

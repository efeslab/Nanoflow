
import os

# Set the HF_HOME environment variable
os.environ['HF_HOME'] = '/code/hf'
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaForCausalLM
import torch
# import torch



model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load the full model
model = LlamaForCausalLM.from_pretrained(model_name_or_path)


# Create a partial model with a fraction of layers, e.g., the first 12 layers


# Prepare input
input_text = "Hi, I'm"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']
print(input_ids.size())

# Perform inference and decode the output
output = model.generate(
    input_ids,
    max_new_tokens=5,
    do_sample=True,        # Activate sampling
    top_k=1,               # Use max sampling (greedy sampling with randomness)
    temperature=1.0        # Optional: Control randomness. Higher values give more diversity.
)
print(output)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
print("input_ids", input_ids)


import os

# Set the HF_HOME environment variable
os.environ['HF_HOME'] = '/code/hf'
from transformers import  AutoTokenizer
# import torch



model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# Create a partial model with a fraction of layers, e.g., the first 12 layers


# Prepare input
input_text = "To plan the visit to Seattle, you need to "
inputs = tokenizer(input_text, return_tensors="pt")

print(inputs)
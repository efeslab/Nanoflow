import sys, os
import torch

sys.path.append("../")
sys.path.append('../pybind/build')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.llama3 import Pipeline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
input_strings = ["Hi, who are you?"]
input_ids = [tokenizer.encode(s) for s in input_strings]
print(input_ids)


pipeline = Pipeline()
pipeline.init("/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# reserved_memory = torch.cuda.memory_reserved(device)
# print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
pipeline.config()
pipeline.update(input_ids)
pipeline.run(output_length=20)


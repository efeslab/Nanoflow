import sys, os
import torch
sys.path.append("../")
sys.path.append('../pybind/build')
from utils.prof_marker import prof_marker
from transformers import AutoTokenizer



# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from models.llama3_NoKVCacheTorch import Pipeline
from models.llama3_KVCacheTorch import Pipeline
# from models.llama3 import Pipeline
# from models.llama3_FlashinferKVCache import Pipeline

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# input_strings = ["Hi, who are you?"]
# input_strings = ["Hi, who are you?", "What's the weather today?"]
input_strings = [ "Hi, who are you?" for _ in range(16)]
# input_strings = [ "The university of washington is located in" for _ in range(16)]
input_ids = [tokenizer.encode(s) for s in input_strings]
print(input_ids)


pipeline = Pipeline()
pipeline.init("/work1/kasikci/kanzhu/models/llama3-8b")

# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# reserved_memory = torch.cuda.memory_reserved(device)
# print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
# pipeline.config()
pipeline.update(input_ids)

output_strings = []
for i in input_ids:
    output_strings.append(i)
#     print("input_ids: ", i)

output_length=20

for i in range(output_length):
    new_tokens = pipeline.run()
    with prof_marker("post_run_stage"):
        for i, item in enumerate(new_tokens):
            output_strings[i].append(item[0])
    with prof_marker("update_stage"):
        # pipeline.update(output_strings)
        pipeline.update(new_tokens, decode_flag=True)
    

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
output_text = tokenizer.batch_decode(output_strings[:1], skip_special_tokens=True)
print(output_text)

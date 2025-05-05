import sys, os
import torch
sys.path.append("../")
sys.path.append('../pybind/build')
from utils.prof_marker import prof_marker
from transformers import AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ]
)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# from models.llama3_NoKVCacheTorch import Pipeline
# from models.llama3_KVCacheTorch import Pipeline
from models.llama3_KVCacheFA import Pipeline
# from models.llama3_FlashinferKVCache import Pipeline

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# input_strings = ["Hi, who are you?"]
# input_strings = ["Hi, who are you?", "What's the weather today?"]
# input_strings = [
#                     "Hi, who are you?", 
#                     "What's the weather today?",
#                     "The university of washington is located in seattle",
#                     "I am a student at the university of washington"
# ]
input_strings = ["Hi, who are you?"] * 1
# input_strings = [ "The university of washington is located in" for _ in range(16)]
input_ids = [tokenizer.encode(s) for s in input_strings]
# for input_id in input_ids:
#     while len(input_id) < 256:
#         input_id.append(input_id[-1])
print(input_ids)

weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
weight_map_amd_kan = "/work1/kasikci/kanzhu/models/llama3-8b"
weight_map_yi = "/root/llama3-8b"

pipeline = Pipeline()
pipeline.init(weight_map_yi)

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

output_length=10
torch.cuda.current_stream().synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    # with_stack=True,
    profile_memory=True,
) as prof:
    for i in range(output_length):
        with prof_marker(f"running_{i}"):
            new_tokens = pipeline.run()
            print(new_tokens)
        with prof_marker("post_run_stage"):
            for i, item in enumerate(new_tokens):
                output_strings[i].append(item[0])
        with prof_marker("update_stage"):
            # pipeline.update(output_strings)
            pipeline.update(new_tokens, decode_flag=True)
prof.export_chrome_trace("trace_fa_fused_copy.json")

output_text = tokenizer.batch_decode(output_strings[:1], skip_special_tokens=True)
print(output_text)

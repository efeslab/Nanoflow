import os
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
# Set the GPU to use (e.g., use only GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["HF_HOME"] = "/code/hf"
# make a folder ./gt_out here, if not exists
os.makedirs("./gt_out", exist_ok=True)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
# print(model.config)
# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

model.half()
model.config.num_hidden_layers = 3

input_strings = ["Hi, who are you?"]
input_ids = [tokenizer(s, return_tensors="pt").input_ids for s in input_strings]
input_ids = [ids.to(torch.int32) for ids in input_ids] 

print("[GlobalInput_0_tokens]")
# print(input_ids[0])
# use tensor save to save the input_ids[0] into /gt_out/GlobalInput_0_tokens
torch.save(input_ids[0][0], "./gt_out/GlobalInput_0_tokens")



outputs = model(input_ids[0])

logits = outputs.logits
torch.save(logits, "./gt_out/GetLogits_" + str(model.config.num_hidden_layers - 1) + "_D")
# print(logits.dtype)

tokens = torch.argmax(logits, dim=-1)

torch.save(tokens.to(dtype=torch.int32)[0], "./gt_out/Sampling_" + str(model.config.num_hidden_layers - 1) + "_tokens")

# print(tokens[0])
output_text = tokenizer.decode(tokens[0][-1], skip_special_tokens=True)
print(output_text)

# last_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
# # Get the most probable next token (argmax sampling)
# next_token_id = torch.argmax(last_token_logits, dim=-1)  # Shape: (1,)

# # Decode the token to get the first generated word
# next_word = tokenizer.decode(next_token_id)

# print("First generated word:", next_word)


# outputs_ids = model.generate(input_ids[0], max_length=50)
# print(output_ids)

# # Decode and print the result
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


f = open("lmsys.csv", "w")
f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")

num_samples = 100000

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("lmsys/lmsys-chat-1m")

# sample num_samples samples from the dataset randomly
dataset = dataset.shuffle(seed=42)
dataset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-13b-16k")

collected_samples = 0
for i in tqdm(range(len(dataset))):
    if len(dataset[i]["conversation"]) < 2:
        continue
    if dataset[i]["conversation"][0]["role"] != "user" or dataset[i]["conversation"][1]["role"] != "assistant":
        continue
    input_tokens = tokenizer.encode(dataset[i]["conversation"][0]['content'])
    output_tokens = tokenizer.encode(dataset[i]["conversation"][1]['content'])
    f.write(f"XXX,{len(input_tokens)},{len(output_tokens)}\n")
    collected_samples += 1
    if collected_samples >= num_samples:
        break

#   ContextTokens       92.20294
#   GeneratedTokens    207.39722
#   dtype: float64
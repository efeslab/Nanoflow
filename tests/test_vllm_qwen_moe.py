import transformers
import vllm
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen1.5-MoE-A2.7B"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, max_model_len=1024)

prompts = ["Hi, who are you?"]
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=20))

print(outs[0].outputs[0].text)
import sys
import os

prompt = input("Enter the prompt: ")
decode_len = int(input("Enter the decode length: "))
request_num = int(input("Enter the request number: "))
prefill_len = -1

# get first word of prompt
first_word = prompt.split(' ')[0]

# write into upper level directory
os.makedirs("test_traces", exist_ok=True)

with open(f"test_traces/{first_word}-{decode_len}-{request_num}.csv", 'w') as f:
    #summary in the first line
    for i in range(request_num):
        f.write(f"{i},{prefill_len},{decode_len},{prompt}\n")
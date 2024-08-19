import sys

prompt = input("Enter the prompt: ")
decode_len = int(input("Enter the decode length: "))
request_rate = int(input("Enter the request rate: "))
prefill_len = -1

# get first word of prompt
first_word = prompt.split(' ')[0]

if request_rate == 0:
    request_interval = 0
else:
    request_interval = 1 / request_rate

with open(f"{first_word}-{decode_len}-{request_rate}.csv", 'w') as f:
    for i in range(10000):
        f.write(f"{i},{prefill_len},{decode_len},{request_interval*i}, {prompt}\n")
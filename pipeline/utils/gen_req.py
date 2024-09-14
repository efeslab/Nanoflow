import sys

prompt = sys.argv[1]
decode_len = int(sys.argv[2])
request_rate = int(sys.argv[3])
output_name = sys.argv[4]
prefill_len = -1

# get first word of prompt
first_word = prompt.split(' ')[0]

if request_rate == 0:
    request_interval = 0
else:
    request_interval = 1 / request_rate

with open(f"{output_name}", "w") as f:
    for i in range(100000):
        f.write(f"{i},{prefill_len},{decode_len},{request_interval*i}, {prompt}\n")
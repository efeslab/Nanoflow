import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sys

# read first argument as resdir
resdir = sys.argv[1]

json_res = {
    # offline throughput with const input and output len
    "offline_throughput_const": {
        "512-512": 0,
        "1024-512": 0,
        "512-1024": 0,
    },
    # offline throughput with input and output len from real trace
    "offline_throughput_real_trace": {
        "splitwise": 0,
        "lmsys": 0,
        "sharegpt": 0,
    },
    # online throughput
    "online_throughput": {
        "splitwise": {
            "request_rate": [],
            "normalized_latency": [],
        },
        "lmsys": {
            "request_rate": [],
            "normalized_latency": [],
        },
        "sharegpt": {
            "request_rate": [],
            "normalized_latency": [],
        }
    }
}


# column format: total_time,token_per_second,token_per_second_per_gpu,total_cycle,cycle_time,average_ttft,average_tpot,average_normalize_latency

for name in ["512-512", "1024-512", "512-1024"]:
    df = pd.read_csv(f"{resdir}/offline-{name}.csv")
    # get token_per_second_per_gpu
    json_res["offline_throughput_const"][name] = df["token_per_second_per_gpu"][0].astype(float)

for name in ["splitwise", "lmsys", "sharegpt"]:
    df = pd.read_csv(f"{resdir}/offline-{name}.csv")
    # get token_per_second_per_gpu
    json_res["offline_throughput_real_trace"][name] = df["token_per_second_per_gpu"][0].astype(float)
    
for name in ["splitwise", "lmsys", "sharegpt"]:
    # list all files start with online-{name}- and end with .csv
    files = [f for f in os.listdir(resdir) if f.startswith(f"online-{name}-") and f.endswith(".csv")]
    for file in files:
        df = pd.read_csv(f"{resdir}/{file}")
        request_rate = int(file.split("-")[2].split(".")[0])
        json_res["online_throughput"][name]["request_rate"].append(request_rate)
        json_res["online_throughput"][name]["normalized_latency"].append(df["average_normalize_latency"][0])
        
print(json_res)
# save json_res to a file
json.dump(json_res, open(f"{resdir}/pllm.json", 'w'))

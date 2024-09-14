import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess

target_names = ["vLLM", "DeepSpeed-FastGen", "TensorRT-LLM", "NanoFlow (Ours)"]
settings_1 = [
    "Input 512\nOutput 512",
    "Input 1024\nOutput 512",
    "Input 512\nOutput 1024",
]
ablation_names = ["Non-overlap", "Nanobatch-only", "Nanoflow", "Nanoflow-offload"]
settings_2 = ["Splitwise", "LMSYS-Chat", "ShareGPT"]
model = "llama-2-70b-8GPU"
fontsize = 20
optimal_thorughput = 1857

def read_data(baseline_file_name,pllm_file_name):
    with open(baseline_file_name, "r") as f:
        data = json.load(f)
    offline_throughput_const=dict()
    offline_throughput_real_trace=dict()
    request_rates=dict()
    normalized_latency=dict()
    ablation_throughput=dict()
    for i in range(3):
        offline_throughput_const[target_names[i]]=list(data[target_names[i]]["offline_throughput_const"].values())
        offline_throughput_real_trace[target_names[i]]=(list(data[target_names[i]]["offline_throughput_real_trace"].values()))
        request_rates[target_names[i]]=dict()
        normalized_latency[target_names[i]]=dict()
        for setting in settings_2:
            request_rates[target_names[i]][setting]=data[target_names[i]]["online_throughput"][setting]["request_rate"]
            normalized_latency[target_names[i]][setting]=data[target_names[i]]["online_throughput"][setting]["normalized_latency"]

    with open(pllm_file_name, "r") as f:
        data = json.load(f)
    offline_throughput_const[target_names[-1]]=list(data["offline_throughput_const"].values())
    offline_throughput_real_trace[target_names[-1]]=(list(data["offline_throughput_real_trace"].values()))
    
    for ablation in ablation_names:
        ablation_throughput[ablation]=list(data["ablation"][ablation].values())
    
    request_rates[target_names[-1]]=dict()
    normalized_latency[target_names[-1]]=dict()
    for setting in settings_2:
        request_rates[target_names[-1]][setting]=data["online_throughput"][setting]["request_rate"]
        normalized_latency[target_names[-1]][setting]=data["online_throughput"][setting]["normalized_latency"]

    return offline_throughput_const,offline_throughput_real_trace,request_rates,normalized_latency, ablation_throughput



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=".")
    args = parser.parse_args()
    offline_throughput_const,offline_throughput_real_trace,request_rates,normalized_latency, ablation_throughput=read_data(f"{args.data_dir}/baseline.json",f"{args.data_dir}/pllm.json")

    print("---------------------- Offline throughput constant ---------------------")
    print(offline_throughput_const)
    speedups = []
    for config in range(3):
        max_baseline = 0
        for baseline in range(3):
            max_baseline = max(max_baseline, offline_throughput_const[target_names[baseline]][config])
        speedups.append(offline_throughput_const[target_names[-1]][config]/max_baseline)
    
    print("Speedups", speedups)
    print("Max speedup", max(speedups))
    print("Avg speedup", np.power(np.prod(speedups), 1/3))
    print("Opt percentage", max(offline_throughput_const[target_names[-1]])/optimal_thorughput)
    
    print("---------------------- Offline throughput real ---------------------")
    print(offline_throughput_real_trace)
    speedups = []
    for config in range(3):
        max_baseline = 0
        for baseline in range(3):
            max_baseline = max(max_baseline, offline_throughput_real_trace[target_names[baseline]][config])
        speedups.append(offline_throughput_real_trace[target_names[-1]][config]/max_baseline)
    
    print("Speedups", speedups)
    print("Max speedup", max(speedups))
    print("Avg speedup", np.power(np.prod(speedups), 1/3))
    print("Opt percentage", max(offline_throughput_real_trace[target_names[-1]])/optimal_thorughput)
    
    print("---------------------- Ablation ---------------------")
    print(ablation_throughput)
    ablation_avg = dict()
    for ablation in ablation_names:
        ablation_avg[ablation] = np.mean(ablation_throughput[ablation])
    print("Ablation avg", ablation_avg)
    our_name = "Nanoflow"
    our_avg = np.mean(ablation_throughput[our_name])
    for ablation in ablation_names:
        if ablation == our_name:
            continue
        print(f"{our_name} vs {ablation}", our_avg/ablation_avg[ablation])
    
    # specifically, for Nanoflow and Non-overlap, compare each config
    for config in range(3):
        print(f"{our_name} vs Non-overlap {config}", ablation_throughput[our_name][config]/ablation_throughput["Non-overlap"][config])

    
    
    
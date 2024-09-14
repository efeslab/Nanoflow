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
settings_ablation = [
    "Input 512\nOutput 512",
    "Input 1024\nOutput 512",
    "Input 512\nOutput 1024",
    "Prefill\n Only",
]
ablation_names = ["Non-overlap", "Nanobatch-only", "Nanoflow", "Nanoflow-offload"]
ablatoin_fig_names = ["Non-overlap", "Nanobatch-only", "NanoFlow", "NanoFlow-offload"]
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


def plot_ablation_througput(ablation_throughput, fig_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    target_names = ablation_names
    print(ablation_throughput)
    x = range(len(settings_ablation))
    width = 0.2  # Adjusted the width to fit four sets of bars
    rects1 = ax.bar(
        [p - 1.5 * width for p in x],
        ablation_throughput[target_names[0]],
        width,
        label=ablatoin_fig_names[0],
        color="#385092",
    )
    rects2 = ax.bar(
        [p - 0.5 * width for p in x],
        ablation_throughput[target_names[1]],
        width,
        label=ablatoin_fig_names[1],
        color="#6B83C4",
    )
    rects3 = ax.bar(
        [p + 0.5 * width for p in x],
        ablation_throughput[target_names[2]],
        width,
        label=ablatoin_fig_names[2],
        color="#F2A278",
    )
    rects4 = ax.bar(
        [p + 1.5 * width for p in x],
        ablation_throughput[target_names[3]],
        width,
        label=ablatoin_fig_names[3],
        color="#70C0F9",
    )  # Added Test D bars

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel("Per-GPU Token \nThroughput (tokens/s)", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_ablation, fontsize=fontsize)
    # set y-axis range
    ax.set_ylim([0, 1.2 * max(ablation_throughput[target_names[3]])])
    # set y-axis font size
    ax.tick_params(axis="y", labelsize=fontsize)
    # Attach a text label above each bar, displaying its height.

    # do a legend floating above and outside of plot, and horizontally lay out the legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=16)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)  # Apply labels to Test D bars

    # plt.show()
    plt.savefig(f"{fig_dir}/{model}_ablation_study.pdf", bbox_inches="tight")
    plt.close()

def plot_offline_throughput_const(offline_throughput_const,fig_dir):
    offline_throughput_const[target_names[3]] = offline_throughput_const[target_names[3]][:-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(settings_1))
    width = 0.2  # Adjusted the width to fit four sets of bars
    rects1 = ax.bar(
        [p - 1.5 * width for p in x],
        offline_throughput_const[target_names[0]],
        width,
        label=target_names[0],
        color="#385092",
    )
    rects2 = ax.bar(
        [p - 0.5 * width for p in x],
        offline_throughput_const[target_names[1]],
        width,
        label=target_names[1],
        color="#6B83C4",
    )
    rects3 = ax.bar(
        [p + 0.5 * width for p in x],
        offline_throughput_const[target_names[2]],
        width,
        label=target_names[2],
        color="#70C0F9",
    )
    rects4 = ax.bar(
        [p + 1.5 * width for p in x],
        offline_throughput_const[target_names[3]],
        width,
        label=target_names[3],
        color="#F2A278",
    )  # Added Test D bars

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel("Per-GPU Token \nThroughput (tokens/s)", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_1, fontsize=fontsize)
    # set y-axis range
    ax.set_ylim([0, 1.1 * optimal_thorughput])
    ax.axhline(y=optimal_thorughput, color="r", linestyle="--")
    ax.text(x=max(x)/2+0.25, y=optimal_thorughput, s=f'optimal={optimal_thorughput}', color='r', va='center', ha='right', backgroundcolor='w', fontsize=16)
    # set y-axis font size
    ax.tick_params(axis="y", labelsize=fontsize)
    # Attach a text label above each bar, displaying its height.

    # do a legend floating above and outside of plot, and horizontally lay out the legend
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=fontsize)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)  # Apply labels to Test D bars

    # plt.show()
    plt.savefig(f"{fig_dir}/{model}_offline_throughput.pdf", bbox_inches="tight")
    plt.close()


def plot_offline_throughput_real_trace(offline_throughput_real_trace, fig_dir):
    x = range(len(settings_2))
    width = 0.2  # Adjusted the width to fit four sets of bars

    fig, ax = plt.subplots(figsize=(12, 4))
    rects1 = ax.bar(
        [p - 1.5 * width for p in x],
        offline_throughput_real_trace[target_names[0]],
        width,
        label=target_names[0],
        color="#385092",
    )
    rects2 = ax.bar(
        [p - 0.5 * width for p in x],
        offline_throughput_real_trace[target_names[1]],
        width,
        label=target_names[1],
        color="#6B83C4",
    )
    rects3 = ax.bar(
        [p + 0.5 * width for p in x],
        offline_throughput_real_trace[target_names[2]],
        width,
        label=target_names[2],
        color="#70C0F9",
    )
    rects4 = ax.bar(
        [p + 1.5 * width for p in x],
        offline_throughput_real_trace[target_names[3]],
        width,
        label=target_names[3],
        color="#F2A278",
    )  # Added Test D bars

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel("Per-GPU Token \nThroughput (tokens/s)", fontsize=fontsize)
    # ax.set_title('Llama-2-7B Token Throughput, Single GPU, 500 Requests', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_2, fontsize=fontsize)
    # ax.legend()
    # set y-axis range
    ax.set_ylim([0, 1.1 * optimal_thorughput])
    ax.axhline(y=optimal_thorughput, color="r", linestyle="--")
    ax.text(x=max(x)/2+0.25, y=optimal_thorughput, s=f'optimal={optimal_thorughput}', color='r', va='center', ha='right', backgroundcolor='w', fontsize=16)
    # set y-axis font size
    ax.tick_params(axis="y", labelsize=fontsize)
    # Attach a text label above each bar, displaying its height.

    # do a legend floating above and outside of plot, and horizontally lay out the legend
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=fontsize)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)  # Apply labels to Test D bars

    # plt.show()
    plt.savefig(f"{fig_dir}/{model}_offline_dataset_throughput.pdf", bbox_inches="tight")
    plt.close()


def plot_online_throughput(request_rates, normalized_latency, fig_dir):
    fontsize = 22
    for setting in settings_2:
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 4))
        for target_name in target_names:
            ax.plot(
                request_rates[target_name][setting],
                normalized_latency[target_name][setting],
                "o-",
                label=target_name,
                markersize=8,
            )

        # set font for x and y axis
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # set y-axis unit
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        # ax.set_xticks(request_rates[target_names[3]][setting])
        ax.set_ylim(top=1.0)
        # Customize the plot

        # add a horizontal legend above the plot
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=4, fontsize=16)

        plt.xlabel("Request rate (req/s)", fontsize=fontsize)
        plt.ylabel("Normalized latency \n(s/token)", fontsize=fontsize)
        # plt.title('Llama-2-70B Online Throughput, 4 GPU, TP=4')
        # ax.legend()
        # make grid coarser
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Show the plot
        # plt.show()

        plt.savefig(f"{fig_dir}/{model}-{setting}_online_throughput.pdf", bbox_inches="tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=".")
    parser.add_argument('--fig_dir', type=str, default=".")
    args = parser.parse_args()
    
    offline_throughput_const,offline_throughput_real_trace,request_rates,normalized_latency, ablation_throughput=read_data(f"{args.data_dir}/baseline.json",f"{args.data_dir}/pllm.json")
    plot_offline_throughput_const(offline_throughput_const, args.fig_dir)
    plot_offline_throughput_real_trace(offline_throughput_real_trace, args.fig_dir)
    plot_online_throughput(request_rates, normalized_latency, args.fig_dir)
    plot_ablation_througput(ablation_throughput, args.fig_dir)
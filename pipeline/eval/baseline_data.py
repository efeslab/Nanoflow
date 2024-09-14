import json
import sys
vllm = {
    # offline throughput with const input and output len
    "offline_throughput_const": {
        "512-512": 469,
        "1024-512": 549,
        "512-1024": 346
    },
    # offline throughput with input and output len from real trace
    "offline_throughput_real_trace": {
        "Splitwise": 561,
        "LMSYS-Chat": 272,
        "ShareGPT": 342,
    },
    # online throughput
    "online_throughput": {
        "Splitwise": {
            "request_rate": [1, 2, 3, 4, 5, 6, 7, 8],
            "normalized_latency": [0.029,0.042,0.063,0.125,0.459,0.850,1.291,1.508],
        },
        "LMSYS-Chat": {
            "request_rate": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
            "normalized_latency": [0.025,0.031,0.034,0.038,0.047,0.053,0.068,0.083,0.116,1.087,1.590,1.788,1.909],
        },
        "ShareGPT": {
            "request_rate": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "normalized_latency": [0.026,0.034,0.042,0.050,0.072,0.124,0.406,0.781,1.361,1.608],
        }
    }
}

deepspeed_fastgen = {
    # offline throughput with const input and output len
    "offline_throughput_const": {
        "512-512": 490,
        "1024-512": 513,
        "512-1024": 372
    },
    # offline throughput with input and output len from real trace
    "offline_throughput_real_trace": {
        "Splitwise": 548,
        "LMSYS-Chat": 293,
        "ShareGPT": 335,
    },
    # online throughput
    "online_throughput": {
        "Splitwise": {
            "request_rate": [1, 2, 3, 4, 5, 6, 7, 8],
            "normalized_latency": [0.04, 0.05, 0.07, 0.12, 0.34, 0.61, 0.83, 1.2],
        },
        "LMSYS-Chat": {
            "request_rate": [2, 4, 6, 8, 10, 12],
            "normalized_latency": [0.04, 0.05, 0.06, 0.07, 0.88, 1.2],
        },
        "ShareGPT": {
            "request_rate": [1, 2, 3, 4, 5, 6, 7, 8],
            "normalized_latency": [0.05, 0.05, 0.06, 0.07, 0.14, 0.43, 0.99, 1.29],
        }
    }
}

tensorrt_llm = {
    # offline throughput with const input and output len
    "offline_throughput_const": {
        "512-512": 735,
        "1024-512": 817,
        "512-1024": 636
    },
    # offline throughput with input and output len from real trace
    "offline_throughput_real_trace": {
        "Splitwise": 831,
        "LMSYS-Chat": 560,
        "ShareGPT": 639,
    },
    # online throughput
    "online_throughput": {
        "Splitwise": {
            "request_rate": [1, 2, 3, 4, 5, 6, 10, 11, 12],
            "normalized_latency": [0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 0.67, 0.84, 1.02],
        },
        "LMSYS-Chat": {
            "request_rate": [5, 10, 15, 17, 18, 19, 20],
            "normalized_latency": [0.07, 0.07, 0.08, 0.17, 0.54, 0.83, 1.25],
        },
        "ShareGPT": {
            "request_rate": [2, 4, 6, 10, 13, 15],
            "normalized_latency":[0.03, 0.04, 0.05, 0.11, 0.66, 1.09],
        }
    }

}


with open(f"{sys.argv[1]}/baseline.json", "w") as f:
    baslines={"vLLM":vllm, "DeepSpeed-FastGen":deepspeed_fastgen, "TensorRT-LLM":tensorrt_llm}
    json.dump(baslines, f)
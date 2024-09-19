import json
example_json = {
    # offline throughput with const input and output len
    "offline_throughput_const": {
        "512-512": 1036,
        "1024-512": 1055,
        "512-1024": 1043,
    },
    # offline throughput with input and output len from real trace
    "offline_throughput_real_trace": {
        "Splitwise": 1048,
        "LMSYS-Chat": 990,
        "ShareGPT": 936,
    },
    # online throughput
    "online_throughput": {
        "Splitwise": {
            "request_rate": [5, 10, 15, 20],
            "normalized_latency": [0.2, 0.4, 0.6, 0.8],
        },
        "LMSYS-Chat": {
            "request_rate": [5, 10, 15, 20],
            "normalized_latency": [0.2, 0.4, 0.6, 0.8],
        },
        "ShareGPT": {
            "request_rate": [5, 10, 15, 20],
            "normalized_latency": [0.2, 0.4, 0.6, 0.8],
        }
    }
}

print(json.dumps(example_json, indent=4))
import json
import csv
import os
from datetime import datetime

base_dir = "./"
save_dir = base_dir + "eval_results/"
offline_const = base_dir + "eval-fix-offline/results/"
offline_real = base_dir + "eval-real-offline/results/"
online_batches = [768,1024,2048]
online_paths = [base_dir + f"eval-real-online-{batch}/results/" for batch in online_batches]
ablation = base_dir + "eval-ablation/results/"

example_json = {
    "offline_throughput_const": {
        "512-512": None,
        "1024-512": None,
        "512-1024": None,
        "512-2": None
    },
    "offline_throughput_real_trace": {
        "Splitwise": None,
        "LMSYS-Chat": None,
        "ShareGPT": None,
    },
    "online_throughput": {
        "Splitwise": {
            "request_rate": [],
            "normalized_latency": [],
        },
        "LMSYS-Chat": {
            "request_rate": [],
            "normalized_latency": [],
        },
        "ShareGPT": {
            "request_rate": [],
            "normalized_latency": [],
        }
    },
    "ablation": {
        "Nanobatch-only": {
            "512-512": None,
            "1024-512": None,
            "512-1024": None,
            "512-2": None
        },
        "Non-overlap": {
            "512-512": None,
            "1024-512": None,
            "512-1024": None,
            "512-2": None
        },
        "Nanoflow": {
            "512-512": None,
            "1024-512": None,
            "512-1024": None,
            "512-2": None
        },
        "Nanoflow-offload": {
            "512-512": None,
            "1024-512": None,
            "512-1024": None,
            "512-2": None
        }
    }
}

def create_directory():
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Result directory '{directory}' created successfully.")
    else:
        print(f"Result directory '{directory}' already exists.")

def calc_offline_throughput_const():
    global example_json
    settings = ["512-512", "512-1024", "1024-512", "512-2"]
    for setting in settings:
        file = f"{setting}-0.stat.csv"
        with open(offline_const + file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read the header row
            data_row = next(reader)  # Read the data row
    
            # Assuming 'token_per_second_per_gpu' is the header name exactly as in the CSV file
            token_per_second_per_gpu_index = headers.index('token_per_second_per_gpu')
            token_per_second_per_gpu_value = float(data_row[token_per_second_per_gpu_index])  # Convert to float if needed
            example_json["offline_throughput_const"][setting] = int(token_per_second_per_gpu_value)

def calc_offline_throughput_real_trace():
    global example_json
    datasets = ["splitwise", "lmsys", "sharegpt"]
    json_var = ["Splitwise", "LMSYS-Chat", "ShareGPT"]   
    for idx in range(3):
        file = f"{datasets[idx]}/0.stat.csv"
        with open(offline_real + file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read the header row
            data_row = next(reader)  # Read the data row
    
            # Assuming 'token_per_second_per_gpu' is the header name exactly as in the CSV file
            token_per_second_per_gpu_index = headers.index('token_per_second_per_gpu')
            token_per_second_per_gpu_value = float(data_row[token_per_second_per_gpu_index])  # Convert to float if needed
            example_json["offline_throughput_real_trace"][json_var[idx]] = int(token_per_second_per_gpu_value)

def get_rate(file_path):
    base_name = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(base_name)[0]
    filename_without_extension = os.path.splitext(filename_without_extension)[0]
    return int(filename_without_extension)

def calc_online_throughput():
    global example_json
    datasets = ["splitwise", "lmsys", "sharegpt"]
    json_var = ["Splitwise", "LMSYS-Chat", "ShareGPT"]

    for idx in range(3):
        result_list = []
        for i in range(len(online_paths)):
            dataset_dir = online_paths[i] + datasets[idx] + "/"
            sorted_files = sorted(filter(lambda x: x.endswith(".stat.csv"),os.listdir(dataset_dir)), key=lambda x: get_rate(x))
            for filename in sorted_files:
                if filename.endswith(".stat.csv") and get_rate(filename) > 0:
                    with open(dataset_dir + filename, 'r') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        data_row = next(reader)
                        token_per_second_per_gpu_index = headers.index('average_normalize_latency')
                        lat = int(float(data_row[token_per_second_per_gpu_index])*1000)/1000
                        rate = get_rate(filename)
                        result_list.append((rate, lat))
        
        # sort the result list by request rate from small to large, if request rate is the same, sort by latency from small to large
        result_list.sort(key=lambda x: (x[0], x[1]))
        print(result_list)
        # for same request rate, keep the smallest latency
        filtered_list = result_list[:1]
        for i in range(1, len(result_list)):
            if result_list[i][0] == result_list[i-1][0]:
                continue
            else:
                filtered_list.append(result_list[i])
        result_list = []
        # if smaller request rate has larger latency, remove it
        for i in range(0, len(filtered_list)-1):
            if filtered_list[i][1] > filtered_list[i+1][1]:
                continue
            else:
                result_list.append(filtered_list[i])
        result_list.append(filtered_list[-1])
        print (result_list)
        for i in range(0, len(result_list)):
            example_json["online_throughput"][json_var[idx]]["request_rate"].append(result_list[i][0])
            example_json["online_throughput"][json_var[idx]]["normalized_latency"].append(result_list[i][1])
                    
def calc_ablation():
    global example_json
    ablation_methods = ["nanobatch-only", "non-overlap", "pllm-offload"]
    json_var = ["Nanobatch-only", "Non-overlap", "Nanoflow-offload"]
    settings = ["512-512", "512-1024", "1024-512", "512-2"]
    for method_idx in range(3):
        method = ablation_methods[method_idx]
        for setting in settings:
            file = f"{method}/{setting}-0.stat.csv"
            with open(ablation + file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Read the header row
                data_row = next(reader)  # Read the data row
        
                # Assuming 'token_per_second_per_gpu' is the header name exactly as in the CSV file
                token_per_second_per_gpu_index = headers.index('token_per_second_per_gpu')
                token_per_second_per_gpu_value = float(data_row[token_per_second_per_gpu_index])  # Convert to float if needed
                example_json["ablation"][json_var[method_idx]][setting] = int(token_per_second_per_gpu_value)
    for setting in settings:
        example_json["ablation"]["Nanoflow"][setting] = example_json["offline_throughput_const"][setting]

def save_to_json_with_timestamp():
    global example_json
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # Example format: 2024-06-29_15-30-00
    json_file = save_dir + f'output_json_file_{timestamp}.json'  # Example: output_json_file_2024-06-29_15-30-00.json
    with open(json_file, 'w') as jfile:
        json.dump(example_json, jfile, indent=4)
    print("Results saved to", json_file)
    return json_file

def save_to_pllm():
    global example_json
    json_file = "./pllm.json"
    with open(json_file, 'w') as jfile:
        json.dump(example_json, jfile, indent=4)
    print("Results saved to", json_file)
    return json_file

if __name__ == "__main__":
    create_directory()
    print("Begin generating results...")
    calc_offline_throughput_const()

    calc_offline_throughput_real_trace()

    calc_online_throughput()

    calc_ablation()

    # save_to_json_with_timestamp()
    save_to_pllm()
    print("Finished generating results.")

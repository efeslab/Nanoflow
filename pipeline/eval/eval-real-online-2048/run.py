import os
import sys
import subprocess
import argparse

from itertools import chain

def flatten_list(elements):
    """
    Flattens a list containing individual elements and ranges into a single list.

    Args:
        elements (list): The original list containing elements and ranges.

    Returns:
        list: A flattened list.
    """
    return list(chain.from_iterable(
        [element] if not isinstance(element, range) else element
        for element in elements
    ))


def main():
    # Ensure the script is called with the correct number of arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--trace_base", type=str, default="../../../datasets/traces", help="The base directory containing the traces.")
    arg_parser.add_argument("--executor_base", type=str, default="../../utils", help="The base directory containing the executor.")  
    arg_parse = arg_parser.parse_args()
    
    current_dir = os.getcwd()
    trace_base = os.path.abspath(arg_parse.trace_base)
    executor_base =  arg_parse.executor_base
    result_base = os.path.join(current_dir, "results")

    # Create the result directory if it doesn't exist
    os.makedirs(result_base, exist_ok=True)

    datasets = [
        "splitwise",
        "lmsys",
        "sharegpt"
    ]
    
    ranges = [
        [range(0, 21, 3)],
        [0, 10, range(20, 50, 4)],
        [0, 5, range(10, 25, 3)]
    ]
    
    ranges = [ flatten_list(r) for r in ranges ]
    print (ranges)

    # Loop through the datasets
    for dataset in datasets:
        dataset_trace = os.path.join(trace_base, dataset)
        dataset_result = os.path.join(result_base, dataset)
        dataset_range = ranges[datasets.index(dataset)]
        os.makedirs(dataset_result, exist_ok=True)
        print(f"Running offline throughput experiment for dataset: {dataset}")
        print(f"Range: {dataset_range}")
        print(f"Trace path: {dataset_trace}")
        print(f"Result path: {dataset_result}")
        

        # Loop through each trace file in the dataset directory
        for trace in os.listdir(dataset_trace):
            trace_path = os.path.join(dataset_trace, trace)
            base_trace_name = os.path.splitext(trace)[0]  # Get the base name without extension
            parts = base_trace_name.split('-')  # Split by '-'
            rate = int(parts[2])
            if rate not in dataset_range:
                continue

            print(f"Running offline throughput experiment trace: {trace_path}")

            # Check if the output file already exists
            log_file = os.path.join(dataset_result, f"{rate}.log")
            result_file = os.path.join(dataset_result, f"{rate}.stat.csv")
            if os.path.isfile(result_file):
                print(f"Offline throughput experiment trace {dataset} rate: {rate} already exists. Skipping...")
                continue

            output_prefix = os.path.join(dataset_result, f"{rate}")
            # Construct the command
            command = [
                "python", "serve_8B.py",
                f"--trace_path={trace_path}",
                f"--output_prefix={output_prefix}",
                f"--config_path=../config_all/llama2-70B/2048.json",
                f"--skip_cycles=2000",
                f"--est_cycle_time=0.1",
                f"--empty_weight=True",
                f"--run_cycles=1500"
            ]

            # Execute the command and log the output
            with open(log_file, "w") as log:
                subprocess.run(command, cwd=executor_base, stdout=log, stderr=log)

if __name__ == "__main__":
    main()

import os
import sys
import subprocess
import argparse

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

    fix_trace = os.path.join(trace_base, "fixed")
    
    pipeline_types = ["non-overlap", "nanobatch-only", "pllm-offload"]

    # Create the result directory if it doesn't exist
    os.makedirs(result_base, exist_ok=True)
    
    for pipeline_type in pipeline_types:
        type_result_base = os.path.join(result_base, pipeline_type)
        os.makedirs(type_result_base, exist_ok=True)
        # Loop through each trace file in the fixed trace directory
        for trace in os.listdir(fix_trace):
            trace_path = os.path.join(fix_trace, trace)
            print(f"Running offline throughput experiment trace: {trace_path}")
            
            base_trace_name = os.path.splitext(trace)[0]  # Get the base name without extension
            parts = base_trace_name.split('-')  # Split by '-'
            input_len = parts[0]
            output_len = parts[1]
            rate = parts[2]

            log_file = os.path.join(type_result_base, f"{input_len}-{output_len}-{rate}.log")
            result_file = os.path.join(type_result_base, f"{input_len}-{output_len}-{rate}.stat.csv")
            # Check if the output file already exists
            if os.path.isfile(result_file):
                print(f"Offline throughput experiment input_len: {input_len}, output_len: {output_len}, rate: {rate} already exists. Skipping...")
                continue
            
            skip_cycles = 2000 if int(output_len) < 10 else 10000
            # Construct the command
            command = [
                "python", "serve_8B.py",
                f"--config_path=../config_all/llama2-70B/{pipeline_type}.json",
                f"--trace_path={trace_path}",
                f"--output_prefix={os.path.join(type_result_base, f"{input_len}-{output_len}-{rate}")}",
                f"--skip_cycles={skip_cycles}",
                f"--empty_weight=True",
                f"--run_cycles=1500"
            ]
            # print (command)

            # Execute the command and log the output
            with open(log_file, "w") as log:
                subprocess.run(command, cwd=executor_base, stdout=log, stderr=log)

if __name__ == "__main__":
    main()

#!/bin/bash

workdir=/code/pllm/compute-bound/utils
tracedir=/code/pllm/datasets/traces
scriptdir=/code/pllm/compute-bound/eval

splitwise_output_dir=${workdir}/results/splitwise
lmsys_output_dir=${workdir}/results/lmsys
sharegpt_output_dir=${workdir}/results/sharegpt

trace_gen_path=/code/pllm/datasets/generate_trace.sh

# Step 1: Offline Throughput - Fixed Input and Output Length
echo "Starting Offline Throughput Experiments with Fixed Input and Output Lengths"
cd $workdir

# Step 1.A: Generate Offline Traces with Fixed Input and Output Lengths and from 3 datasets (splitwise, lmsys, sharegpt)
mkdir -p offline_traces && cd offline_traces

input_output_pairs=(
  "2048 1024"
  "1024 512"
  "512 1024"
)



cd $workdir
mkdir -p results


# Step 1.B: Run Offline Throughput Experiments with Fixed Input and Output Lengths

for pair in "${input_output_pairs[@]}"; do
  # check whether the output file exists
  if [ -f $workdir/results/offline-${input_len}-${output_len}.csv ]; then
    echo "Offline throughput experiment input_len: $input_len, output_len: $output_len already exists. Skipping..."
    continue
  fi
  read input_len output_len <<< "$pair"
  echo "Running offline throughput experiment input_len: $input_len, output_len: $output_len"
  python3 $workdir/scheduler.py --trace_path=$workdir/offline_traces/${input_len}-${output_len}-0.csv --output_path=$workdir/results/offline-${input_len}-${output_len}.csv > $workdir/results/offline-${input_len}-${output_len}.log
done



# Step 1.C: Run Offline Throughput Experiments from 3 datasets (splitwise, lmsys, sharegpt)

datasets=(
  "splitwise"
  "lmsys"
  "sharegpt"
)

for dataset in "${datasets[@]}"; do
  # check whether the output file exists
  if [ -f $workdir/results/offline-${dataset}.csv ]; then
    echo "Offline throughput experiment dataset: $dataset already exists. Skipping..."
    continue
  fi
  echo "Running offline throughput experiment dataset: $dataset"
  python3 $workdir/scheduler.py --trace_path=${tracedir}/${dataset}/${dataset}-rate-0-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/offline-${dataset}.csv > $workdir/results/offline-${dataset}.log
done

# check whether the output file exists
echo "Running offline throughput - splitwise experiment"
python3 $workdir/scheduler.py --trace_path=$splitwise_output_dir/splitwise-rate-0-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/offline-splitwise.csv

echo "Running offline throughput - lmsys experiment"
python3 $workdir/scheduler.py --trace_path=$lmsys_output_dir/lmsys-rate-0-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/offline-lmsys.csv

echo "Running offline throughput - sharegpt experiment"
python3 $workdir/scheduler.py --trace_path=$sharegpt_output_dir/sharegpt-rate-0-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/offline-sharegpt.csv


# Step 2: Online Throughput - Input and Output Length from 3 datasets (splitwise, lmsys, sharegpt)
echo "Starting Online Throughput Experiments with Input and Output Lengths from 3 datasets"

# define a rate array from 1 to 30
rates=$(seq 1 30)

# for rate from 1 to 30
for rate in $rates; do
  echo "Generating traces for rate: $rate"
  bash $trace_gen_path $rate
  # check whether the output file exists
  if [ -f $workdir/results/online-splitwise-${rate}.csv ]; then
    echo "Online throughput experiment rate: $rate already exists. Skipping..."
    continue
  fi
  echo "Running online throughput - splitwise experiment for rate: $rate"
  python3 $workdir/scheduler.py --trace_path=$splitwise_output_dir/splitwise-rate-${rate}-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/online-splitwise-${rate}.csv
  echo "Running online throughput - lmsys experiment for rate: $rate"
  python3 $workdir/scheduler.py --trace_path=$lmsys_output_dir/lmsys-rate-${rate}-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/online-lmsys-${rate}.csv
  echo "Running online throughput - sharegpt experiment for rate: $rate"
  python3 $workdir/scheduler.py --trace_path=$sharegpt_output_dir/sharegpt-rate-${rate}-5min-reqs-10000-exp-delay.csv --output_path=$workdir/results/online-sharegpt-${rate}.csv
done

# Step 3: Merge and Plot
echo "Merging and Plotting Results"
python3 $scriptdir/merge_results.py $workdir/results
python3 $scriptdir/baseline_data.py $workdir/results
cd $workdir/results
mkdir -p figures
python3 $scriptdir/plot_all.py --data_dir=$workdir/results --fig_dir=$workdir/results/figures
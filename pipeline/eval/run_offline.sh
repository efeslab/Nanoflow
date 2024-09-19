#!/bin/bash

source ~/miniconda3/bin/activate vllm

cd ~/pllm/compute-bound/utils

# python3 ./scheduler.py /home/ubuntu/docker_shared/pllm-benchmarks/datasets/offline_throughput/splitwise/splitwise-length-5min-reqs-10000-exp-delay.csv
# python3 ./scheduler.py /home/ubuntu/docker_shared/pllm-benchmarks/datasets/offline_throughput/lmsys/lmsys-length-5min-reqs-10000-exp-delay.csv
# python3 ./scheduler.py /home/ubuntu/docker_shared/pllm-benchmarks/datasets/offline_throughput/sharegpt/sharegpt-length-5min-reqs-10000-exp-delay.csv

#!/bin/bash


# for i in 10 9_5 9 8_5 8 7_5 7 6_5 6 5_5 5 4_5 4 3_5 3 2_5 2 1_5 1
# for i in 20 19 18 17 16 15 14 13 12 11
min=10
# for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# for i in 3_5 4
# for i in 1_5
# for i in 15 5 2
# do  
#     # num_requests=$((i * min * 60))
#     # replace _ with . to make it a float
#     rate=$(echo $i | sed 's/_/./')
#     num_requests=$(echo "$rate * $min * 60" | bc)
#     # make num_requests an integer
#     num_requests=${num_requests%.*}
#     dataset_file=/home/ubuntu/docker_shared/pllm-benchmarks/datasets/online_throughput/length_only/splitwise/splitwise-10min-rate-${i}-reqs-${num_requests}-exp-delay.csv
#     # vllm
#     # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend vllm         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result
#     # fastgen
#     # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend deepspeed-mii         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result  --host localhost --port 28080 --endpoint /mii/llama-deployment

#     # pllm
#     timestamp=$(date '+%Y-%m-%d-%H-%M-%S')
#     python3 ./scheduler.py ${dataset_file}  > /home/ubuntu/pllm/compute-bound/utils/logs/${timestamp}_splitwise-rate-${i}-reqs-${num_requests}.log

# done

for i in 5
# for i in 15 10
do  
    # num_requests=$((i * min * 60))
    # replace _ with . to make it a float
    rate=$(echo $i | sed 's/_/./')
    num_requests=$(echo "$rate * $min * 60" | bc)
    # make num_requests an integer
    num_requests=${num_requests%.*}
    dataset_file=/home/ubuntu/docker_shared/pllm-benchmarks/datasets/online_throughput/length_only/sharegpt/sharegpt-10min-rate-${i}-reqs-${num_requests}-exp-delay.csv
    # vllm
    # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend vllm         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result
    # fastgen
    # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend deepspeed-mii         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result  --host localhost --port 28080 --endpoint /mii/llama-deployment

    # tensorrtllm
    timestamp=$(date '+%Y-%m-%d-%H-%M-%S')
    python3 ./scheduler.py ${dataset_file}  > /home/ubuntu/pllm/compute-bound/utils/logs/${timestamp}_sharegpt-rate-${i}-reqs-${num_requests}.log
done

# for i in 18
# for i in 25
# do  
#     # num_requests=$((i * min * 60))
#     # replace _ with . to make it a float
#     rate=$(echo $i | sed 's/_/./')
#     num_requests=$(echo "$rate * $min * 60" | bc)
#     # make num_requests an integer
#     num_requests=${num_requests%.*}
#     dataset_file=/home/ubuntu/docker_shared/pllm-benchmarks/datasets/online_throughput/length_only/lmsys/lmsys-10min-rate-${i}-reqs-${num_requests}-exp-delay.csv
#     # vllm
#     # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend vllm         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result
#     # fastgen
#     # python3 ~/vllm/benchmarks/benchmark_serving.py         --backend deepspeed-mii         --model meta-llama/Llama-2-70b-hf --trace-path $dataset_file --save-result  --host localhost --port 28080 --endpoint /mii/llama-deployment

#     # tensorrtllm
#     timestamp=$(date '+%Y-%m-%d-%H-%M-%S')
#     python3 ./scheduler.py ${dataset_file}  > /home/ubuntu/pllm/compute-bound/utils/logs/${timestamp}_lmsys-rate-${i}-reqs-${num_requests}.log
# done




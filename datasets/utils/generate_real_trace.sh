#!/bin/bash

tracedir=../traces
splitwise_output_dir=${tracedir}/splitwise
lmsys_output_dir=${tracedir}/lmsys
sharegpt_output_dir=${tracedir}/sharegpt
mkdir -p ${splitwise_output_dir}
mkdir -p ${lmsys_output_dir}
mkdir -p ${sharegpt_output_dir}

minute=5
num_requests=15000

# get the argument as rate
rate=$1

# splitwise
filename="${splitwise_output_dir}/splitwise-rate-${rate}-${minute}min-reqs-${num_requests}-exp-delay.csv"
safe_filename=$(echo "$filename" | sed 's/\([0-9]\)\.\([0-9]\)/\1_\2/')
python3 ./prepare_dataset.py \
  --output ${safe_filename} \
  --request-rate ${rate} \
  --time-delay-dist exponential_dist \
  --tokenizer  lmsys/longchat-13b-16k\
   splitwise \
   --num-requests 17563 \
   --trace-path ${tracedir}/splitwise.csv \
   --mode length
# lmsys
filename="${lmsys_output_dir}/lmsys-rate-${rate}-${minute}min-reqs-${num_requests}-exp-delay.csv"
safe_filename=$(echo "$filename" | sed 's/\([0-9]\)\.\([0-9]\)/\1_\2/')
python3 ./prepare_dataset.py \
  --output ${safe_filename} \
  --request-rate ${rate} \
  --time-delay-dist exponential_dist \
  --tokenizer  lmsys/longchat-13b-16k\
   splitwise \
   --num-requests 50000 \
   --trace-path ${tracedir}/lmsys.csv \
   --mode length
# sharegpt
filename="${sharegpt_output_dir}/sharegpt-rate-${rate}-${minute}min-reqs-${num_requests}-exp-delay.csv"
safe_filename=$(echo "$filename" | sed 's/\([0-9]\)\.\([0-9]\)/\1_\2/')
python3 ./prepare_dataset.py \
  --output ${safe_filename} \
  --request-rate ${rate} \
  --time-delay-dist exponential_dist \
  --tokenizer  lmsys/longchat-13b-16k\
   splitwise \
   --num-requests 50000 \
   --trace-path ${tracedir}/sharegpt.csv \
   --mode length
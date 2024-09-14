#!/bin/bash
tracedir=../traces
mkdir -p ${tracedir}
fixTraceDir=${tracedir}/fixed
mkdir -p ${fixTraceDir}


input_output_pairs=(
  "512 512"
  "1024 512"
  "512 1024"
  "128 1024"
  "512 2"
)

for pair in "${input_output_pairs[@]}"; do

  read input_len output_len <<< "$pair"
  echo "Generating trace for input_len: $input_len, output_len: $output_len"
  python3 gen_const_len_req.py $input_len $output_len 0 ${fixTraceDir}

done
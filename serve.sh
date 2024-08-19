cd pipeline/utils
read -e -p "Prompt [default: The University of Washington is]: " -i "The University of Washington is located" prompt
read -e -p "Decode length [default: 100]: " -i "100" decode_length
read -e -p "Output file [default: trace.csv]: " -i "trace.csv" output_file

python gen_req.py "${prompt}" ${decode_length} 0 ${output_file}

python serve.py --trace_path ${output_file}
cat ${output_file}.out
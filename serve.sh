current_dir=$(pwd)
parentdir="$(dirname "$current_dir")"
mkdir -p $parentdir/hf

export HF_HOME=$parentdir/hf
HF_HOME=$parentdir/hf
#check if token is cached
if [ ! -f $HF_HOME/token ]; then
    echo "Please login to Hugging Face to cache your token."
    huggingface-cli login
fi


cd pipeline/utils
read -e -p "Prompt [default: The University of Washington is located]: " -i "The University of Washington is located" prompt
read -e -p "Decode length [default: 100]: " -i "100" decode_length
read -e -p "Output file [default: trace.csv]: " -i "trace.csv" output_file

# Prompt for model selection and map the selection to a specific model path
echo "Select model:"
echo "1) llama2-70B"
echo "2) llama3-70B"
echo "3) llama3.1-70B"
echo "4) llama3-8B"
echo "5) llama3.1-8B"
echo "6) Qwen2-72B"

read -p "Enter the number corresponding to your model choice: " model_choice

case $model_choice in
    1)
        config_path="../config_all/llama2-70B/2048.json"
        ;;
    2)
        config_path="../config_all/llama3-70B/2048.json"
        ;;
    3)
        config_path="../config_all/llama3.1-70B/2048.json"
        ;;
    4)
        config_path="../config_all/llama3-8B/correct_40G/1024.json"
        ;;
    5)
        config_path="../config_all/llama3.1-8B/1024.json"
        ;;
    6)
        config_path="../config_all/qwen2-72B/2048.json"
        ;;
    *)
        echo "Invalid choice. Defaulting to llama3-8B."
        config_path="../config_all/llama3-8B/1024.json"
        ;;
esac


python gen_req.py "${prompt}" ${decode_length} 0 ${output_file}

python serve_8B.py -t ${output_file} -c ${config_path} -r 200
output_file_base="${output_file%.csv}"
cat ${output_file_base}.req_words
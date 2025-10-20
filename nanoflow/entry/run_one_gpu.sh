# Pick one:
# MODEL="8B"
# MODEL="Qwen1.5-MoE-A2.7B"
MODEL="Qwen2-57B-A14B-Instruct"

# Pick one:
TEST="correctness"
# TEST="one_cycle"

TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=4 \
python test_one_gpu.py \
  --model "$MODEL" \
  --test "$TEST"

# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=6 nsys profile -o nsys/qwen2-moe-naive-impl python run_llama3.py --model Qwen1.5-MoE-A2.7B --test performance
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_naive_impl python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_auto_search python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_auto_search_cuda_graph python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile --cuda-graph-trace node -o llama3-8B_auto_search_cuda_graph_node_mode python run_llama3.py
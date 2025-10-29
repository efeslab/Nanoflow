# Pick one:
MODEL="Llama3-8B"
# MODEL="Llama3-70B"
# MODEL="Qwen1.5-MoE-A2.7B"
# MODEL="Qwen2-57B-A14B-Instruct"

# Pick one:
# TEST="correctness"
TEST="performance"
# TEST="one_cycle"

# Pick one:
# KVCacheType="none"
# KVCacheType="torch"
KVCacheType="flashinfer"

NSYS_PROFILE_NAME="nsys/llama3-8B_performance_naive_%n"
NSYS_PROFILE_NAME="nsys/llama3-8B_performance_double_buffer_%n"

TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=4 \
python test_one_gpu.py \
  --model "$MODEL" \
  --test "$TEST"\
  --kvcache_type "$KVCacheType"

# TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=4 \
# nsys profile -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true -o "$NSYS_PROFILE_NAME" \
# python test_one_gpu.py \
#   --model "$MODEL" \
#   --test "$TEST"\
#   --kvcache_type "$KVCacheType"
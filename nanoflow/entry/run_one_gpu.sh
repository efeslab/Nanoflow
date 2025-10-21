# Pick one:
# MODEL="Llama3-8B"
# MODEL="Llama3-70B"
MODEL="Qwen1.5-MoE-A2.7B"
# MODEL="Qwen2-57B-A14B-Instruct"

# Pick one:
TEST="correctness"
# TEST="one_cycle"

# Pick one:
# KVCacheType="none"
# KVCacheType="torch"
KVCacheType="flashinfer"

NSYS_PROFILE_NAME="./nsys/llama3-8B_flashinfer_impl"

TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=7 \
python test_one_gpu.py \
  --model "$MODEL" \
  --test "$TEST"\
  --kvcache_type "$KVCacheType"

TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=7 \
nsys profile -o "$NSYS_PROFILE_NAME" python test_one_gpu.py \
  --model "$MODEL" \
  --test "$TEST"\
  --kvcache_type "$KVCacheType"
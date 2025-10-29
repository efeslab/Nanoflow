MODEL="Llama3-8B"
TEST="correctness"
KVCacheType="flashinfer"

TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=0 \
python test_one_gpu.py \
  --model "$MODEL" \
  --test "$TEST"\
  --kvcache_type "$KVCacheType"
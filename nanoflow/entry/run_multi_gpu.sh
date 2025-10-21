# MODEL="8B"
MODEL="70B"
# MODEL="Qwen1.5-MoE-A2.7B-EP"
# MODEL="Qwen2-57B-A14B-Instruct-EP"
# MODEL="Qwen2-57B-A14B-Instruct-TP-EP"

# TEST="correctness"
# TEST="performance"
TEST="prefill_only"
# TEST="profile"

EP_SIZE=2
TP_SIZE=2

# KV_CACHE_TYPE="none"
# KV_CACHE_TYPE="torch"
KV_CACHE_TYPE="flashinfer"

NETWORK_TYPE="allreduce"
# NETWORK_TYPE="allgather"

NSYS_PROFILE_NAME="./nsys/llama3-70B_flashinfer_allreduce_tp2_prefill_only_naive_impl_%n"

# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=6,7 \
# MASTER_ADDR=localhost MASTER_PORT=12555 \
# python test_multi_gpu.py \
# --model "$MODEL" \
# --test "$TEST" \
# --tensor_parallel_size "$TP_SIZE" \
# --expert_parallel_size "$EP_SIZE" \
# --kvcache_type "$KV_CACHE_TYPE" \
# --network_type "$NETWORK_TYPE" \
# # --cuda_graph \
# # --auto_search

TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=6,7 \
MASTER_ADDR=localhost MASTER_PORT=12555 \
nsys profile -o "$NSYS_PROFILE_NAME" python test_multi_gpu.py \
--model "$MODEL" \
--test "$TEST" \
--tensor_parallel_size "$TP_SIZE" \
--expert_parallel_size "$EP_SIZE" \
--kvcache_type "$KV_CACHE_TYPE" \
--network_type "$NETWORK_TYPE" \

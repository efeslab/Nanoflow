# MODEL="8B"
MODEL="70B"
# MODEL="Qwen1.5-MoE-A2.7B-EP"
# MODEL="Qwen2-57B-A14B-Instruct-EP"
# MODEL="Qwen2-57B-A14B-Instruct-TP-EP"

# TEST="correctness"
# TEST="performance"
TEST="prefill_only"
# TEST="profile"

EP_SIZE=4
TP_SIZE=4

# KV_CACHE_TYPE="none"
# KV_CACHE_TYPE="torch"
KV_CACHE_TYPE="flashinfer"

NETWORK_TYPE="allreduce"
# NETWORK_TYPE="allgather"

# NSYS_PROFILE_NAME="./nsys/llama3-70B_tp4_prefill_only_naive_%n"
NSYS_PROFILE_NAME="./nsys/llama3-70B_tp4_prefill_only_overlap_%n"

# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 \
# MASTER_ADDR=localhost MASTER_PORT=12555 \
# python test_multi_gpu.py \
# --model "$MODEL" \
# --test "$TEST" \
# --tensor_parallel_size "$TP_SIZE" \
# --expert_parallel_size "$EP_SIZE" \
# --kvcache_type "$KV_CACHE_TYPE" \
# --network_type "$NETWORK_TYPE" \
# # --use_auto_search \
# # --use_nanosplit \

TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_ADDR=localhost MASTER_PORT=12555 \
nsys profile -o "$NSYS_PROFILE_NAME" python test_multi_gpu.py \
--model "$MODEL" \
--test "$TEST" \
--tensor_parallel_size "$TP_SIZE" \
--expert_parallel_size "$EP_SIZE" \
--kvcache_type "$KV_CACHE_TYPE" \
--network_type "$NETWORK_TYPE" \
--use_auto_search \
--use_nanosplit \

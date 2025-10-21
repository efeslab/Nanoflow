########################## TP2 ##########################
# MODEL="8B"
MODEL="70B"
# MODEL="Qwen1.5-MoE-A2.7B-EP"
# MODEL="Qwen2-57B-A14B-Instruct-EP"
# MODEL="Qwen2-57B-A14B-Instruct-TP-EP"

TEST="correctness"
# TEST="performance"
# TEST="profile"

EP_SIZE=2
TP_SIZE=2

# KVCacheType="none"
# KVCacheType="torch"
KV_CACHE_TYPE="flashinfer"

# NetworkType="allreduce"
NETWORK_TYPE="allgather"

# take effect when TEST is "performance"
USE_CUDA_GRAPH=False
USE_AUTO_SEARCH=False

NSYS_PROFILE_NAME="./nsys/qwen2-moe-57B-a14b-instruct_ep2_naive_impl"

TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=4,5 \
MASTER_ADDR=localhost MASTER_PORT=12555 \
python test_multi_gpu.py \
--model "$MODEL" \
--test "$TEST" \
--tensor_parallel_size "$TP_SIZE" \
--expert_parallel_size "$EP_SIZE" \
--kvcache_type "$KV_CACHE_TYPE" \
--network_type "$NETWORK_TYPE" \
--cuda_graph "$USE_CUDA_GRAPH" \
--auto_search "$USE_AUTO_SEARCH"

# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=4,5 \
# MASTER_ADDR=localhost MASTER_PORT=12555 \
# nsys profile -o "$nsys_profile_name" python test_multi_gpu.py \
# --model "$MODEL" \
# --test "$TEST" \
# --expert_parallel_size "$EP_size"
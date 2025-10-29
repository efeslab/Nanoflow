# MODEL="Llama3-8B"
# MODEL="Llama3-70B"
# MODEL="Qwen1.5-MoE-A2.7B-EP"
# MODEL="Qwen2-57B-A14B-Instruct-EP"
MODEL="Qwen2-57B-A14B-Instruct-TP-EP"

# TEST="correctness"
# TEST="prefill_only"
# TEST="decode_only"
TEST="performance"
# TEST="profile"

DP_SIZE=1
TP_SIZE=4
EP_SIZE=4

# KV_CACHE_TYPE="none"
# KV_CACHE_TYPE="torch"
KV_CACHE_TYPE="flashinfer"

NETWORK_TYPE="allreduce"
# NETWORK_TYPE="allgather"

# NSYS_PROFILE_NAME="nsys/llama3-70B_decode_only_naive_%n"
# NSYS_PROFILE_NAME="nsys/llama3-70B_decode_only_overlap_%n"

NSYS_PROFILE_NAME="nsys/qwen2-57B-a14B-instruct-tp-ep_prefill_only_double_buffer_%n"
# NSYS_PROFILE_NAME="nsys/qwen2-57B-a14B-instruct-tp-ep_prefill_only_overlap_%n"

TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_ADDR=localhost MASTER_PORT=12555 \
python test_multi_gpu.py \
--test "$TEST" \
--model "$MODEL" \
--data_parallel_size "$DP_SIZE" \
--tensor_parallel_size "$TP_SIZE" \
--expert_parallel_size "$EP_SIZE" \
--kvcache_type "$KV_CACHE_TYPE" \
--network_type "$NETWORK_TYPE" \
# --use_cuda_graph \
# --use_auto_search \
# --use_nanosplit \

# nsys profile -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --cuda-graph node -o "$NSYS_PROFILE_NAME" \

# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=4,5,6,7 \
# MASTER_ADDR=localhost MASTER_PORT=12555 \
# nsys profile -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --cuda-graph node -x true -o "$NSYS_PROFILE_NAME" \
# python test_multi_gpu.py \
# --test "$TEST" \
# --model "$MODEL" \
# --tensor_parallel_size "$TP_SIZE" \
# --expert_parallel_size "$EP_SIZE" \
# --kvcache_type "$KV_CACHE_TYPE" \
# --network_type "$NETWORK_TYPE" \
# --use_cuda_graph \
# --use_nanosplit \
# --use_auto_search \

# CUDA_LAUNCH_BLOCKING=1
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o llama3-70B_naive_impl python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o llama3-70B_2way_auto_search python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o llama3-70B_2way_auto_search_cuda_graph python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile --cuda-graph-trace node -o llama3-70B_2way_auto_search_cuda_graph_node_mode_%n python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=1,2,3,4 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o test_multi_gpu_use_auto_search_cuda_graph_%n python test_multi_gpu3.py
TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o test_nccl_wrapper_%n python test_multi_gpu3.py
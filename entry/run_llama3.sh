TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_naive_impl python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_auto_search python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile -o llama3-8B_auto_search_cuda_graph python run_llama3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0 nsys profile --cuda-graph-trace node -o llama3-8B_auto_search_cuda_graph_node_mode python run_llama3.py
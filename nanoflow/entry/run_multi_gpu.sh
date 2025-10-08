########################## TP2 ##########################
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-70B_naive_impl python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-70B_2way_auto_search python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-70B_2way_auto_search_cuda_graph python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile --cuda-graph-trace node -o ./nsys/llama3-70B_2way_auto_search_cuda_graph_node_mode_%n python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=1,2,3,4 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/test_multi_gpu_use_auto_search_cuda_graph_%n python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/test_nccl_wrapper_%n python test_multi_gpu3.py --tensor_parallel_size 2
TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=1,2 MASTER_ADDR=localhost MASTER_PORT=12555 python test_multi_gpu3.py --model 8B --test correctness --tensor_parallel_size 2
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=1,2 MASTER_ADDR=localhost MASTER_PORT=12555 python test_multi_gpu3.py --model Qwen1.5-MoE-A2.7B-EP --test correctness --expert_parallel_size 2

########################### TP4 ##########################
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-8B_naive_impl_%n python test_multi_gpu3.py --model 8B --test performance --tensor_parallel_size 4
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-70B_2way_auto_search_%n python test_multi_gpu3.py --model 70B --test performance --tensor_parallel_size 4
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/llama3-70B_2way_auto_search_cuda_graph_%n python test_multi_gpu3.py --model 70B --test performance --tensor_parallel_size 4
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile --cuda-graph-trace node -o ./nsys/llama3-70B_2way_auto_search_cuda_graph_node_mode_%n python test_multi_gpu3.py --model 70B --test performance --tensor_parallel_size 4
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=1,2,3,4 MASTER_ADDR=localhost MASTER_PORT=12548 nsys profile -o ./nsys/test_multi_gpu_use_auto_search_cuda_graph_%n python test_multi_gpu3.py
# TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=localhost MASTER_PORT=12548 python test_multi_gpu3.py --model 70B --test correctness --tensor_parallel_size 4

########################### TP8 ##########################
# TORCH_CUDA_ARCH_LIST="9.0" MASTER_ADDR=localhost MASTER_PORT=12550 python test_multi_gpu3.py --model 70B --test performance --tensor_parallel_size 8
# TORCH_CUDA_ARCH_LIST="9.0" MASTER_ADDR=localhost MASTER_PORT=12550 nsys profile -o ./nsys/llama70B_tp8_naive python test_multi_gpu3.py --model 70B --test performance --tensor_parallel_size 8

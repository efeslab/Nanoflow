# CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,5,6,7 MASTER_ADDR=localhost  MASTER_PORT=12548 nsys profile -o test_profile_one_cycle_%n python test_multi_gpu3.py
# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 CUDA_VISIBLE_DEVICES=0,5,6,7 MASTER_ADDR=localhost  MASTER_PORT=12548 python test_multi_gpu3.py
# CUDA_VISIBLE_DEVICES=0,5,6,7 MASTER_ADDR=localhost  MASTER_PORT=12548 python test_multi_gpu3.py
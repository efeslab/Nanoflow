import os

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    gemm_sm_nccl_channel = [
        (88, 1), (88, 2), (96, 2), (104, 4), (112, 1), (112, 2), (112, 4), (112, 8), (112, 16), (112, 24)
    ]
    # for gemm_sm_count in range(80, 132, 8):
        # for nccl_channel_count in [1, 2, 4, 8, 16, 24]:
    for gemm_sm_count, nccl_channel_count in gemm_sm_nccl_channel:
            command = f"nsys profile -f true -o overlap_comp_{gemm_sm_count}_comm_{nccl_channel_count}_new.nsys-rep python test_compute_comm_overlap.py --gemm_sm_count {gemm_sm_count} --nccl_channel_count {nccl_channel_count}"
            print(command)
            os.system(command)

if __name__ == "__main__":
    main()

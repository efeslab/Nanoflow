import argparse
import sys
import os
import torch

sys.path.append("../")
from utils.cu_mask import create_streams_with_cumask

# from flashinfer import single_prefill_with_kv_cache
# from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
# import nvtx
# import nvmath

sys.path.append("../pybind_amd/build")
# from bind_all_reduce import NCCLWrapper # type: ignore

USE_NVTX = False
prof_marker = nvtx.annotate if USE_NVTX else torch.profiler.record_function

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


@torch.inference_mode()
def worker(
    rank: int, world_size: int, gemm_sm_count: int, nccl_channel_count: int
) -> None:
    torch.cuda.set_device(rank)
    # nccl_wrapper = NCCLWrapper(rank, world_size, unique_id)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    print(f"Rank {rank} done init")
    dim = 16384
    qo_len = 8192
    kv_len = 8192
    num_heads = 32
    head_dim = 128
    q = torch.randn(qo_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    x = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
    y = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
    z = torch.randn(dim, dim, device="cuda", dtype=torch.float16)

    def compute_kernel():
        # single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa3")
        torch.matmul(x, y)

    def comm_kernel():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        # nccl_wrapper.all_reduce(x, "sum")

    def barrier():
        torch.distributed.barrier()
        # nccl_wrapper.barrier()

    with prof_marker("sequential"):
        for _ in range(100):
            comm_kernel()
            compute_kernel()
        barrier()
        torch.cuda.synchronize()
        print(f"Rank {rank} done sequential")

    with prof_marker("overlap"):
        comm_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        for _ in range(100):
            with torch.cuda.stream(comm_stream):  # type: ignore
                comm_kernel()
            with torch.cuda.stream(compute_stream):  # type: ignore
                compute_kernel()
        barrier()
        torch.cuda.synchronize()
        print(f"Rank {rank} done overlap")

    print(f"Rank {rank} gemm_sm_count: {gemm_sm_count}")
    compute_stream, comm_stream = create_streams_with_cumask(
        [gemm_sm_count, 304 - gemm_sm_count], f"cuda:{rank}"
    )
    with prof_marker("green_ctx_gemm_only_no_target"):
        for _ in range(100):
            with torch.cuda.stream(compute_stream):  # type: ignore
                compute_kernel()
        barrier()
        torch.cuda.synchronize()

    # handle = torch.cuda.current_blas_handle()
    # nvmath.bindings.cublas.set_sm_count_target(handle, sm_count) # type: ignore
    with prof_marker("green_ctx_gemm_only"):
        for _ in range(100):
            with torch.cuda.stream(compute_stream):  # type: ignore
                compute_kernel()
        barrier()
        torch.cuda.synchronize()
    with prof_marker("green_ctx_comm_only"):
        for _ in range(100):
            with torch.cuda.stream(comm_stream):  # type: ignore
                comm_kernel()
        barrier()
        torch.cuda.synchronize()
        print(f"Rank {rank} done green_ctx_comm_only")
    with prof_marker("overlap_green_ctx"):
        for _ in range(100):
            with torch.cuda.stream(comm_stream):  # type: ignore
                comm_kernel()
            with torch.cuda.stream(compute_stream):  # type: ignore
                compute_kernel()
        barrier()
        torch.cuda.synchronize()
        print(f"Rank {rank} done overlap_green_ctx")
    with prof_marker("overlap_comm_gemm_green_ctx"):
        for _ in range(100):
            comm_kernel()
            with torch.cuda.stream(compute_stream):  # type: ignore
                compute_kernel()
        barrier()
        torch.cuda.synchronize()
        print(f"Rank {rank} done overlap_comm_gemm_green_ctx")

    torch.distributed.destroy_process_group()
    torch.cuda.synchronize()
    print("Here at child process")


def worker_wrapper(
    rank: int, world_size: int, gemm_sm_count: int, nccl_channel_count: int
):
    if rank == 0:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            worker(rank, world_size, gemm_sm_count, nccl_channel_count)
        prof.export_chrome_trace(
            f"trace_nccl_{nccl_channel_count}_gemm_{gemm_sm_count}.json"
        )
        print("Profiler done")
    else:
        worker(rank, world_size, gemm_sm_count, nccl_channel_count)


def main(nccl_channel_count: int, gemm_sm_count: int):
    torch.multiprocessing.set_start_method("spawn", force=True)
    world_size = 2
    processes = []
    # unique_id = NCCLWrapper.get_nccl_unique_id()
    for rank in range(world_size):
        p = torch.multiprocessing.Process(
            target=worker_wrapper,
            args=(rank, world_size, gemm_sm_count, nccl_channel_count),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--nccl_channel_count", type=int, default=24)
    # parser.add_argument("--gemm_sm_count", type=int, default=300)
    # args = parser.parse_args()
    for nccl_channel_count in [1, 2, 4, 8, 12, 16, 24]:
        for gemm_sm_count in [256, 264, 272, 280, 288, 296]:
            os.environ["NCCL_MAX_NCHANNELS"] = str(nccl_channel_count)
            os.environ["NCCL_MIN_NCHANNELS"] = str(nccl_channel_count)
            main(nccl_channel_count, gemm_sm_count)

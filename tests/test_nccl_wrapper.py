import torch
import sys
import os
sys.path.append("../pybind/build")
import nvtx

from bind_all_reduce import NCCLWrapper # type: ignore

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

@torch.inference_mode()
def worker(rank: int, world_size: int, unique_id) -> None:
    torch.cuda.set_device(rank)
    nccl_wrapper = NCCLWrapper(rank, world_size, unique_id)
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     init_method="env://",
    #     world_size=world_size,
    #     rank=rank,
    # )

    dim = (rank + 1) * 16384
    x = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
    y = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
    z = torch.empty(dim, dim, device="cuda", dtype=torch.float16)
    compute_stream: torch.cuda.Stream = torch.cuda.Stream() # type: ignore
    comm_stream: torch.cuda.Stream = torch.cuda.Stream() # type: ignore

    with torch.cuda.stream(comm_stream):
        for _ in range(20):
            z = x + y
    torch.cuda.synchronize()

    for i in range(20):
        post_barrier_event = torch.cuda.Event()
        with torch.cuda.stream(compute_stream):
            with nvtx.annotate(f"gemm_1_{i}"):
                z = x @ y
        with torch.cuda.stream(comm_stream):
            with nvtx.annotate(f"barrier_2_{i}"):
                nccl_wrapper.barrier()
            comm_stream.synchronize()
            # torch.distributed.barrier()
            # post_barrier_event.record(comm_stream)
        with torch.cuda.stream(compute_stream):
            # post_barrier_event.wait(compute_stream)
            with nvtx.annotate(f"gemm_3_{i}"):
                z = x @ y
    torch.cuda.synchronize()
    # torch.distributed.destroy_process_group()

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    world_size = 2
    processes = []
    unique_id = NCCLWrapper.get_nccl_unique_id()
    for rank in range(world_size):
        p = torch.multiprocessing.Process(target=worker, args=(rank, world_size, unique_id))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

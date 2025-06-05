import sys
import torch
sys.path.append("../pybind/build")
import bind_init_net


def worker(rank):
    torch.cuda.set_device(rank)
    num_cuda_devices = torch.cuda.device_count()
    print("Process started on GPU: ", rank)
    bind_init_net.init_net(rank, num_cuda_devices)

if __name__ == '__main__':
    import pickle
    import sys, os
    import time
    sys.path.append("../")
    
    os.environ["HF_HOME"] = "/code/hf"
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')  # Ensure safe CUDA behavior
    world_size = torch.cuda.device_count()
    program_start_time = time.time()

    processes = []
    for rank in range(world_size):
        start_time = time.perf_counter()
        # print(f"Starting process {rank} on GPU {rank}")
        args = (rank,)
        p = mp.Process(target=worker, args=args)

        p.start()
        processes.append(p)

    print(f"All processes finished in {time.time() - program_start_time:.2f} seconds")
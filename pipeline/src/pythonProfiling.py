import sys
sys.path.append('../build')
import pllm_python
import torch 
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import subprocess
from pathlib import Path
import time



def perf():
    # test cpu to GPU bandwidth
    # tensorList = []
    # for i in range(100):
    #     tensorList.append(torch.randn((8192, 1024), dtype=torch.half, device='cpu').pin_memory())

    # for i in range(10):
    #     tensorList[i] = tensorList[i].to('cuda:0')
    # torch.cuda.synchronize()
    # start = time.time()
    # for i in range(10, 100):
    #     tensorList[i] = tensorList[i].to('cuda:0')
    # torch.cuda.synchronize()
    # print(f"CPU to GPU bandwidth: {90*8192*1024*2/(time.time()-start)/1024/1024/1024} GB/s")

    # # cocurrently to 8 GPUs
    m = 8192
    n = 1
    tensorList = []
    for i in range(96):
        tensorList.append(torch.randn((m, n), dtype=torch.half, device='cpu').pin_memory())
    
    for i in range(16):
        print(i%8)
        tensorList[i] = tensorList[i].to(f'cuda:{i%8}')
    torch.cuda.synchronize()

    start = time.time()
    for i in range(16, 96):
        tensorList[i] = tensorList[i].to(f'cuda:{i%8}', non_blocking=True)
    torch.cuda.synchronize()
    print(f"CPU to 8 GPUs bandwidth: {80*n*m*2/(time.time()-start)/1024/1024/1024} GB/s, time {(time.time()-start)/10}")

    start = time.time()
    for i in range(16, 96):
        tensorList[i] = tensorList[i].to(f'cpu', non_blocking=True)
    torch.cuda.synchronize()
    print(f"8 GPUs to CPU bandwidth: {80*n*m*2/(time.time()-start)/1024/1024/1024} GB/s, time {(time.time()-start)/10}")

    start = time.time()
    for i in range(16, 96):
        tensorList[i] = tensorList[i].to(f'cuda:{i%8}', non_blocking=True)
    torch.cuda.synchronize()
    print(f"CPU to 8 GPUs bandwidth: {80*n*m*2/(time.time()-start)/1024/1024/1024} GB/s, time {(time.time()-start)/10}")






with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
    with record_function("model_inference"):
        perf()

prof.export_chrome_trace("trace.json")


def compress_file(filename):
    file_path = Path(filename)
    compressed_file_path = file_path.with_suffix(file_path.suffix + '.gz')

    # Check if the compressed file already exists and remove it
    if compressed_file_path.exists():
        try:
            subprocess.run(['rm', str(compressed_file_path)], check=True)
            print(f"Existing compressed file {compressed_file_path} removed.")
        except subprocess.CalledProcessError:
            print(f"Failed to remove existing compressed file {compressed_file_path}.")

    try:
        # Compress file using pigz, keeping the original file
        subprocess.run(['pigz', '-k', '-3', filename], check=True)
        print(f"{filename} compressed successfully using pigz.")
    except subprocess.CalledProcessError:
        print(f"Failed to compress {filename} using pigz.")    

compress_file("trace.json")

import torch

torch.backends.cuda.matmul.allow_tf32 = True   # ok; irrelevant if using fp16/bf16 inputs
device = "cuda"

M,N,K = 2048, 4096, 6144
A = torch.randn(M, K, device=device, dtype=torch.float16)
B = torch.randn(K, N, device=device, dtype=torch.float16)

# Warmup
for _ in range(10):
    C = A @ B

torch.cuda.synchronize()
import time
t0 = time.time()
for _ in range(20):
    C = A @ B
torch.cuda.synchronize()
t1 = time.time()

iters = 20
flops = 2 * M * N * K
t_avg = (t1 - t0) / iters
tflops = flops / t_avg / 1e12
print(f"FP16 GEMM: {tflops:.1f} TFLOP/s")

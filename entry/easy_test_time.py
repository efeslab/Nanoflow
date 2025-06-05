import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

origin_C = torch.randn((4096, 4096), dtype=torch.float16, device='cuda')

start.record()
C = origin_C[0: 1024, :].contiguous()
end.record()
torch.cuda.synchronize()
print("Time taken to slice C ROWWISE:", start.elapsed_time(end), "ms")

start.record()
C = origin_C[:, 0: 1024].contiguous()
end.record()
torch.cuda.synchronize()
print("Time taken to slice C COLUMNWISE:", start.elapsed_time(end), "ms")


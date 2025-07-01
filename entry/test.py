import sys
sys.path.append("../")
sys.path.append("../pybind/build")
from utils.greenctx import create_greenctx
import torch
# Create green context streams for testing
normal_stream = torch.cuda.Stream()
test_stream_01, test_stream_09, test_stream_01_sm, test_stream_09_sm = create_greenctx(0.1, 0.9, 0)
g = torch.cuda.CUDAGraph()
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
with torch.cuda.graph(g, stream=test_stream_01):
    for round in range(10):
        with torch.cuda.stream(test_stream_01):
            z = torch.add(x, y)
torch.cuda.synchronize()

with torch.cuda.stream(test_stream_09):
    g.replay()
torch.cuda.synchronize()
print("Test completed successfully.")
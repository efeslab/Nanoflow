import sys
sys.path.append('..')
import pybind_amd.bind_marker.build.bind_marker as marker
import torch
print(marker.__file__)


marker.roctxMark('Starting computation')


# Push a range
marker.roctxRangePush('Computation phase')

a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c= a + b

# Pop the range
marker.roctxRangePop()

d = torch.randn(1000, 1000, device='cuda')
marker.roctxRangePush('Computation phase 2')
e = a * d
# Pop the range
marker.roctxRangePop()

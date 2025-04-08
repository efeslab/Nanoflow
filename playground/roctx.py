import ctypes
import torch

roctx = ctypes.CDLL("libroctx64.so")

# Define roctxMarkA function
roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
roctx.roctxMarkA.restype = ctypes.c_int

# Define roctxRangePushA function
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
roctx.roctxRangePushA.restype = ctypes.c_int

# Define roctxRangePop function
roctx.roctxRangePop.argtypes = []
roctx.roctxRangePop.restype = ctypes.c_int

roctx.roctxMarkA(b'Starting computation')

# Push a range
roctx.roctxRangePushA(b'Computation phase')

a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c= a + b

# Pop the range
roctx.roctxRangePop()

d = torch.randn(1000, 1000, device='cuda')
roctx.roctxRangePushA(b'Computation phase 2')
e = a * d
# Pop the range
roctx.roctxRangePop()

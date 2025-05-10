import cupy.cuda.nccl as nccl
from mpi4py import MPI
import cupy as cp


class NcclAllReduce:
    def __init__(self, nccl_comm: nccl.NcclCommunicator, memory: cp.ndarray):
        self.nccl_comm = nccl_comm
        self.memory = memory
        if memory.dtype == cp.float32:
            self.nccl_dtype = nccl.NCCL_FLOAT32
        elif memory.dtype == cp.float16:
            self.nccl_dtype = nccl.NCCL_FLOAT16
        elif memory.dtype == cp.int32:
            self.nccl_dtype = nccl.NCCL_INT32
        else:
            raise RuntimeError("Make sure that the data type is mapped to the correct NCCL data type")

    def __call__(self, stream):
        stream_ptr = stream.ptr if stream else 0
        self.nccl_comm.allReduce(
            self.memory.data.ptr, self.memory.data.ptr, self.memory.size, self.nccl_dtype, nccl.NCCL_SUM, stream_ptr
        )
        return self.memory

############################################
#
# Modified from: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/punica/utils/kvcache.py
# Distributed version of KV-Cache: Head-parallelism
#
############################################

from typing import Sequence

import sys
sys.path.append('../build')
import pllm_python
import torch
from pybindUtil import toGPU, toGPUTensor
from torch.profiler import profile, record_function, ProfilerActivity
import time
from typing import List


class DistKVPool:
    """
    Automatically mangages a memory pool, which is distributed on available devices.
    Use Head-parallelism, therefore all GPUs are identical.
    Memory Pool is mananged at the granularity of page.
    """
    def __init__(
      self,
      num_layers: int,
      num_kv_heads: int,
      head_dim: int,
      capacity: int,
      page_size: int,
      num_devices: int,
    ):
        self.available_devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
        # Test whether the devices are available
        for device in self.available_devices:
            torch.zeros(1, device=device)
            
        # NOTE(Yilong): Assume underlying layout is HND.
        assert num_kv_heads % num_devices == 0, "num_kv_heads must be divisible by num_devices"
            
        # Metadata is identical for all GPUs
        self._free = set(range(capacity))
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size
        self.num_devices = num_devices
            
    @property
    def num_free_pages(self) -> int:
        return len(self._free)

    def alloc_page(self) -> int:
        assert len(self._free) > 0, "Out of memory"
        idx = self._free.pop()
        return idx

    def free_page(self, idx: int):
        # assert 0 <= idx < self._buf[0].size(1), "Invalid page index"
        assert idx not in self._free
        self._free.add(idx)
        
class DistKVCache:
    """
    A distributed key-value cache.
    Maintains metadata per requests.
    No actual memory is allocated here.
    """
    
    def __init__(self, pool: DistKVPool):
        self._pool = pool
        self._indices : List[int] = []
        self._seqlen : int = 0
    
    @property
    def seqlen(self) -> int:
        return self._seqlen
    
    @property
    def indicies(self) -> List[int]:
        return self._indices
    
    @property
    def last_page_offset(self) -> int:
        return (self.seqlen - 1) % self._pool.page_size + 1
    
    def allocate_tokens(self, num_tokens: int):
        assert 0 < num_tokens <= self._pool.num_free_pages * self._pool.page_size, "Out of memory"
        # Appended tokens = num_tokens - page_size + last_page_offset
        # Appended pages = (Appended tokens + page_size - 1) // page_size
        num_appended_pages = (num_tokens + self.last_page_offset - 1) // self._pool.page_size
        for _ in range(num_appended_pages):
            self._indices.append(self._pool.alloc_page())
        self._seqlen += num_tokens
    
    def release(self):
        """Release all pages"""
        self._seqlen = 0
        for idx in self._indices:
            self._pool.free_page(idx)
        self._indices.clear()
    
class BatchedDistKVCache():
    """
    Function class for arranging metadata of multiple requests within the entire batch.
    Layout follows descriptions in `../include/vortexData.cuh`.
    """
    def __init__(self, decode_kvs: Sequence[DistKVCache], prefill_kvs: Sequence[DistKVCache]):
        """
        Given all useful metadata, arrange them into a pre-defined layout.
        Basically arrange all decode to the start of the matrix and squeeze the prefill to the end.
        """
        # batch_size = len(decode_kvs) + len(prefill_kvs)
        # [batch_size + 1,]
        self._kv_indptr : List[int] = [0]
        # [num_pages_in_total, ]
        self._kv_indices : List[int] = []
        # [batch_size, ]
        self._kv_last_page_len : List[int] = []
        
        # Here we do not materialize data into specific devices,
        # for distributed assignment.
        for kv in decode_kvs:
            self._kv_indices.extend(kv.indicies)
            self._kv_last_page_len.append(kv.last_page_offset)
            self._kv_indptr.append(self._kv_indptr[-1] + len(kv.indicies))
        
        for kv in prefill_kvs:
            self._kv_indices.extend(kv.indicies)
            self._kv_last_page_len.append(kv.last_page_offset)
            self._kv_indptr.append(self._kv_indptr[-1] + len(kv.indicies))

    def prepare(self):
        self.newTensor = torch.tensor(self._kv_indices, dtype=torch.int32, device='cpu').pin_memory()
    
    def toGPU(self, nrnaks: int):
        """
        Assign metadata to devices.
        """
        with record_function("batch_kv_kv_indices"):
            print("kv indices len: ", len(self._kv_indices))
            [self.kv_indices_ptr, self._kv_indices_tensor] = toGPUTensor(self.newTensor, nrnaks, dtype=torch.int32)
        with record_function("batch_kv_kv_indptr"):
            [self.kv_indptr_ptr, self._kv_indptr_tensor] = toGPU(self._kv_indptr, nrnaks, dtype=torch.int32)
        with record_function("batch_kv_kv_last_page_len"):
            [self.kv_last_page_len_ptr, self._kv_last_page_len_tensor] = toGPU(self._kv_last_page_len, nrnaks, dtype=torch.int32)
        
        return self.kv_indices_ptr, self.kv_indptr_ptr, self.kv_last_page_len_ptr
    
    def toCPUPinned(self):
        self.pinned_kv_indices = torch.tensor(self._kv_indices, dtype=torch.int32, device='cpu').pin_memory()
        self.pinned_kv_indptr = torch.tensor(self._kv_indptr, dtype=torch.int32, device='cpu').pin_memory()
        self.pinned_kv_last_page_len = torch.tensor(self._kv_last_page_len, dtype=torch.int32, device='cpu').pin_memory()
        return self.pinned_kv_indices.data_ptr(), self.pinned_kv_indptr.data_ptr(), self.pinned_kv_last_page_len.data_ptr()
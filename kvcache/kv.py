############################################
#
# Modified from: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/punica/utils/kvcache.py
# Distributed version of KV-Cache: Head-parallelism
#
############################################

from typing import Sequence

import torch
# from pybindUtil import toGPU, toGPUTensor
from torch.profiler import profile, record_function, ProfilerActivity
import time

class KVCacheNone():
    def __init__(self):
        self.cache = {}
    
    def put(self, idx, key, value):
        self.cache[idx] = (key, value)

    def get(self, idx):
        return self.cache.get(idx, None)

class KVCacheTorch():
    def __init__(self):
        self.cache = {}
        self.cache_indices = {}
        self.hidden_dim = 1024
        self.max_size_per_request = 2048
    
    def put(self, idx, key, value):
        if idx not in self.cache:
            reserved_key = torch.empty((self.max_size_per_request, self.hidden_dim), dtype=key.dtype, device=key.device)
            reserved_value = torch.empty((self.max_size_per_request, self.hidden_dim), dtype=value.dtype, device=value.device)
            
            # Insert the provided key and value at the beginning of the reserved space.
            reserved_key[:key.shape[0]] = key
            reserved_value[:value.shape[0]] = value

            # Store the reserved tensors in the cache.
            self.cache[idx] = (reserved_key, reserved_value)
            self.cache_indices[idx] = (key.shape[0], value.shape[0])

        else:
            old_key, old_value = self.cache[idx]
            key_offset, value_offset = self.cache_indices[idx]
            assert key_offset + key.shape[0] <= self.max_size_per_request, "Key size exceeds maximum size"
            assert value_offset + value.shape[0] <= self.max_size_per_request, "Value size exceeds maximum size"
            old_key[key_offset:key_offset + key.shape[0]] = key
            old_value[value_offset:value_offset + value.shape[0]] = value
            self.cache_indices[idx] = (key_offset + key.shape[0], value_offset + value.shape[0])
            
    def get(self, idx):
        if idx in self.cache:
            reserved_key, reserved_value = self.cache[idx]
            key_offset, value_offset = self.cache_indices[idx]
            return reserved_key[:key_offset], reserved_value[:value_offset]
        return None
    
    def get_indices(self, idx):
        return self.cache_indices.get(idx, None)

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
        self._free = [set(range(capacity)) for _ in range(num_layers)]
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size
        self.num_devices = num_devices
        # kv_data format is "HND"
        self.kv_shape = [num_layers, capacity, num_kv_heads, page_size, head_dim]
        self.k_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[0])
        self.v_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[0])
            
    # @property
    def num_free_pages(self, layer) -> int:
        return len(self._free[layer])

    def alloc_page(self, layer) -> int:
        assert len(self._free[layer]) > 0, "Out of memory"
        idx = self._free[layer].pop()
        return idx

    def free_page(self, layer: int,  idx: int):
        # assert 0 <= idx < self._buf[0].size(1), "Invalid page index"
        assert idx not in self._free[layer]
        self._free[layer].add(idx)
        
class DistKVCache:
    """
    A distributed key-value cache.
    Maintains metadata per requests.
    No actual memory is allocated here.
    """
    
    def __init__(self, pool: DistKVPool):
        self._pool = pool
        self._indices : list[int] = []
        self._seqlen : int = 0
        self.page_size = pool.page_size
    
    @property
    def seqlen(self) -> int:
        return self._seqlen
    
    @property
    def indicies(self) -> list[int]:
        return self._indices
    
    @property
    def last_page_offset(self) -> int:
        # print("self.seqlen:", self.seqlen)
        # print("self._pool.page_size:", self._pool.page_size)
        # return self.seqlen % self._pool.page_size
        return (self.seqlen - 1) % self._pool.page_size + 1
    
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
    def __init__(self, pool: DistKVPool, layer: int):
        """
        Given all useful metadata, arrange them into a pre-defined layout.
        Basically arrange all decode to the start of the matrix and squeeze the prefill to the end.
        """ 
        self._pool = pool
        self._layer = layer
        self.cache = {}
        self._kv_indptr : list[int] = [0]
        # [num_pages_in_total, ]
        self._kv_indices : list[int] = []
        # [batch_size, ]
        self._kv_last_page_len : list[int] = []

    def get_pool(self):
        return self._pool

    def put_abstract(self, idx: int, num_tokens: int):
        if idx not in self.cache:
            self.cache[idx] = DistKVCache(self._pool)
        assert 0 < num_tokens <= self._pool.num_free_pages(self._layer) * self._pool.page_size, f"Out of memory for {num_tokens} tokens, {self._pool.num_free_pages(self._layer)} pages left"
        # Appended tokens = num_tokens - page_size + last_page_offset
        # Appended pages = (Appended tokens + page_size - 1) // page_size
        num_appended_pages = (num_tokens + self.cache[idx].last_page_offset - 1) // self._pool.page_size
        for _ in range(num_appended_pages):
            self.cache[idx]._indices.append(self._pool.alloc_page(self._layer))
        # print("indices:", self.cache[idx]._indices)
        # print("before adding num_tokens:", self.cache[(layer, idx)]._seqlen)
        self.cache[idx]._seqlen += num_tokens
        # print("after adding num_tokens:", self.cache[(layer, idx)]._seqlen)

    def get(self, idx: int):
        # if (layer, idx) not in self.cache:
        #     self.cache[(layer, idx)] = DistKVCache(self._pool)
        #     return None, None
        # else:
        kvcache = self.cache[idx]
        # print("shape of self._pool.kv_data[kvcache.indicies[:-1], 0]: ", self._pool.kv_data[kvcache.indicies[:-1], 0].shape)
        # print("kvcache.indicies[:-1]:", kvcache.indicies[:-1])
        # print("kv cache indicies:", kvcache.indicies)
        # print("kv cache last page offset:", kvcache.last_page_offset)

        ki = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 0]
                self._pool.k_data[self._layer, kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 0, :, :kvcache.last_page_offset, :]
                    self._pool.k_data[self._layer,kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 1]
                self._pool.v_data[self._layer,kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 1, :, :kvcache.last_page_offset, :]
                    self._pool.v_data[self._layer, kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        return ki, vi

    def get_whole_kv_data(self):
        return self._pool.k_data[self._layer], self._pool.v_data[self._layer]


    def get_seqlen(self, idx: int):
        return self.cache[idx].seqlen


    def update(self):
        # batch_size = len(decode_kvs) + len(prefill_kvs)
        # [batch_size + 1,]
        self._kv_indptr : list[int] = [0]
        # [num_pages_in_total, ]
        self._kv_indices : list[int] = []
        # [batch_size, ]
        self._kv_last_page_len : list[int] = []
        
        # Here we do not materialize data into specific devices,
        # for distributed assignment.
        for _, kv in self.cache.items():
            self._kv_indices.extend(kv.indicies)
            # print("kv.last_page_offset:", kv.last_page_offset)
            self._kv_last_page_len.append(kv.last_page_offset)
            self._kv_indptr.append(self._kv_indptr[-1] + len(kv.indicies))
        # print("updated_kv_indices:", self._kv_indices)
        # print("updated_kv_last_page_len:", self._kv_last_page_len)
        # print("updated_kv_indptr:", self._kv_indptr)
        
        return self._kv_indptr, self._kv_indices, self._kv_last_page_len

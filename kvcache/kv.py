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
        self.name = 'No KV Cache'
        self.cache = {}
    
    def put(self, idx, key, value):
        self.cache[idx] = (key, value)

    def get(self, idx):
        return self.cache.get(idx, None)

class KVCacheTorch():
    def __init__(self):
        self.name = 'Torch KV Cache'
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
        # self.available_devices = [torch.device("cuda:3")]
        # print("available devices:", self.available_devices)
        # Test whether the devices are available
        for device in self.available_devices:
            # print("device:", device)
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
        # kv_data format is "HND"
        self.kv_shape = [num_layers, capacity, num_kv_heads, page_size, head_dim]
        self.k_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[0])
        self.v_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[0])
            
    # @property
    def num_free_pages(self) -> int:
        return len(self._free)

    def alloc_page(self) -> int:
        assert len(self._free) > 0, "Out of memory"
        idx = self._free.pop()
        return idx

    def put_for_profile(self, layer: int, batch_size: int, k: torch.Tensor, v: torch.Tensor):
        needed_pages = (batch_size + self.page_size - 1) // self.page_size
        last_offset = batch_size % self.page_size
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)
        assert needed_pages <= self.num_free_pages(layer), "Out of memory"
        for i in range(needed_pages - 1):
            idx = self.alloc_page(layer)
            self.k_data[layer, idx] = k_t[:, i * self.page_size : (i + 1) * self.page_size, :]
            self.v_data[layer, idx] = v_t[:, i * self.page_size : (i + 1) * self.page_size, :]
        idx = self.alloc_page(layer)
        self.k_data[layer, idx, :, :last_offset, :] = k_t[:, (needed_pages - 1) * self.page_size : needed_pages * self.page_size + last_offset, :]
        self.v_data[layer, idx, :, :last_offset, :] = v_t[:, (needed_pages - 1) * self.page_size : needed_pages * self.page_size + last_offset, :]

    def free_page(self,  idx: int):
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
    def __init__(self, pool: DistKVPool):
        """
        Given all useful metadata, arrange them into a pre-defined layout.
        Basically arrange all decode to the start of the matrix and squeeze the prefill to the end.
        """ 
        self.name = 'Flashinfer KV Cache'
        self._pool = pool
        self.cache = {}
        self._kv_indptr = torch.tensor([0], dtype=torch.int32, device=self._pool.k_data.device)
        self._kv_indices = torch.tensor([], dtype=torch.int32, device=self._pool.k_data.device)
        self._kv_last_page_len = torch.tensor([], dtype=torch.int32, device=self._pool.k_data.device)

    def get_pool(self):
        return self._pool

    def pre_allocate(self, idx: int, num_tokens: int):
        if idx not in self.cache:
            self.cache[idx] = DistKVCache(self._pool)
        assert 0 < num_tokens <= self._pool.num_free_pages() * self._pool.page_size, f"Out of memory for {num_tokens} tokens, {self._pool.num_free_pages()} pages left"
        # Appended tokens = num_tokens - page_size + last_page_offset
        # Appended pages = (Appended tokens + page_size - 1) // page_size
        num_appended_pages = (num_tokens + self.cache[idx].last_page_offset - 1) // self._pool.page_size
        for _ in range(num_appended_pages):
            self.cache[idx]._indices.append(self._pool.alloc_page())
        # print("indices:", self.cache[idx]._indices)
        # print("before adding num_tokens:", self.cache[(layer, idx)]._seqlen)
        self.cache[idx]._seqlen += num_tokens
        # print("after adding num_tokens:", self.cache[(layer, idx)]._seqlen)

    def get(self, layer: int, idx: int):
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
                self._pool.k_data[layer, kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 0, :, :kvcache.last_page_offset, :]
                    self._pool.k_data[layer,kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 1]
                self._pool.v_data[layer,kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 1, :, :kvcache.last_page_offset, :]
                    self._pool.v_data[layer, kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        return ki, vi

    def get_whole_kv_data(self, layer: int):
        return self._pool.k_data[layer], self._pool.v_data[layer]

    def get_whole_kv_data_all_layers(self):
        return self._pool.k_data, self._pool.v_data

    def get_seqlen(self, idx: int):
        return self.cache[idx].seqlen

    def update(self):
        # Here we do not materialize data into specific devices,
        # for distributed assignment.
        kv_indptr_list = [0]
        kv_indices_list = []
        kv_last_page_len_list = []
        for _, kv in self.cache.items():
            kv_indptr_list.append(kv_indptr_list[-1] + len(kv.indicies))
            kv_indices_list.extend(kv.indicies)
            kv_last_page_len_list.append(kv.last_page_offset)
        self._kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=self._pool.k_data.device)
        self._kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=self._pool.k_data.device)
        self._kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=self._pool.k_data.device)

        return self._kv_indptr, self._kv_indices, self._kv_last_page_len
        

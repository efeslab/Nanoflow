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
from utils.prof_marker import prof_marker
import time

class KVCacheNone():
    def __init__(self):
        self.name = 'No KV Cache'
        self.cache = {}
    
    def put(self, layer, idx, key, value):
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)
    
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device):
        self.input_req_idx = input_req_idx
        return None

    def get_whole_kv_data(self, device, layer: int):
        return None, None
    
    def get_whole_kv_data_all_layers(self, device):
        return None, None
    
    def get_indices(self, layer, idx):
        return 0

class KVCacheTorch():
    def __init__(self, num_kv_heads, head_dim, tp_size=1):
        self.name = 'Torch KV Cache'
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.cache = {}
        self.cache_indices = {}
        self.hidden_dim = num_kv_heads * head_dim // tp_size
        self.max_size_per_request = 2048
    
    def put(self, layer, idx, key, value):
        if (layer, idx) not in self.cache:
            reserved_key = torch.empty((self.max_size_per_request, self.hidden_dim), dtype=key.dtype, device=key.device)
            reserved_value = torch.empty((self.max_size_per_request, self.hidden_dim), dtype=value.dtype, device=value.device)
            
            # Insert the provided key and value at the beginning of the reserved space.
            reserved_key[:key.shape[0]] = key
            reserved_value[:value.shape[0]] = value

            # Store the reserved tensors in the cache.
            self.cache[(layer, idx)] = (reserved_key, reserved_value)
            self.cache_indices[(layer, idx)] = key.shape[0]
            # print(f"put the request {layer}, {idx}")

        else:
            old_key, old_value = self.cache[(layer, idx)]
            kv_offset = self.cache_indices[(layer, idx)]
            assert kv_offset + key.shape[0] <= self.max_size_per_request, "Key size exceeds maximum size"
            assert kv_offset + value.shape[0] <= self.max_size_per_request, "Value size exceeds maximum size"
            old_key[kv_offset:kv_offset + key.shape[0]] = key
            old_value[kv_offset:kv_offset + value.shape[0]] = value
            self.cache_indices[(layer, idx)] = kv_offset + key.shape[0]
            
    def get(self, layer, idx):
        # print(f"find the request {layer}, {idx}")
        if (layer, idx) in self.cache:
            # print(f"{layer, idx} is in kv cache.")
            reserved_key, reserved_value = self.cache[(layer, idx)]
            kv_offset = self.cache_indices[(layer, idx)]
            return reserved_key[:kv_offset], reserved_value[:kv_offset]
        # print(f"{layer, idx} is not in kv cache.")
        raise ValueError(f"Request {layer, idx} not found in cache")
        return None
    
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device):
       self.input_req_idx = input_req_idx
       return None

    def get_indices(self, layer, idx):
        return self.cache_indices.get((layer, idx), 0)
    def get_whole_kv_data(self, device, layer: int):
        return None, None
    def get_whole_kv_data_all_layers(self, device):
        return None, None

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
      device_list: list[str],
    ):
        self.available_devices = [torch.device(dev) for dev in device_list]
        # self.available_devices = [torch.device("cuda:3")]
        # print("available devices:", self.available_devices)
        # Test whether the devices are available
        for device in self.available_devices:
            # print("device:", device)
            torch.zeros(1, device=device)
        
        # # NOTE(Yilong): Assume underlying layout is HND.
        # assert num_kv_heads % len(device_list) == 0, "num_kv_heads must be divisible by num_devices"
            
        # Metadata is identical for all GPUs
        self._free = set(range(capacity))
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size
        self.device_list = device_list
        # kv_data format is "HND"
        self.kv_shape = [num_layers, capacity, num_kv_heads, page_size, head_dim]
        self.k_datas = {}
        self.v_datas = {} 
        for device in self.available_devices:
            # print("device:", str(device))
            k_data = torch.empty(self.kv_shape, dtype=torch.float16, device=device)
            v_data = torch.empty(self.kv_shape, dtype=torch.float16, device=device)
            self.k_datas[str(device)] = k_data
            self.v_datas[str(device)] = v_data
            
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
        device_list = pool.device_list
        self.cache = {}
        self.kv_indptr_devices = dict([(device, torch.tensor([0], dtype=torch.int32, device=device)) for device in device_list])
        self.kv_indices_devices = dict([(device, torch.tensor([], dtype=torch.int32, device=device)) for device in device_list])
        self.kv_last_page_len_devices = dict([(device, torch.tensor([], dtype=torch.int32, device=device)) for device in device_list])
        self.rev_input_indptr_devices = dict([(device, torch.tensor([], dtype=torch.int32, device=device)) for device in device_list])
        self.per_token_offset_devices = dict([(device, torch.tensor([], dtype=torch.int32, device=device)) for device in device_list])

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

    def get(self, device: str, layer: int, idx: int):
        kvcache = self.cache[idx]
        ki = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 0]
                self._pool.k_datas[device][layer, kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 0, :, :kvcache.last_page_offset, :]
                    self._pool.k_datas[device][layer,kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 1]
                self._pool.v_datas[device][layer,kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 1, :, :kvcache.last_page_offset, :]
                    self._pool.v_datas[device][layer, kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        return ki, vi

    def get_whole_kv_data(self, device, layer: int):
        return self._pool.k_datas[device][layer], self._pool.v_datas[device][layer]

    def get_whole_kv_data_all_layers(self, device):
        return self._pool.k_datas[device], self._pool.v_datas[device]

    def get_seqlen(self, idx: int):
        return self.cache[idx].seqlen
        
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device):
        total_tokens = cumsum_input[-1]
        rev_input_indptr_tensor = torch.empty(total_tokens, dtype=torch.int32)
        per_token_offset_tensor = torch.empty(total_tokens, dtype=torch.int32)

        rev_input_indptr_tensor[0:decode_batchsize] = torch.arange(decode_batchsize, dtype=torch.int32)
        for temp_idx in range(decode_batchsize):
            global_req_idx = input_req_idx[temp_idx]
            self.pre_allocate(global_req_idx, 1)
            seq_len = self.get_seqlen(global_req_idx)
            per_token_offset_tensor[temp_idx] = seq_len - 1

        for temp_idx in range(decode_batchsize, len(cumsum_input) - 1):
            global_req_idx = input_req_idx[temp_idx]
            start = cumsum_input[temp_idx]
            end = cumsum_input[temp_idx + 1]
            count = end - start
            self.pre_allocate(global_req_idx, count)
            seq_len = self.get_seqlen(global_req_idx)
            # append i to the rev_input_indptr for end-start times
            rev_input_indptr_tensor[start:end] = temp_idx
            # extend the per_token_offset with a list from last_offest to last_offest + (end - start)
            per_token_offset_tensor[start:end] = torch.arange(seq_len - count, seq_len, dtype=torch.int32)

        self.rev_input_indptr_devices[device] = rev_input_indptr_tensor.to(device)
        self.per_token_offset_devices[device] = per_token_offset_tensor.to(device)

        num_reqs = len(input_req_idx)
        kv_counts = [len(self.cache[req_idx].indicies) for req_idx in input_req_idx]
        total_kv_tokens = sum(kv_counts)
        kv_indptr_tensor = torch.empty(num_reqs + 1, dtype=torch.int32)
        kv_indices_tensor = torch.empty(total_kv_tokens, dtype=torch.int32)
        kv_last_page_len_tensor = torch.empty(num_reqs, dtype=torch.int32)

        cur_offset = 0
        kv_indptr_tensor[0] = 0
        for i, global_req_idx in enumerate(input_req_idx):
            if global_req_idx not in self.cache:
                raise ValueError(f"Request {global_req_idx} not found in cache")
            kv = self.cache[global_req_idx]
            count = len(kv.indicies)
            
            kv_indices_tensor[cur_offset : cur_offset + count] = torch.tensor(kv.indicies, dtype=torch.int32)
            kv_indptr_tensor[i + 1] = cur_offset + count
            kv_last_page_len_tensor[i] = kv.last_page_offset
            cur_offset += count
        
        self.kv_indptr_devices[device] = kv_indptr_tensor.to(device)
        self.kv_indices_devices[device] = kv_indices_tensor.to(device)
        self.kv_last_page_len_devices[device] = kv_last_page_len_tensor.to(device)

    @property
    def page_size(self):
        return self._pool.page_size
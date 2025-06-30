############################################
#
# Modified from: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/punica/utils/kvcache.py
# Distributed version of KV-Cache: Head-parallelism
#
############################################

import logging
from typing import Sequence

import torch
# from pybindUtil import toGPU, toGPUTensor
from torch.profiler import profile, record_function, ProfilerActivity
from utils.prof_marker import prof_marker
import time

from .triton.kv_copy import copy_fa_nopage_kvcache, copy_torch_kvcache


class KVCacheNone:
    def __init__(self):
        self.name = 'No KV Cache'
        self.cache = {}
    
    def put(self, layer, idx, key, value):
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)
    
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device_id):
        self.input_req_idx = input_req_idx
        return None

    def get_whole_kv_data(self, device_id, layer: int):
        return None, None
    
    def get_whole_kv_data_all_layers(self, device_id):
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
            reserved_key = torch.empty(
                (self.max_size_per_request, self.hidden_dim),
                dtype=key.dtype,
                device=key.device,
            )
            reserved_value = torch.empty(
                (self.max_size_per_request, self.hidden_dim),
                dtype=value.dtype,
                device=value.device,
            )

            # Insert the provided key and value at the beginning of the reserved space.
            reserved_key[: key.shape[0]] = key
            reserved_value[: value.shape[0]] = value

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
    
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device_id):
       self.input_req_idx = input_req_idx
       return None

    def get_indices(self, layer, idx):
        return self.cache_indices.get((layer, idx), 0)
    def get_whole_kv_data(self, device_id, layer: int):
        return None, None
    def get_whole_kv_data_all_layers(self, device_id):
        return None, None



class KVCacheBatched:
    r"""Batched KV cache for FlashAttention without page."""
    def __init__(
        self,
        *,
        num_layers,
        num_heads,
        head_dim,
        max_seqlen: int = 256,
        device_id: int = 0,
        tp_size: int = 1,
    ) -> None:
        r"""Initialize the KV cache.

        Parameters
        ----------
        device_id : int
            The device ID to use for the cache.
        num_layers : int
            The number of layers in the model.
        num_heads : int
            The number of KV heads in the model.
        head_dim : int
            The dimension of each attention head.
        max_size_per_request : int
            The maximum sequence length for each request.
        """

        self.name = "FlashAttention KV Cache (No Page)"
        self.k_cache: list[torch.Tensor] | None = None # Lazy initialized
        self.v_cache: list[torch.Tensor] | None = None
        self.batch_size: int | None = None
        self.indices: torch.Tensor | None = None
        self.device_id = device_id
        self.num_layers = num_layers
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.max_size_per_request = max_seqlen
        self.tp_size = tp_size
        self.input_req_idx: torch.Tensor | None = None
        self.last_key: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None


    def get_indices(self, start_req_idx: int, end_req_idx: int) -> torch.Tensor:
        return self.indices[start_req_idx:end_req_idx]


    def get_last_kv(self, start_idx: int, end_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.last_key[start_idx:end_idx],
            self.last_value[start_idx:end_idx],
        )


    def get_kv_data(
        self, layer: int, start_req_idx: int, end_req_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[layer][start_req_idx:end_req_idx],
            self.v_cache[layer][start_req_idx:end_req_idx]
        )    


    def update(self, input_req_idx: list[int], qo_indices: list[int]) -> None:
        r"""Update the KV cache with a new batch size.

        Parameters
        ----------
        batch_size : int
            The batch size to use for the cache.
        
        Notes
        -----
        This function (re)initializes the KV cache with the given batch size,
        and should not be called multiple times for a single batch.
        """
        logging.info(f"KVCache updated on device {self.device_id} with batch {qo_indices}")
        batch_size = len(input_req_idx)
        self.input_req_idx = torch.tensor(input_req_idx, dtype=torch.int32, device=f"cuda:{self.device_id}")
        if self.batch_size == batch_size:
            qo_seqlens = torch.tensor(qo_indices).diff().to(self.indices.device)
            self.indices += qo_seqlens
            return
        old_k_cache, old_v_cache, old_indices = (
            self.k_cache,
            self.v_cache,
            self.indices,
        )
        self.batch_size = batch_size
        self.indices = torch.zeros((self.batch_size,), dtype=torch.int32, device=f"cuda:{self.device_id}")
        self.last_key = torch.zeros(
            qo_indices[-1],
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=f"cuda:{self.device_id}",
        )
        self.last_value = torch.zeros(
            qo_indices[-1],
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=f"cuda:{self.device_id}",
        )
        self.k_cache = [
            torch.zeros(
                batch_size,
                self.max_size_per_request,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device=f"cuda:{self.device_id}",
            ) for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.zeros(
                batch_size,
                self.max_size_per_request,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device=f"cuda:{self.device_id}",
            ) for _ in range(self.num_layers)
        ]
        if old_k_cache is not None and old_v_cache is not None and old_indices is not None:
            self.indices[: old_indices.shape[0]] = old_indices
            for i in range(self.num_layers):
                self.k_cache[i][: old_k_cache[i].shape[0]] = old_k_cache[i]
                self.v_cache[i][: old_v_cache[i].shape[0]] = old_v_cache[i]
        qo_seqlens = torch.tensor(qo_indices).diff().to(self.indices.device)
        self.indices += qo_seqlens


    def store_last_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_idx: int = 0,
        end_idx: int = -1
    ) -> None:
        r"""Store the last computed key and value tensors.

        Parameters
        ----------
        key : torch.Tensor
            The key tensor to store.
        value : torch.Tensor
            The value tensor to store.
        
        Notes
        -----
        This is a hack to allow FlashAttention prefill kernel to access the
        last computed key and value tensors.
        """
        assert self.last_key is not None and self.last_value is not None, "Cache not initialized. Call update() first."
        self.last_key[start_idx:end_idx].copy_(
            key.view(
                end_idx - start_idx,
                self.num_heads,
                self.head_dim,
            )
        )
        self.last_value[start_idx:end_idx].copy_(
            value.view(
                end_idx - start_idx,
                self.num_heads,
                self.head_dim,
            )
        )

    def put_batch(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        rev_input_indices: torch.Tensor,
        per_token_offset: torch.Tensor,
    ) -> None:
        r"""Put a batch of key and value tensors into the KV cache.

        Parameters
        ----------
        layer : int
            The layer ID to put the cache for.
        qo_indices : torch.Tensor
            The indices of the queries in the batch.
            Shape: [batch_size + 1,]
        key : torch.Tensor
            The key tensor to put into the cache.
            Shape: [batch_size, key_dim]
        value : torch.Tensor
            The value tensor to put into the cache.
            Shape: [batch_size, value_dim]
        rev_input_indices : torch.Tensor
            The reverse input indices for the batch.
            Shape: [batch_size,]
        per_token_offset : torch.Tensor
            The per token offset for the batch.
            Shape: [batch_size,]
        """
        if self.k_cache is None or self.v_cache is None or self.batch_size is None:
            raise ValueError("Cache not initialized. Call update() first.")
        if layer == 0:
            assert self.indices is not None, "Cache not initialized. Call update() first."
        kv_cache_shape = (self.batch_size, self.max_size_per_request, self.num_heads * self.head_dim)
        key_cache = self.k_cache[layer].view(kv_cache_shape)
        value_cache = self.v_cache[layer].view(kv_cache_shape)
        copy_fa_nopage_kvcache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            rev_input_indices=rev_input_indices,
            per_token_offset=per_token_offset,
        )


try:
    from vllm._custom_ops import reshape_and_cache
    VLLM_CACHE = True
except ImportError:
    VLLM_CACHE = False


class KVCachevLLM:
    r"""Paged key-value cache for vLLM backend."""

    def __init__(
        self,
        *,
        num_layers,
        num_heads,
        head_dim,
        max_seqlen: int,
        max_batch_size: int,
        block_size: int = 32,
        device_id: int = 0,
        dtype: torch.dtype = torch.float16,
        tp_size: int = 1,
    ) -> None:
        r"""Initialize the KV cache.

        Parameters
        ----------
        num_layers : int
            The number of layers in the model.
        num_heads : int
            The number of KV heads in the model.
        head_dim : int
            The dimension of each attention head.
        block_size : int
            The size of each block in the KV cache.
        max_blocks_per_request : int
            The maximum number of blocks per request.
        device_id : int
            The device ID to use for the cache.
        tp_size : int
            The tensor parallelism size.
        """
        assert VLLM_CACHE, "vLLM KV Cache requires vLLM custom ops to be installed."
        self.name = "vLLM KV Cache"
        self.device_id = device_id
        self.dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.max_blocks_per_request = (max_seqlen + block_size - 1) // block_size
        self.max_seqlen = self.max_blocks_per_request * self.block_size
        x = 16 // torch.tensor([], dtype=self.dtype).element_size()
        self.k_cache = [
            torch.zeros(
                max_batch_size * self.max_blocks_per_request,
                self.num_heads,
                self.head_dim // x,
                self.block_size,
                x,
                dtype=torch.float16,
                device=f"cuda:{self.device_id}",
            ) for _ in range(self.num_layers)
        ]
        self.v_cache = [
            torch.zeros(
                max_batch_size * self.max_blocks_per_request,
                self.num_heads,
                self.head_dim,
                self.block_size,
                dtype=torch.float16,
                device=f"cuda:{self.device_id}",
            ) for _ in range(self.num_layers)
        ]
        self.block_table = torch.stack([
            torch.arange(0, self.max_blocks_per_request) + i * self.max_blocks_per_request
            for i in range(max_batch_size)
        ], dim=0).to(device=f"cuda:{self.device_id}", dtype=torch.int32)
        self.indices = torch.zeros((max_batch_size,), dtype=torch.int32, device="cpu")
        self.last_key: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None
        self.unscaled = torch.tensor([1], dtype=torch.int32)


    def get_indices(self, start_req_idx: int, end_req_idx: int) -> torch.Tensor:
        return self.indices[start_req_idx:end_req_idx]


    def get_slot_mapping(self, rev_indptr: torch.Tensor, per_token_offset: torch.Tensor) -> torch.Tensor:
        return (rev_indptr * self.max_seqlen + per_token_offset).to(torch.long)


    def get_block_table(self, start_req_idx: int, end_req_idx: int) -> torch.Tensor:
        return self.block_table[start_req_idx:end_req_idx]


    def get_whole_kv_cache(
        self, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer], self.v_cache[layer]


    def get_block_size(self) -> int:
        return self.block_size


    def update(self, input_req_idx: list[int], qo_indices: list[int]) -> None:
        r"""Update the KV cache with a new batch size.

        Parameters
        ----------
        batch_size : int
            The batch size to use for the cache.
        
        Notes
        -----
        This function (re)initializes the KV cache with the given batch size,
        and should not be called multiple times for a single batch.
        """
        logging.info(f"KVCache updated on device {self.device_id} with batch {qo_indices}")
        for idx, seqlen in zip(input_req_idx, torch.tensor(qo_indices).diff().tolist()):
            self.indices[idx] += seqlen
        self.last_key = torch.zeros(
            qo_indices[-1],
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=f"cuda:{self.device_id}",
        )
        self.last_value = torch.zeros(
            qo_indices[-1],
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=f"cuda:{self.device_id}",
        )


    def store_last_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_idx: int = 0,
        end_idx: int = -1
    ) -> None:
        r"""Store the last computed key and value tensors.

        Parameters
        ----------
        key : torch.Tensor
            The key tensor to store.
        value : torch.Tensor
            The value tensor to store.
        
        Notes
        -----
        This is a hack to allow FlashAttention prefill kernel to access the
        last computed key and value tensors.
        """
        assert self.last_key is not None and self.last_value is not None, "Cache not initialized. Call update() first."
        self.last_key[start_idx:end_idx].copy_(
            key.view(
                end_idx - start_idx,
                self.num_heads,
                self.head_dim,
            )
        )
        self.last_value[start_idx:end_idx].copy_(
            value.view(
                end_idx - start_idx,
                self.num_heads,
                self.head_dim,
            )
        )


    def get_last_kv(self, start_idx: int, end_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get the last computed key and value tensors for a specific layer.

        Parameters
        ----------
        device_id : int
            The device ID to use for the cache.
        layer : int
            The layer ID to get the cache for.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The key and value tensors for the specified layer.
        
        Notes
        -----
        This is a hack to allow FlashAttention prefill kernel to access the
        last computed key and value tensors.
        """
        return (
            self.last_key[start_idx:end_idx],
            self.last_value[start_idx:end_idx],
        )



    def put_batch(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        r"""Put a batch of key and value tensors into the KV cache.

        Parameters
        ----------
        layer : int
            The layer ID to put the cache for.
        key : torch.Tensor
            The key tensor to put into the cache.
            Shape: [batch_size, key_dim]
        value : torch.Tensor
            The value tensor to put into the cache.
            Shape: [batch_size, value_dim]
        slot_mapping:
            The mapping from the input tokens to the cache slots.
            Shape: [batch_size,]
        """
        if layer == 0:
            assert self.indices is not None, "Cache not initialized. Call update() first."
        reshape_and_cache(
            key.view(-1, self.num_heads, self.head_dim),
            value.view(-1, self.num_heads, self.head_dim),
            self.k_cache[layer],
            self.v_cache[layer],
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
            k_scale=self.unscaled,
            v_scale=self.unscaled,
        )


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
        self.k_datas = []
        self.v_datas = [] 
        for device_id in range(num_devices):
            k_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[device_id])
            v_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.available_devices[device_id])
            self.k_datas.append(k_data)
            self.v_datas.append(v_data)
            
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
        num_devices = pool.num_devices
        self.cache = {}
        self.kv_indptr_devices = [torch.tensor([0], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self.kv_indices_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self.kv_last_page_len_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self.rev_input_indptr_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self.per_token_offset_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]

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

    def get(self, device_id: int, layer: int, idx: int):
        kvcache = self.cache[idx]
        ki = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 0]
                self._pool.k_datas[device_id][layer, kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 0, :, :kvcache.last_page_offset, :]
                    self._pool.k_datas[device_id][layer,kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                # self._pool.kv_data[kvcache.indicies[:-1], 1]
                self._pool.v_datas[device_id][layer,kvcache.indicies[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indicies[-1], 1, :, :kvcache.last_page_offset, :]
                    self._pool.v_datas[device_id][layer, kvcache.indicies[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        return ki, vi

    def get_whole_kv_data(self, device_id, layer: int):
        return self._pool.k_datas[device_id][layer], self._pool.v_datas[device_id][layer]

    def get_whole_kv_data_all_layers(self, device_id):
        return self._pool.k_datas[device_id], self._pool.v_datas[device_id]

    def get_seqlen(self, idx: int):
        return self.cache[idx].seqlen
        
    def update(self, cumsum_input, input_req_idx, decode_batchsize, device_id):
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

        self.rev_input_indptr_devices[device_id] = rev_input_indptr_tensor.to(f"cuda:{device_id}")
        self.per_token_offset_devices[device_id] = per_token_offset_tensor.to(f"cuda:{device_id}")

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
        
        self.kv_indptr_devices[device_id] = kv_indptr_tensor.to(f"cuda:{device_id}")
        self.kv_indices_devices[device_id] = kv_indices_tensor.to(f"cuda:{device_id}")
        self.kv_last_page_len_devices[device_id] = kv_last_page_len_tensor.to(f"cuda:{device_id}")

    @property
    def page_size(self):
        return self._pool.page_size
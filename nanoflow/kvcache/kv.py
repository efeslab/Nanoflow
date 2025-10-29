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
from nanoflow.utils.prof_marker import prof_marker
import time

from .triton.kv_copy import copy_fa_nopage_kvcache, copy_torch_kvcache


class KVCacheNone:
    def __init__(self):
        self.name = 'No KV Cache'
        self.cache = {}
    
    def reset(self):
        self.cache = {}

    def put(self, layer, idx, key, value):
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)

    def update(self, cumsum_input, input_req_idx, decode_batchsize, use_cuda_graph=False):
        self.input_req_idx = input_req_idx
        return None

    def get_whole_kv_data(self, layer: int):
        return None, None
    
    def get_whole_kv_data_all_layers(self):
        return None, None
    
    def get_indices(self, layer, idx):
        return 0

class KVCacheTorch():
    def __init__(self, num_kv_heads, head_dim, tp_size=1):
        self.name = 'Torch KV Cache'
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
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
    
    def update(self, cumsum_input, input_req_idx, decode_batchsize, use_cuda_graph=False):
       assert not use_cuda_graph, "cuda graph mode of KVCacheTorch isn't implemented"
       self.input_req_idx = input_req_idx
       return None

    def get_indices(self, layer, idx):
        return self.cache_indices.get((layer, idx), 0)
    def get_whole_kv_data(self, layer: int):
        return None, None
    def get_whole_kv_data_all_layers(self):
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
        self.indices = torch.zeros((max_batch_size,), dtype=torch.int32, device=f"cuda:{self.device_id}")
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
      start_layer_idx: int,
      end_layer_idx: int,
      num_kv_heads: int,
      head_dim: int,
      capacity: int,
      page_size: int,
      tp_size: int,
      worker_device: str,
    ):
        torch.zeros(1, device=torch.device(worker_device))
        
        # # NOTE(Yilong): Assume underlying layout is HND.
        # assert num_kv_heads % len(device_list) == 0, "num_kv_heads must be divisible by num_devices"
            
        # Metadata is identical for all GPUs
        self._free = set(range(capacity))
        self._max_num_pages = capacity
        
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx
        self.num_layers = end_layer_idx - start_layer_idx
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size
        self.worker_device = worker_device
        # kv_data format is "HND"
        self.kv_shape = [self.num_layers, capacity, self.num_kv_heads, page_size, head_dim]
        self.k_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.worker_device)
        self.v_data = torch.empty(self.kv_shape, dtype=torch.float16, device=self.worker_device)

        # self.kv_shape_cpu = [num_layers, capacity, self.num_kv_heads, page_size, head_dim]
        # self.k_data_cpu = torch.empty(self.kv_shape_cpu, dtype=torch.float16, device="cpu")
        # self.v_data_cpu = torch.empty(self.kv_shape_cpu, dtype=torch.float16, device="cpu")
        
        print(f"KV Cache takes {self.k_data.numel() * 2 * 2 / 1024 / 1024 / 1024} GB")

    # @property
    def num_free_pages(self) -> int:
        return len(self._free)

    def alloc_page(self) -> int:
        assert len(self._free) > 0, "Out of memory"
        idx = self._free.pop()
        return idx

    def free_page(self,  idx: int):
        # assert 0 <= idx < self._buf[0].size(1), "Invalid page index"
        assert idx not in self._free
        self._free.add(idx)
    
    def reset(self):
        self._free = set(range(self.capacity))
        
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
    def indices(self) -> list[int]:
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
        self.device = pool.worker_device
        self.cache = {}
        self.num_indices = 0

        self.kv_indptr = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.kv_indices = torch.empty(self._pool._max_num_pages, dtype=torch.int32, device=self.device) # reserve a large enough space for all indices
        self.kv_last_page_len = torch.tensor([], dtype=torch.int32, device=self.device)
        self.rev_input_indptr = torch.tensor([], dtype=torch.int32, device=self.device)
        self.per_token_offset = torch.tensor([], dtype=torch.int32, device=self.device)

        self.double_buffer_enabled = False
        self.kv_indptr_tmp = torch.tensor([0], dtype=torch.int32, device="cpu")
        self.kv_indices_tmp = torch.empty(self._pool._max_num_pages, dtype=torch.int32, device="cpu")
        self.kv_last_page_len_tmp = torch.tensor([], dtype=torch.int32, device="cpu")
        self.rev_input_indptr_tmp = torch.tensor([], dtype=torch.int32, device="cpu")
        self.per_token_offset_tmp = torch.tensor([], dtype=torch.int32, device="cpu")

    def get_pool(self):
        return self._pool

    def reset(self):
        self._pool.reset()
        self.cache = {}
        self.kv_indptr = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.kv_indices = torch.empty(self._pool._max_num_pages, dtype=torch.int32, device=self.device) # reserve a large enough space for all indices
        self.kv_last_page_len = torch.tensor([], dtype=torch.int32, device=self.device)
        self.rev_input_indptr = torch.tensor([], dtype=torch.int32, device=self.device)
        self.per_token_offset = torch.tensor([], dtype=torch.int32, device=self.device)

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
        kvcache = self.cache[idx]
        layer_idx = layer - self._pool.start_layer_idx
        ki = torch.cat(
            [
                # self._pool.kv_data[kvcache.indices[:-1], 0]
                self._pool.k_data[layer_idx, kvcache.indices[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indices[-1], 0, :, :kvcache.last_page_offset, :]
                    self._pool.k_data[layer_idx,kvcache.indices[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                # self._pool.kv_data[kvcache.indices[:-1], 1]
                self._pool.v_data[layer_idx,kvcache.indices[:-1]]
                .permute(0, 2, 1, 3)
                .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim),
                (
                    # self._pool.kv_data[kvcache.indices[-1], 1, :, :kvcache.last_page_offset, :]
                    self._pool.v_data[layer_idx, kvcache.indices[-1], :, :kvcache.last_page_offset, :]
                    .permute(1, 0, 2)
                    .reshape(-1, self._pool.num_kv_heads, self._pool.head_dim)
                )
            ],
            dim=0,
        )
        return ki, vi

    def get_whole_kv_data(self, layer: int):
        layer_idx = layer - self._pool.start_layer_idx
        return self._pool.k_data[layer_idx], self._pool.v_data[layer_idx]

    def get_whole_kv_data_all_layers(self):
        return self._pool.k_data, self._pool.v_data

    def get_seqlen(self, idx: int):
        return self.cache[idx].seqlen

    def update_template(self, cumsum_input, input_req_idx, decode_batchsize, rev_input_indptr_ref, per_token_offset_ref, kv_indptr_ref, kv_indices_ref, kv_last_page_len_ref, use_cuda_graph=False):
        total_tokens = cumsum_input[-1]
        rev_input_indptr_tensor = torch.empty(total_tokens, dtype=torch.int32)
        per_token_offset_tensor = torch.empty(total_tokens, dtype=torch.int32)

        rev_input_indptr_tensor[0:decode_batchsize] = torch.arange(decode_batchsize, dtype=torch.int32)
        for temp_idx in range(decode_batchsize):
            global_req_idx = input_req_idx[temp_idx]
            self.pre_allocate(global_req_idx, 1)
            seq_len = self.get_seqlen(global_req_idx)
            per_token_offset_tensor[temp_idx] = seq_len - 1
            # print("decode batch, req_idx:", global_req_idx, "seq_len:", seq_len)

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
        if use_cuda_graph:
            rev_input_indptr_ref.copy_(rev_input_indptr_tensor)
            per_token_offset_ref.copy_(per_token_offset_tensor)
        else:
            rev_input_indptr_ref.resize_(rev_input_indptr_tensor.numel())
            rev_input_indptr_ref.copy_(rev_input_indptr_tensor)
            per_token_offset_ref.resize_(per_token_offset_tensor.numel())
            per_token_offset_ref.copy_(per_token_offset_tensor)

        num_reqs = len(input_req_idx)
        self.num_indices = sum([len(self.cache[req_idx].indices) for req_idx in input_req_idx])
        kv_indptr_tensor = torch.empty(num_reqs + 1, dtype=torch.int32)
        kv_indices_tensor = torch.empty(self.num_indices, dtype=torch.int32)
        kv_last_page_len_tensor = torch.empty(num_reqs, dtype=torch.int32)

        cur_offset = 0
        kv_indptr_tensor[0] = 0
        for i, global_req_idx in enumerate(input_req_idx):
            if global_req_idx not in self.cache:
                raise ValueError(f"Request {global_req_idx} not found in cache")
            kv = self.cache[global_req_idx]
            count = len(kv.indices)

            kv_indices_tensor[cur_offset : cur_offset + count] = torch.tensor(kv.indices, dtype=torch.int32)
            kv_indptr_tensor[i + 1] = cur_offset + count
            kv_last_page_len_tensor[i] = kv.last_page_offset
            cur_offset += count
        if self.num_indices > self._pool._max_num_pages:
            raise ValueError(f"Too many indices {self.num_indices} for the pool size {self._pool._max_num_pages}")
        if use_cuda_graph:
            kv_indptr_ref.copy_(kv_indptr_tensor)
            print(f"shape of kv_indices: {kv_indices_ref[:self.num_indices].shape}, kv_indices_tensor: {kv_indices_tensor.shape}")
            kv_indices_ref[:self.num_indices].copy_(kv_indices_tensor)
            kv_last_page_len_ref.copy_(kv_last_page_len_tensor)
        else:
            kv_indptr_ref.resize_(kv_indptr_tensor.numel())
            kv_indptr_ref.copy_(kv_indptr_tensor)
            kv_indices_ref[:self.num_indices].copy_(kv_indices_tensor)
            kv_last_page_len_ref.resize_(kv_last_page_len_tensor.numel())
            kv_last_page_len_ref.copy_(kv_last_page_len_tensor)

    def update_for_next_cycle(self, cumsum_input, input_req_idx, decode_batchsize, cuda_graph_enabled=False):
        self.update_template(cumsum_input, input_req_idx, decode_batchsize, self.rev_input_indptr_tmp, self.per_token_offset_tmp, self.kv_indptr_tmp, self.kv_indices_tmp, self.kv_last_page_len_tmp, cuda_graph_enabled)

    def update(self, cumsum_input, input_req_idx, decode_batchsize, double_buffer_enabled=False, cuda_graph_enabled=False):
        if double_buffer_enabled:
            self.rev_input_indptr.copy_(self.rev_input_indptr_tmp)
            self.per_token_offset.copy_(self.per_token_offset_tmp)
            self.kv_indptr.copy_(self.kv_indptr_tmp)
            self.kv_indices[:self.num_indices].copy_(self.kv_indices_tmp[:self.num_indices])
            self.kv_last_page_len.copy_(self.kv_last_page_len_tmp)
        else:
            self.update_template(cumsum_input, input_req_idx, decode_batchsize, self.rev_input_indptr, self.per_token_offset, self.kv_indptr, self.kv_indices, self.kv_last_page_len, cuda_graph_enabled)


    @property
    def page_size(self):
        return self._pool.page_size
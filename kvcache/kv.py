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
import time

from triton_ops.kv_copy import copy_kvcache


class KVCacheNone:
    def __init__(self):
        self.name = 'No KV Cache'
        self.cache = {}
    
    def put(self, layer, idx, key, value):
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)
    
    def get_whole_kv_data(self, device_id, layer: int):
        return None, None
    def get_whole_kv_data_all_layers(self, device_id):
        return None, None
    def get_indices(self, layer, idx):
        return True


class KVCacheTorch:
    def __init__(self):
        self.name = "Torch KV Cache"
        self.cache = {}
        self.cache_indices = {}
        self.hidden_dim = 1024
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
            self.cache_indices[(layer, idx)] = (key.shape[0], value.shape[0])
            # print(f"put the request {layer}, {idx}")

        else:
            old_key, old_value = self.cache[(layer, idx)]
            key_offset, value_offset = self.cache_indices[(layer, idx)]
            assert (
                key_offset + key.shape[0] <= self.max_size_per_request
            ), "Key size exceeds maximum size"
            assert (
                value_offset + value.shape[0] <= self.max_size_per_request
            ), "Value size exceeds maximum size"
            old_key[key_offset : key_offset + key.shape[0]] = key
            old_value[value_offset : value_offset + value.shape[0]] = value
            self.cache_indices[(layer, idx)] = (
                key_offset + key.shape[0],
                value_offset + value.shape[0],
            )

    def put_batch(
        self,
        layer: int,
        qo_indices: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rev_input_indices: torch.Tensor,
        per_token_offset: torch.Tensor,
    ) -> None:
        batch_size = qo_indices.shape[0] - 1
        key_ptr = [None for _ in range(batch_size)]
        value_ptr = [None for _ in range(batch_size)]
        seq_lens = qo_indices.diff()
        for i in range(batch_size):
            if (layer, i) not in self.cache:
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
                self.cache[(layer, i)] = (reserved_key, reserved_value)
                self.cache_indices[(layer, i)] = (0, 0)
            key_cache, value_cache = self.cache[(layer, i)]
            key_ptr[i] = key_cache
            value_ptr[i] = value_cache
            self.cache_indices[(layer, i)] = (
                self.cache_indices[(layer, i)][0] + seq_lens[i],
                self.cache_indices[(layer, i)][1] + seq_lens[i],
            )
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            seq_len = key.shape[0]
            for i in range(seq_len):
                input_idx = rev_input_indices[i]
                position = per_token_offset[i]
                logging.debug(f"id {i}: k_cache {hex(key_ptr[input_idx][position].data_ptr())}")
                logging.debug(f"id {i}: v_cache {hex(value_ptr[input_idx][position].data_ptr())}")
        key_ptr = [key_ptr[i].data_ptr() for i in range(len(key_ptr))]
        value_ptr = [value_ptr[i].data_ptr() for i in range(len(value_ptr))]

        key_ptr_tensor = torch.tensor(key_ptr, dtype=torch.uint64, device=key.device)
        value_ptr_tensor = torch.tensor(
            value_ptr, dtype=torch.uint64, device=value.device
        )
        copy_kvcache(
            key=key,
            value=value,
            key_cache_ptr=key_ptr_tensor,
            value_cache_ptr=value_ptr_tensor,
            rev_input_indices=rev_input_indices,
            per_token_offset=per_token_offset,
        )

    def get(self, layer, idx):
        # print(f"find the request {layer}, {idx}")
        if (layer, idx) in self.cache:
            # print(f"{layer, idx} is in kv cache.")
            reserved_key, reserved_value = self.cache[(layer, idx)]
            key_offset, value_offset = self.cache_indices[(layer, idx)]
            return reserved_key[:key_offset], reserved_value[:value_offset]
        # print(f"{layer, idx} is not in kv cache.")
        return None
    
    def get_indices(self, layer, idx):
        return self.cache_indices.get((layer, idx), None)
    def get_whole_kv_data(self, device_id, layer: int):
        return None, None
    def get_whole_kv_data_all_layers(self, device_id):
        return None, None


class KVCacheFANoPage:
    r"""KV cache for FlashAttention backend without page management.
    
    Note that this KV cache is somewhat static and only allocate
    memory at each batch initialization. Specifically, given a
    batch size and a maximum sequence length, the cache will allocate
    memory for the entire batch. This design is based on the current
    pipeline workflow, and should be further optimized if the memory
    overhead is unacceptable."""

    def __init__(
        self,
        *,
        device_id: int = 0,
        num_layers: int = 32,
        num_heads: int = 8,
        head_dim: int = 128,
        max_size_per_request: int = 2048
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
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_size_per_request = max_size_per_request
        self.last_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(self.num_layers)
        ]
    
    @property
    def initialized(self) -> bool:
        r"""Check if the KV cache is initialized.

        Returns
        -------
        bool
            True if the cache is initialized, False otherwise.
        """
        return self.k_cache is not None and self.v_cache is not None and self.batch_size is not None
    
    def get_indices(self, layer_id: int, request_id: int) -> int:
        r"""Get the offset of the KV cache for a request.

        Parameters
        ----------
        layer_id : int
            Unrelated.
        request_id : int
            The request ID to get the cache for.
        Returns
        -------
        int
            The offset / position of the request.
        """
        if self.indices is None:
            raise ValueError("Cache not initialized. Call update() first.")
        return self.indices[request_id].item() # type: ignore


    def update(self, batch_size: int) -> None:
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
        if self.k_cache is not None or self.v_cache is not None:
            logging.warning("Cache already exists, overwriting it.")
        self.batch_size = batch_size
        self.indices = torch.zeros((self.batch_size,), dtype=torch.int32, device=f"cuda:{self.device_id}")
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

    def get(self, layer_id: int, request_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get the KV cache for a specific layer and request.
        
        Parameters
        ----------
        layer_id : int
            The layer ID to get the cache for.
        request_id : int
            The request ID to get the cache for.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The key and value tensors for the specified layer and request.
        
        Notes
        -----
        Deprecated. Use `get_layer()` instead for better performance.
        """
        if self.k_cache is None or self.v_cache is None or self.batch_size is None:
            raise ValueError("Cache not initialized. Call update() first.")
        return (
            self.k_cache[layer_id][request_id],
            self.v_cache[layer_id][request_id],
        )

    def get_layer(self, batch_size: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get the KV cache for a specific layer.
        
        Parameters
        ----------
        batch_size : int
            The batch size to use for the cache.
        layer_id : int
            The layer ID to get the cache for.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The key and value tensors for the specified layer.
        """
        if self.k_cache is None or self.v_cache is None:
            raise ValueError("Cache not initialized. Call update() first.")
        if self.batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch. Expected {self.batch_size}, got {batch_size}."
            )
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def store_last_kv(self, key: torch.Tensor, value: torch.Tensor, device_id: int, layer: int) -> None:
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
        self.last_kv[layer] = (
            key.view(
                -1,
                self.num_heads,
                self.head_dim,
            ),
            value.view(
                -1,
                self.num_heads,
                self.head_dim,
            )
        )


    def get_last_kv(self, device_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        last_kv_layer = self.last_kv[layer]
        if last_kv_layer is None:
            raise ValueError("Last KV cache not initialized. Call store_last_kv() first.")
        return last_kv_layer

    def get_whole_indices(self) -> torch.Tensor:
        r"""Get the sequence length in the KV cache.
        
        Returns
        -------
        torch.Tensor
            The indices tensor for the KV cache.
        """
        if self.indices is None:
            raise ValueError("Cache not initialized. Call update() first.")
        return self.indices


    def get_whole_kv_data(
        self, device_id: int, layer: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r"""Get the KV cache for a specific layer.
        
        Parameters
        ----------
        device_id : int
            The device ID to use for the cache.
        layer : int
            The layer ID to get the cache for.
        
        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            The key and value tensors for the specified layer.
        
        Note
        ----
        If the cache is not initialized, this function will return None.
        """
        if self.k_cache is None or self.v_cache is None or self.batch_size is None:
            return None, None
        return self.k_cache[layer], self.v_cache[layer]


    def put_batch(
        self,
        layer: int,
        qo_indices: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rev_input_indices: torch.Tensor,
        per_token_offset: torch.Tensor,
    ) -> None:
        if self.k_cache is None or self.v_cache is None or self.batch_size is None:
            raise ValueError("Cache not initialized. Call update() first.")
        batch_size = qo_indices.shape[0] - 1
        assert batch_size == self.batch_size, "Batch size mismatch."
        if layer == 0:
            assert self.indices is not None, "Cache not initialized. Call update() first."
            self.indices += qo_indices.diff()
        for i in range(batch_size):
            # Get the start and end indices for the current request
            start_idx = qo_indices[i]
            end_idx = qo_indices[i + 1]

            # Get the key and value tensors for the current request
            key_tensor = key[start_idx:end_idx].view(
                -1,
                self.num_heads,
                self.head_dim,
            )
            value_tensor = value[start_idx:end_idx].view(
                -1,
                self.num_heads,
                self.head_dim,
            )

            # Get the offset for the current request
            offset = self.indices[i] # type: ignore

            # Update the cache with the new key and value tensors
            self.k_cache[layer][i][offset - key_tensor.shape[0] : offset] = key_tensor
            self.v_cache[layer][i][offset - value_tensor.shape[0] : offset] = value_tensor

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
        self._kv_indptr_devices = [torch.tensor([0], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self._kv_indices_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]
        self._kv_last_page_len_devices = [torch.tensor([], dtype=torch.int32, device=self._pool.k_datas[device_id].device) for device_id in range(num_devices)]

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

    def update(self, device_id):
        # Here we do not materialize data into specific devices,
        # for distributed assignment.
        kv_indptr_list = [0]
        kv_indices_list = []
        kv_last_page_len_list = []
        for _, kv in self.cache.items():
            kv_indptr_list.append(kv_indptr_list[-1] + len(kv.indicies))
            kv_indices_list.extend(kv.indicies)
            kv_last_page_len_list.append(kv.last_page_offset)
        self._kv_indptr_devices[device_id] = torch.tensor(kv_indptr_list, dtype=torch.int32, device=f"cuda:{device_id}")
        self._kv_indices_devices[device_id] = torch.tensor(kv_indices_list, dtype=torch.int32, device=f"cuda:{device_id}")
        self._kv_last_page_len_devices[device_id] = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=f"cuda:{device_id}")

        return self._kv_indptr_devices[device_id], self._kv_indices_devices[device_id], self._kv_last_page_len_devices[device_id]
        

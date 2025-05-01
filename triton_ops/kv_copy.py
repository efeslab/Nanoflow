"""
Author: Yi Pan <conlesspan@outlook.com>
Date: 2025-04-30
Description: Triton kernels and binings for KV cache copy.
"""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
from typing import Optional


@triton.jit
def _copy_kvcache_kernel(
    n,
    d,
    k_ptr,  # [n, d], batched key tensor
    k_cache_ptr,  # [b,], key cache of each request
    v_ptr,  # [n, d], batched value tensor
    v_cache_ptr,  # [b,], value cache of each request
    rev_input_indices_ptr,  # [n,]
    per_token_offset_ptr,  # [n,]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    input_idx = tl.load(rev_input_indices_ptr + pid)
    position = tl.load(per_token_offset_ptr + pid)
    k_row = k_ptr + position * d
    v_row = v_ptr + position * d
    k_cache_row = (tl.load(k_cache_ptr + input_idx) + d * position).to(tl.pointer_type(tl.float16))
    v_cache_row = (tl.load(v_cache_ptr + input_idx) + d * position).to(tl.pointer_type(tl.float16))

    for i in range(0, d, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < d
        k = tl.load(k_row + offsets, mask=mask)
        v = tl.load(v_row + offsets, mask=mask)
        tl.store(k_cache_row + offsets, k, mask=mask)
        tl.store(v_cache_row + offsets, v, mask=mask)


def copy_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache_ptr: torch.Tensor,
    value_cache_ptr: torch.Tensor,
    rev_input_indices: torch.Tensor,
    per_token_offset: torch.Tensor,
) -> None:
    n, dim = key.shape
    block_size = triton.next_power_of_2(dim)

    _copy_kvcache_kernel[(n,)](
        n,
        dim,
        key,
        key_cache_ptr,
        value,
        value_cache_ptr,
        rev_input_indices,
        per_token_offset,
        BLOCK_SIZE=block_size,
    )

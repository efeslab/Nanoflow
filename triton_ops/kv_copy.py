"""
Author: Yi Pan <conlesspan@outlook.com>
Date: 2025-04-30
Description: Triton kernels and binings for KV cache copy.
"""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def _copy_torch_kvcache_kernel(
    k_ptr,  # [n, d], batched key tensor
    k_cache_ptr,  # [b,], key cache of each request
    v_ptr,  # [n, d], batched value tensor
    v_cache_ptr,  # [b,], value cache of each request
    rev_input_indices_ptr,  # [n,]
    per_token_offset_ptr,  # [n,]
    # tensor dimensions
    seqlen,
    hidden_dim,
    # strides
    stride_k_seqlen,
    stride_k_dim,
    stride_v_seqlen,
    stride_v_dim,
    stride_k_cache_batch,
    stride_v_cache_batch,
    stride_rev_input_indices,
    stride_per_token_offset,
    # meta parameters
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    input_idx = tl.load(rev_input_indices_ptr + pid * stride_rev_input_indices)
    position = tl.load(per_token_offset_ptr + pid * stride_per_token_offset)
    k_row = k_ptr + pid * stride_k_seqlen
    v_row = v_ptr + pid * stride_v_seqlen
    k_cache_row = tl.load(k_cache_ptr + input_idx * stride_k_cache_batch).to(k_ptr.dtype) + position * hidden_dim
    v_cache_row = tl.load(v_cache_ptr + input_idx * stride_v_cache_batch).to(v_ptr.dtype) + position * hidden_dim

    # print("k_row: ", k_row)
    # print("k_cache_row: ", k_row)
    for i in range(0, hidden_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        k = tl.load(k_row + offsets * stride_k_dim, mask=mask)
        v = tl.load(v_row + offsets * stride_v_dim, mask=mask)
        tl.store(k_cache_row + offsets, k, mask=mask)
        tl.store(v_cache_row + offsets, v, mask=mask)

@triton.jit
def _copy_fa_nopage_kvcache_kernel(
    k_ptr,  # [n, d], batched key tensor
    v_ptr,  # [n, d], batched value tensor
    k_cache_ptr,  # [b, max_seqlen, d], key cache of each request
    v_cache_ptr,  # [b, max_seqlen, d], value cache of each request
    rev_input_indices_ptr,  # [n,]
    per_token_offset_ptr,  # [n,]
    # tensor dimensions
    seqlen,
    hidden_dim,
    # strides
    stride_k_seqlen,
    stride_k_dim,
    stride_v_seqlen,
    stride_v_dim,
    stride_k_cache_batch,
    stride_k_cache_seqlen,
    stride_k_cache_dim,
    stride_v_cache_batch,
    stride_v_cache_seqlen,
    stride_v_cache_dim,
    stride_rev_input_indices,
    stride_per_token_offset,
    # meta parameters
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    input_idx = tl.load(rev_input_indices_ptr + pid * stride_rev_input_indices)
    position = tl.load(per_token_offset_ptr + pid * stride_per_token_offset)
    k_row = k_ptr + pid * stride_k_seqlen
    v_row = v_ptr + pid * stride_v_seqlen
    k_cache_row = k_cache_ptr + input_idx * stride_k_cache_batch + position * stride_k_cache_seqlen
    v_cache_row = v_cache_ptr + input_idx * stride_v_cache_batch + position * stride_v_cache_seqlen

    for i in range(0, hidden_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        k = tl.load(k_row + offsets * stride_k_dim, mask=mask)
        v = tl.load(v_row + offsets * stride_v_dim, mask=mask)
        tl.store(k_cache_row + offsets * stride_k_cache_dim, k, mask=mask)
        tl.store(v_cache_row + offsets * stride_v_cache_dim, v, mask=mask)
    

def copy_torch_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache_ptr: torch.Tensor,
    value_cache_ptr: torch.Tensor,
    rev_input_indices: torch.Tensor,
    per_token_offset: torch.Tensor,
) -> None:
    r"""Copy key and value tensors to the KV cache.
    
    Parameters
    ----------
    key : torch.Tensor
        The key tensor to be copied.
        Shape: [n, d]
    value : torch.Tensor
        The value tensor to be copied.
        Shape: [n, d]
    key_cache_ptr : torch.Tensor
        The pointer to the key cache.
        Shape: [b,]
    value_cache_ptr : torch.Tensor
        The pointer to the value cache.
        Shape: [b,]
    rev_input_indices : torch.Tensor
        The reverse input indices.
        Shape: [n,]
    per_token_offset : torch.Tensor
        The per token offset.
        Shape: [n,]
    """
    n, dim = key.shape
    block_size = triton.next_power_of_2(dim)

    _copy_torch_kvcache_kernel[(n,)](
        k_ptr=key,
        k_cache_ptr=key_cache_ptr,
        v_ptr=value,
        v_cache_ptr=value_cache_ptr,
        rev_input_indices_ptr=rev_input_indices,
        per_token_offset_ptr=per_token_offset,
        seqlen=n,
        hidden_dim=dim,
        stride_k_seqlen=key.stride(0),
        stride_k_dim=key.stride(1),
        stride_v_seqlen=value.stride(0),
        stride_v_dim=value.stride(1),
        stride_k_cache_batch=key_cache_ptr.stride(0),
        stride_v_cache_batch=value_cache_ptr.stride(0),
        stride_rev_input_indices=rev_input_indices.stride(0),
        stride_per_token_offset=per_token_offset.stride(0),
        BLOCK_SIZE=block_size,
    )
        

def copy_fa_nopage_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rev_input_indices: torch.Tensor,
    per_token_offset: torch.Tensor,
) -> None:
    r"""Copy key and value tensors to the KV cache.

    Parameters
    ----------
    key : torch.Tensor
        The key tensor to be copied.
        Shape: [n, d]
    value : torch.Tensor
        The value tensor to be copied.
        Shape: [n, d]
    key_cache : torch.Tensor
        The key cache tensor.
        Shape: [b, max_seqlen, d]
    value_cache : torch.Tensor
        The value cache tensor.
        Shape: [b, max_seqlen, d]
    rev_input_indices : torch.Tensor
        The reverse input indices.
        Shape: [n,]
    per_token_offset : torch.Tensor
        The per token offset.
        Shape: [n,]
    """
    n, dim = key.shape
    block_size = triton.next_power_of_2(dim)

    _copy_fa_nopage_kvcache_kernel[(n,)](
        k_ptr=key,
        v_ptr=value,
        k_cache_ptr=key_cache,
        v_cache_ptr=value_cache,
        rev_input_indices_ptr=rev_input_indices,
        per_token_offset_ptr=per_token_offset,
        seqlen=n,
        hidden_dim=dim,
        stride_k_seqlen=key.stride(0),
        stride_k_dim=key.stride(1),
        stride_v_seqlen=value.stride(0),
        stride_v_dim=value.stride(1),
        stride_k_cache_batch=key_cache.stride(0),
        stride_k_cache_seqlen=key_cache.stride(1),
        stride_k_cache_dim=key_cache.stride(2),
        stride_v_cache_batch=value_cache.stride(0),
        stride_v_cache_seqlen=value_cache.stride(1),
        stride_v_cache_dim=value_cache.stride(2),
        stride_rev_input_indices=rev_input_indices.stride(0),
        stride_per_token_offset=per_token_offset.stride(0),
        BLOCK_SIZE=block_size,
    )
import logging
import torch

from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCachevLLM
from vllm._custom_ops import paged_attention_rocm

from utils.util_functions import tensor_offset_to_req_idx

_PARTITION_SIZE_ROCM = 256

class DecPagedAttnBatchedImpl(OperationImpl):
    r"""FlashAttention implementation of the DecAttn operator.

    This implementation uses the flash_attn library to perform the decoding attention.
    """

    category_tag = "vllm"  # type: ignore[assignment]

    def __init__(
        self, op_base: "DecPagedAttn", stream: torch.cuda.Stream, device_id: int
    ) -> None:
        r"""Initialize the DecAttn operator.

        Parameters
        ----------
        op_base : DecAttn
            The base operator it implements.
        device_id : int
            The device ID.
        """
        self.op_base: DecPagedAttn
        super().__init__(op_base, stream, device_id)  # type: ignore
        self.device_id = device_id
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.scale = 1.0 / (self.head_dim**0.5)
        self.k_scale = torch.tensor(1.0, dtype=torch.float32)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32)

    def run(
        self,
        layer: int,
        qo_indicies: torch.Tensor,
        Q: torch.Tensor,
        kv_tuple: None,
        KVCache: KVCachevLLM,
        output: torch.Tensor,
    ) -> None:
        r"""Run the DecAttn operator.

        Parameters
        ----------
        layer : int
            The layer index.g
        qo_indicies : torch.Tensor
            The indices that mark the start and end of the query slices for each batch request.
        Q : torch.Tensor
            The query tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        kv_tuple : tuple[torch.Tensor | None, torch.Tensor | None]
            Unrelated.
        KVCache : KVCachevLLM
            The KV cache in vLLM layout. Note that here we use the last_kv it stores.
        output : torch.Tensor
            The output tensor.
            Shape: [n_total, num_qo_heads * head_dim]

        Notes
        -----
        Deprecated. Use `run` instead.
        """
        if Q.shape[0] == 0:
            return

        with torch.cuda.stream(self.stream):
            q = Q.view(-1, self.num_qo_heads, self.head_dim)
            k_cache, v_cache = KVCache.get_whole_kv_cache(layer)
            kv_seqlens = self.op_base.kv_seqlens
            assert (
                q.shape[0] == kv_seqlens.shape[0]
            ), f"q.shape {q.shape} mismatch with cache_seqlens.shape {kv_seqlens.shape}"
            max_seq_len = self.op_base.max_seqlen
            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            tmp_output = torch.empty(
                size=(q.shape[0], self.num_qo_heads, max_num_partitions, self.head_dim),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(q.shape[0], self.num_qo_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            block_tables = self.op_base.block_tables

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("--------------------------------")
                logging.debug(f"output.shape: {output.shape}\nexp_sums.shape: {exp_sums.shape}\nmax_logits.shape: {max_logits.shape}\ntmp_output.shape: {tmp_output.shape}\nq.shape: {q.shape}\nk_cache.shape: {k_cache.shape}\nv_cache.shape: {v_cache.shape}\nnum_kv_heads: {self.num_kv_heads}\nscale: {self.scale}\nblock_tables: {block_tables}\nseq_lens: {kv_seqlens}\nblock_size: {KVCache.get_block_size()}\nmax_seq_len: {max_seq_len}\nalibi_slopes: {None}\nkv_cache_dtype: auto\nk_scale: {self.k_scale}\nv_scale: {self.v_scale}\nfp8_out_scale: {None}\npartition_size: {_PARTITION_SIZE_ROCM}")
                logging.debug("--------------------------------")
            paged_attention_rocm(
                out=output,
                exp_sum=exp_sums,
                max_logits=max_logits,
                tmp_out=tmp_output,
                query=q,
                key_cache=k_cache,
                value_cache=v_cache,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=block_tables,
                seq_lens=kv_seqlens,
                block_size=KVCache.get_block_size(),
                max_seq_len=max_seq_len,
                alibi_slopes=None,
                kv_cache_dtype="auto",
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                fp8_out_scale=None,
                partition_size=_PARTITION_SIZE_ROCM
            )


class DecPagedAttn(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {"Q": IOWrapper(self, "Q", device).is_input()}
        self.outputs = {"output": IOWrapper(self, "output", device).is_output()}
        self.externals: dict[str, KVCachevLLM] = {}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = DecPagedAttn_Layer
        self.batched_decode_wrapper = None
        # Initialize metadata
        self.qo_indicies: torch.Tensor
        self.kv_seqlens: torch.Tensor 
        self.max_seqlen: int
        self.block_tables: torch.Tensor

    def init_impl_map(self):
        self.add_impl(DecPagedAttnBatchedImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size: int = 1):
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.inputs["Q"].init_shape(
            (0, num_qo_heads * head_dim)
        )
        self.outputs["output"].init_shape(
            (0, num_qo_heads * head_dim)
        )

    def update(self, cumsum_input: list[int], device: str):
        self.qo_indicies = torch.tensor(
            cumsum_input, dtype=torch.int32, device=device
        )
        io_device = self.inputs["Q"]
        self.start_req_idx = tensor_offset_to_req_idx(
            self.qo_indicies.tolist(), io_device.tensor_offset
        )
        self.end_req_idx = tensor_offset_to_req_idx(
            self.qo_indicies.tolist(), io_device.tensor_offset + io_device.batch_size
        )
        self.qo_indicies = (
            self.qo_indicies[self.start_req_idx : self.end_req_idx + 1] - io_device.tensor_offset
        )
        self.qo_seqlens = self.qo_indicies.diff()
        self.kv_seqlens = self.externals["KVCache"].get_indices(self.start_req_idx, self.end_req_idx)
        self.max_seqlen = int(self.kv_seqlens.max().item()) if self.kv_seqlens.numel() > 0 else 0
        self.block_tables = self.externals["KVCache"].get_block_table(self.start_req_idx, self.end_req_idx)


    def profile(self):
        pass


class DecPagedAttn_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(
            self.layer,
            self.parent.qo_indicies,
            Q,
            None,
            self.parent.externals["KVCache"],
            self.outputs["output"].tensor,
        )

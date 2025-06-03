import logging
import torch

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCachevLLM
from vllm._custom_ops import paged_attention_rocm

from utils.help_functions import tensor_offset_to_req_idx  # type: ignore[import]

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
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"Q": IOWrapper(self, "Q")}
        self.outputs = {"output": IOWrapper(self, "output")}
        self.externals = {"KVCache": None}
        self.impl_map = {}
        self.init_impl_map()
        self.batched_decode_wrapper = None
        self.op_device = DecPagedAttn_Device

    def init_impl_map(self):
        self.add_impl(DecPagedAttnBatchedImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size: int = 1):
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        for op_device in self.children:
            op_device.setShapeForIOWrappers()

    def update(self, cumsum_input: list[int], device_id: int):
        self.qo_indicies = torch.tensor(
            cumsum_input, dtype=torch.int32, device=f"cuda:{device_id}"
        )
        io_device = self.children[device_id].inputs["Q"]
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
        self.max_seqlen = self.kv_seqlens.max().item() if self.kv_seqlens.numel() > 0 else 0
        self.block_tables = self.externals["KVCache"].get_block_table(self.start_req_idx, self.end_req_idx)


    def profile(self):
        pass


class DecPagedAttn_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = DecPagedAttn_Layer

    def setShapeForIOWrappers(self):
        self.inputs["Q"].init_shape(
            (0, self.parent.num_qo_heads * self.parent.head_dim)
        )
        self.outputs["output"].init_shape(
            (0, self.parent.num_qo_heads * self.parent.head_dim)
        )


class DecPagedAttn_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device=op_device)

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(
            self.layer,
            self.parent.parent.qo_indicies,
            Q,
            None,
            self.parent.externals["KVCache"],
            self.outputs["output"].tensor,
        )

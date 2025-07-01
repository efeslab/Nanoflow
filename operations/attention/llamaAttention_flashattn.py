import logging
import torch

from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheBatched, KVCachevLLM
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from utils.util_functions import tensor_offset_to_req_idx  # type: ignore[import]


class DecAttnFABatchedImpl(OperationImpl):
    r"""FlashAttention implementation of the DecAttn operator.

    This implementation uses the flash_attn library to perform the decoding attention.
    """

    category_tag = "flash_attn_batched"

    def __init__(
        self, op_base: "DecAttnFA", stream: torch.cuda.Stream, device_id: int
    ) -> None:
        r"""Initialize the DecAttn operator.

        Parameters
        ----------
        op_base : DecAttn
            The base operator it implements.
        device_id : int
            The device ID.
        """
        self.op_base: DecAttnFA
        super().__init__(op_base, stream, device_id)  # type: ignore
        self.device_id = device_id
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.scale = 1.0 / (self.head_dim**0.5)

    def run(
        self,
        layer: int,
        qo_indices: torch.Tensor,
        Q: torch.Tensor,
        kv_tuple: None,
        KVCache: KVCacheBatched,
        output: torch.Tensor,
    ) -> None:
        r"""Run the DecAttn operator.

        Parameters
        ----------
        layer : int
            The layer index.g
        qo_indices : torch.Tensor
            The indices that mark the start and end of the query slices for each batch request.
        Q : torch.Tensor
            The query tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        kv_tuple : tuple[torch.Tensor | None, torch.Tensor | None]
            Unrelated.
        KVCache : KVCacheBatched
            The KV cache in batched layout. Note that here we use the last_kv it stores.
            Shape: list of [batch_size, max_seq_len, num_kv_heads, head_dim]
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
            q = Q.view(-1, 1, self.num_qo_heads, self.head_dim)
            k_cache, v_cache = KVCache.get_kv_data(
                layer,
                self.op_base.start_req_idx,
                self.op_base.end_req_idx,
            )
            assert k_cache is not None and v_cache is not None
            cache_seqlens = KVCache.get_indices(
                self.op_base.start_req_idx, self.op_base.end_req_idx
            )
            assert (
                q.shape[0] == cache_seqlens.shape[0]
            ), f"q.shape {q.shape} mismatch with cache_seqlens.shape {cache_seqlens.shape}"
            o = flash_attn_with_kvcache(  # type: ignore
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
            )
            assert isinstance(o, torch.Tensor)
            o = o.view(-1, self.num_qo_heads * self.head_dim)
            output.copy_(o)


class DecAttnFA(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {"Q": IOWrapper(self, "Q", device).is_input()}
        self.outputs = {"output": IOWrapper(self, "output", device).is_output()}
        self.externals: dict[str, KVCacheBatched] = {}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = DecAttnFA_Layer
        self.batched_decode_wrapper = None
        self.qo_indices: torch.Tensor
        self.start_req_idx: int
        self.end_req_idx: int

    def init_impl_map(self):
        self.add_impl(DecAttnFABatchedImpl)

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

    def update(self, cumsum_input: list[int], device_id: int):
        self.qo_indices = torch.tensor(
            cumsum_input, dtype=torch.int32, device=f"cuda:{device_id}"
        )
        io_device = self.children[device_id].inputs["Q"]
        self.start_req_idx = tensor_offset_to_req_idx(
            self.qo_indices.tolist(), io_device.tensor_offset
        )
        self.end_req_idx = tensor_offset_to_req_idx(
            self.qo_indices.tolist(), io_device.tensor_offset + io_device.batch_size
        )
        self.qo_indices = (
            self.qo_indices[self.start_req_idx : self.end_req_idx + 1]
            - io_device.tensor_offset
        )

    def profile(self):
        pass


class DecAttnFA_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(
            self.layer,
            self.parent.qo_indices,
            Q,
            None,
            self.parent.externals["KVCache"],
            self.outputs["output"].tensor,
        )


class PFAttnFABatchedImpl(OperationImpl):
    r"""FlashAttention implementation of the PFAttn operator.

    This implementation uses the flash_attn library to perform the prefill attention.
    """

    category_tag = "flash_attn_batched"

    def __init__(self, op_base: "PFAttnFA", stream: torch.cuda.Stream, device_id: int):
        r"""Initialize the PFAttn operator.

        Parameters
        ----------
        op_base : PFAttn
            The base operator it implements.
        device_id : int
            The device ID.
        """
        self.op_base: PFAttnFA
        super().__init__(op_base, stream, device_id)  # type: ignore
        self.device_id = device_id
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.scale = 1.0 / (self.head_dim**0.5)

    def run(
        self,
        layer: int,
        qo_indices: torch.Tensor,
        Q: torch.Tensor,
        kv_tuple: None,
        KVCache: KVCacheBatched | KVCachevLLM,
        output: torch.Tensor,
    ):
        r"""Run the PFAttn operator.

        Parameters
        ----------
        layer : int
            The layer index.g
        qo_indices : torch.Tensor
            The indices that mark the start and end of the query slices for each batch request.
        Q : torch.Tensor
            The query tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        kv_tuple : tuple[torch.Tensor | None, torch.Tensor | None]
            Unrelated.
        KVCache : KVCacheBatched | KVCachevLLM
            The KV cache in batched layout. Note that here we use the last_kv it stores.
            Shape: list of [batch_size, max_seq_len, num_kv_heads, head_dim]
        output : torch.Tensor
            The output tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        """
        if Q.shape[0] == 0:
            return
        with torch.cuda.stream(self.stream):
            q = Q.view(-1, self.num_qo_heads, self.head_dim)
            k, v = KVCache.get_last_kv(
                self.op_base.io_device.tensor_offset,
                self.op_base.io_device.tensor_offset
                + self.op_base.io_device.batch_size,
            )
            o = flash_attn_varlen_func(  # type: ignore
                q,
                k,
                v,
                cu_seqlens_q=qo_indices,
                cu_seqlens_k=qo_indices,
                max_seqlen_q=self.op_base.max_seqlen_q,  # type: ignore
                max_seqlen_k=self.op_base.max_seqlen_k,  # type: ignore
                softmax_scale=self.scale,
                causal=True,
            )
            assert isinstance(o, torch.Tensor)
            o = o.view(-1, self.num_qo_heads * self.head_dim)
            output.copy_(o)


class PFAttnFA(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "Q": IOWrapper(self, "Q", device).is_input(),
        }
        self.outputs = {"output": IOWrapper(self, "output", device).is_output()}
        # Note: for consistency with other operators (like RopeAppend), we expect the external KV cache to be
        # available as "KVCache". If needed, you can change the key name.
        self.externals: dict[str, KVCachevLLM | KVCacheBatched] = {}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = PFAttnFA_Layer
        self.qo_indices: torch.Tensor
        self.max_seqlen_q: int
        self.max_seqlen_k: int
        self.io_device: IOWrapper
        self.start_req_idx: int
        self.end_req_idx: int

    def init_impl_map(self):
        self.add_impl(PFAttnFABatchedImpl)

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

    def update(self, cumsum_input: list[int], device_id: int):
        self.qo_indices = torch.tensor(
            cumsum_input, dtype=torch.int32, device=f"cuda:{device_id}"
        )
        seq_lens = self.qo_indices.diff()
        self.max_seqlen_q = int(torch.max(seq_lens).item())
        self.max_seqlen_k = self.max_seqlen_q
        self.io_device = self.children[device_id].inputs["Q"]
        self.start_req_idx = tensor_offset_to_req_idx(
            self.qo_indices.tolist(), self.io_device.tensor_offset
        )
        self.end_req_idx = tensor_offset_to_req_idx(
            self.qo_indices.tolist(),
            self.io_device.tensor_offset + self.io_device.batch_size,
        )
        self.qo_indices = (
            self.qo_indices[self.start_req_idx : self.end_req_idx + 1]
            - self.qo_indices[self.start_req_idx]
        )

    def profile(self):
        raise NotImplementedError("Profile not implemented for PFAttnFA")


class PFAttnFA_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(
            self.layer,
            self.parent.qo_indices,
            Q,
            None,
            self.parent.externals["KVCache"],
            self.outputs["output"].tensor,
        )

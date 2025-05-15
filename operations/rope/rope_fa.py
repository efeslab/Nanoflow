import logging
import torch

from operations.rope.help_functions import apply_rope  # type: ignore[import]
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheFANoPage
from triton_ops.rope import apply_rotary_emb
from utils.prof_marker import prof_marker
from utils.help_functions import tensor_offset_to_req_idx


class RopeAppendFANoPageImpl(OperationImpl):
    category_tag = "flash_attn_no_page"  # type: ignore[assignment]

    def __init__(
        self, op_base: "RopeAppendFA", stream: torch.cuda.Stream, device_id: int
    ):
        super().__init__(op_base, stream, device_id)
        self.rope_type = op_base.rope_type
        self.device_id = device_id
        if self.rope_type == "llama3":
            self.base = 500000.0
            self.rotary_dim = 128
        self.theta = op_base.theta
        self.original_max_position_embeddings = op_base.original_max_position_embeddings
        self.low_freq_factor = op_base.low_freq_factor
        self.high_freq_factor = op_base.high_freq_factor
        self.factor = op_base.factor
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.cache = self._compute_cos_sin_cache().to(
            dtype=torch.float16, device=f"cuda:{device_id}"
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.original_max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache.device != query.device:
            self.cache = self.cache.to(query.device)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"device {query.device} query shape: {query.shape}\nquery: {query}")
        
        positions = self.op_base.per_token_offset  # type: ignore
        assert isinstance(positions, torch.Tensor)
        num_tokens = positions.shape[0]
        cos_sin = self.cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query_rot = query[..., : self.rotary_dim]
        # query_pass = query[..., self.rotary_dim:]
        apply_rotary_emb(
            query_rot,
            cos,
            sin,
            inplace=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # query_rot = _apply_rotary_emb_torch(query_rot, cos, sin)
        # query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        query = query.view(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key_rot = key[..., : self.rotary_dim]
        # key_pass = key[..., self.rotary_dim:]
        apply_rotary_emb(
            key_rot,
            cos,
            sin,
            inplace=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # key_rot = _apply_rotary_emb_torch(key_rot, cos, sin)
        # key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        key = key.view(key_shape)

        # if logging.getLogger().isEnabledFor(logging.DEBUG):
        #     logging.debug(f"device {query_rot.device} query_rot shape: {query_rot.shape}\nquery_rot: {query_rot}")
        #     logging.debug(f"device {key_rot.device} key_rot shape: {key_rot.shape}\nkey_rot: {key_rot}")

        return query, key

    def run(
        self,
        layer: int,
        kqv: torch.Tensor,
        KVCache: KVCacheFANoPage,
        output: torch.Tensor,
        offset: int = 0,
    ):
        # Determine the number of elements for each slice.
        layout_strides = [
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_qo_heads * self.head_dim,
        ]
        # Split kqv into key, query, and value (here assumed to be in the order: k, q, v).
        # print("kqv shape:", kqv.shape)
        k, v, q = torch.split(kqv, layout_strides, dim=1)
        k = k.contiguous()
        v = v.contiguous()
        q = q.contiguous()

        # Process each batch element.
        with prof_marker("RopeAppendTorch: Rope"):
            q, k = self.forward(q, k, self.op_base.qo_indicies, self.op_base.max_seqlen)

        with prof_marker("RopeAppendTorch: KVCachePutBatch"):
            KVCache.put_batch(
                layer,
                self.op_base.qo_indicies,
                k,
                v,
                self.op_base.rev_input_indptr,
                self.op_base.per_token_offset,
            )
        with prof_marker("RopeAppendTorch: FinalCopy"):
            output.copy_(q)
            KVCache.store_last_kv(k, v, self.device_id, layer)


class RopeAppendFA(Operations):
    def __init__(
        self,
        name: str,
        rope_type: str = "llama3",
        theta: float = 10000.0,
        factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position_embeddings: int = 8192,
    ):
        """
        Args:
            name (str): The name of this operator.
            rope_type (str): The type of RoPE implementation to use. For llama3, pass "llama3".
            theta (float): The base used to compute the inverse frequency (typically set from config.rope_theta).
            factor (float): Scaling factor used in llama3.
            low_freq_factor (float): Lower bound frequency factor (llama3).
            high_freq_factor (float): Upper bound frequency factor (llama3).
            original_max_position_embeddings (int): The original maximum context length used in pretraining.
        """
        super().__init__(name)
        self.inputs = {"kqv": IOWrapper(self, "kqv")}
        self.outputs = {"q": IOWrapper(self, "q")}
        self.externals = {"KVCache": None}

        # Save RoPE configuration.
        self.rope_type = rope_type
        self.theta = theta  # typically config.rope_theta
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        self.impl_map = {}
        self.init_impl_map()
        self.op_device = RopeAppendFA_Device

    def init_impl_map(self):
        self.add_impl(RopeAppendFANoPageImpl)  # type: ignore

    def setShape(self, num_kv_heads: int, num_qo_heads: int, head_dim: int, tp_size: int = 1) -> None:  # type: ignore
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.updateChildrenIOShape()

    def update(self, qo_indicies: list[int], decode_batchsize: int, device_id: int):
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.update(qo_indicies, decode_batchsize, device_id)
        else:
            """Stores the starting indices for the query/key segments."""
            self.qo_indicies = torch.tensor(qo_indicies)
            self.seqlens = self.qo_indicies.diff()
            self.max_seqlen = self.seqlens.max().item()
            self.per_token_offset = torch.zeros(int(self.qo_indicies[-1].item()))
            self.rev_input_indptr = torch.zeros(int(self.qo_indicies[-1].item()))
            indices = self.externals["KVCache"].get_whole_indices()
            for i, seqlen in enumerate(self.seqlens.tolist()):
                self.per_token_offset[
                    self.qo_indicies[i] : self.qo_indicies[i] + seqlen
                ] = torch.arange(seqlen) + indices[i].cpu()
                self.rev_input_indptr[
                    self.qo_indicies[i] : self.qo_indicies[i] + seqlen
                ] = i
            self.qo_indicies = self.qo_indicies.to(
                dtype=torch.int32, device=f"cuda:{device_id}"
            )
            self.per_token_offset = self.per_token_offset.to(
                dtype=torch.int32, device=f"cuda:{device_id}"
            )
            self.rev_input_indptr = self.rev_input_indptr.to(
                dtype=torch.int32, device=f"cuda:{device_id}"
            )

    def copy_nano(self, index: int):
        new_op = RopeAppendFA(
            f"{self.name}{index}",
            self.rope_type,
            self.theta,
            self.factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.original_max_position_embeddings,
        )
        new_op.externals = self.externals
        new_op.expand_all_gpu_and_layers(len(self.device_list), 32)  # type: ignore
        new_op.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        new_op.set_stream(self.stream)  # type: ignore
        self.nano_ops.append(new_op)  # type: ignore

        return new_op

    def profile(self) -> None:
        raise NotImplementedError("Profile method is not implemented for RopeAppendFA.")


class RopeAppendFA_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = RopeAppendFA_Layer

    def setShapeForIOWrappers(self):
        # The input tensor "kqv" is assumed to have a flattened layout:
        # [batch_size, (num_qo_heads + 2 * num_kv_heads) * head_dim]
        self.inputs["kqv"].init_shape(
            (
                0,
                (self.parent.num_qo_heads + 2 * self.parent.num_kv_heads)
                * self.parent.head_dim,
            )
        )
        # The output "q" has shape [batch_size, num_qo_heads * head_dim]
        self.outputs["q"].init_shape(
            (0, self.parent.num_qo_heads * self.parent.head_dim)
        )


class RopeAppendFA_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals[
            "KVCache"
        ].get_whole_kv_data(self.device_id, self.layer)

    def run(self):
        self.impl.run(
            self.layer,
            self.inputs["kqv"].tensor,
            self.externals["KVCache"],
            self.outputs["q"].tensor,
            offset=0,
        )

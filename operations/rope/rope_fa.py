import logging
import torch

from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheBatched, KVCachevLLM
from utils.prof_marker import prof_marker
from utils.util_functions import tensor_offset_to_req_idx

from .triton.kernels.rope import apply_rotary_emb


class RopeAppendBatchedFAImpl(OperationImpl):
    category_tag = "flash_attn_batched"  # type: ignore[assignment]

    def __init__(
        self, op_base: "RopeAppendBatched", stream: torch.cuda.Stream, device_id: int
    ):
        self.op_base: RopeAppendBatched
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

        positions = self.op_base.per_token_offset  # type: ignore
        assert isinstance(positions, torch.Tensor)
        num_tokens = positions.shape[0]
        cos_sin = self.cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, self.num_qo_heads, self.head_dim)
        query_rot = query[..., : self.rotary_dim]
        apply_rotary_emb(
            query_rot,
            cos,
            sin,
            inplace=True,
            seqlen_offsets=cu_seqlens[:-1],
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # query_rot = _apply_rotary_emb_torch(query_rot, cos, sin)
        # query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        query = query.view(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, self.num_kv_heads, self.head_dim)
        key_rot = key[..., : self.rotary_dim]
        # key_pass = key[..., self.rotary_dim:]
        apply_rotary_emb(
            key_rot,
            cos,
            sin,
            inplace=True,
            seqlen_offsets=cu_seqlens[:-1],
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
        KVCache: KVCacheBatched | KVCachevLLM,
        output: torch.Tensor,
        offset: int = 0,
    ):
        if kqv.shape[0] == 0:
            return

        with torch.cuda.stream(self.stream):
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
            with prof_marker("RopeAppendBatched: Rope"):
                q, k = self.forward(
                    q, k, self.op_base.qo_indices, self.op_base.max_seqlen
                )

            with prof_marker("RopeAppendBatched: KVCachePutBatch"):
                if isinstance(KVCache, KVCacheBatched):
                    KVCache.put_batch(
                        layer,
                        k,
                        v,
                        self.op_base.rev_input_indptr,
                        self.op_base.per_token_offset,
                    )
                elif isinstance(KVCache, KVCachevLLM):
                    KVCache.put_batch(layer, k, v, self.op_base.slot_mapping)
                else:
                    raise ValueError("Unsupported KVCache type")

            with prof_marker("RopeAppendBatched: FinalCopy"):
                output.copy_(q)
                KVCache.store_last_kv(
                    k,
                    v,
                    self.op_base.io_device.tensor_offset,
                    self.op_base.io_device.tensor_offset
                    + self.op_base.io_device.batch_size,
                )


try:
    from vllm._custom_ops import rotary_embedding
    VLLM_CACHE = True
except ImportError:
    VLLM_CACHE = False


class RopeAppendBatchedvLLMImpl(OperationImpl):
    category_tag = "vllm"  # type: ignore[assignment]

    def __init__(
        self, op_base: "RopeAppendBatched", stream: torch.cuda.Stream, device_id: int
    ):
        self.op_base: RopeAppendBatched
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

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        t = torch.arange(self.original_max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).contiguous()
        return cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        if self.cache.device != query.device:
            self.cache = self.cache.to(query.device)
        positions = self.op_base.per_token_offset
        rotary_embedding(positions, query, key, self.head_dim, self.cache, False)
        

    def run(
        self,
        layer: int,
        kqv: torch.Tensor,
        KVCache: KVCacheBatched | KVCachevLLM,
        output: torch.Tensor,
        offset: int = 0,
    ):
        if kqv.shape[0] == 0:
            return

        with torch.cuda.stream(self.stream):
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
            with prof_marker("RopeAppendBatched: Rope"):
                self.forward(q, k)

            with prof_marker("RopeAppendBatched: KVCachePutBatch"):
                if isinstance(KVCache, KVCacheBatched):
                    KVCache.put_batch(
                        layer,
                        k,
                        v,
                        self.op_base.rev_input_indptr,
                        self.op_base.per_token_offset,
                    )
                elif isinstance(KVCache, KVCachevLLM):
                    KVCache.put_batch(layer, k, v, self.op_base.slot_mapping)
                else:
                    raise ValueError("Unsupported KVCache type")

            with prof_marker("RopeAppendBatched: FinalCopy"):
                output.copy_(q)
                KVCache.store_last_kv(
                    k,
                    v,
                    self.op_base.io_device.tensor_offset,
                    self.op_base.io_device.tensor_offset
                    + self.op_base.io_device.batch_size,
                )


class RopeAppendBatched(Operations):
    def __init__(
        self,
        name: str,
        device,
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
        super().__init__(name, device)
        self.inputs = {"kqv": IOWrapper(self, "kqv", device).is_input()}
        self.outputs = {"q": IOWrapper(self, "q", device).is_output()}
        self.externals: dict[str, KVCacheBatched | KVCachevLLM] = {}
        # Save RoPE configuration.
        self.rope_type = rope_type
        self.theta = theta  # typically config.rope_theta
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = RopeAppendBatched_Layer
        self.start_req_idx: int
        self.end_req_idx: int
        self.qo_indices: torch.Tensor
        self.seqlens: torch.Tensor
        self.max_seqlen: int
        self.per_token_offset: torch.Tensor
        self.rev_input_indptr: torch.Tensor
        self.indices: torch.Tensor

    def init_impl_map(self):
        self.add_impl(RopeAppendBatchedFAImpl)  # type: ignore
        self.add_impl(RopeAppendBatchedvLLMImpl)

    def setShape(self, num_kv_heads: int, num_qo_heads: int, head_dim: int, tp_size: int = 1) -> None:  # type: ignore
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.inputs["kqv"].init_shape(
            (0, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim // self.tp_size)
        )
        self.outputs["q"].init_shape(
            (0, self.num_qo_heads * self.head_dim // self.tp_size)
        )

    def update(self, qo_indices: list[int], decode_batchsize: int, device_id: int):
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.update(qo_indices, decode_batchsize, device_id)
        else:
            """Stores the starting indices for the query/key segments."""
            self.io_device = self.children[device_id].inputs["kqv"]
            self.start_req_idx = tensor_offset_to_req_idx(
                qo_indices, self.io_device.tensor_offset
            )
            self.end_req_idx = tensor_offset_to_req_idx(
                qo_indices, self.io_device.tensor_offset + self.io_device.batch_size
            )
            if self.start_req_idx == self.end_req_idx:
                return
            self.qo_indices = (
                torch.tensor(
                    qo_indices[self.start_req_idx : self.end_req_idx + 1],
                    dtype=torch.int32,
                    device=f"cuda:{device_id}",
                )
                - qo_indices[self.start_req_idx]
            )
            self.seqlens = self.qo_indices.diff()
            self.max_seqlen = int(self.seqlens.max().item())
            self.per_token_offset = torch.zeros(int(self.qo_indices[-1].item()))
            self.rev_input_indptr = torch.zeros(int(self.qo_indices[-1].item()))
            self.indices = self.externals["KVCache"].get_indices(
                self.start_req_idx, self.end_req_idx
            )
            for i, seqlen in enumerate(self.seqlens.tolist()):
                self.per_token_offset[
                    self.qo_indices[i] : self.qo_indices[i] + seqlen
                ] = (torch.arange(-seqlen, 0) + self.indices[i].cpu())
                self.rev_input_indptr[
                    self.qo_indices[i] : self.qo_indices[i] + seqlen
                ] = (i + self.start_req_idx)
            self.qo_indices = self.qo_indices.to(
                dtype=torch.int32, device=f"cuda:{device_id}"
            )
            self.per_token_offset = self.per_token_offset.to(
                dtype=torch.long, device=f"cuda:{device_id}"
            )
            self.rev_input_indptr = self.rev_input_indptr.to(
                dtype=torch.int32, device=f"cuda:{device_id}"
            )
            if isinstance(self.externals["KVCache"], KVCachevLLM):
                self.slot_mapping = self.externals["KVCache"].get_slot_mapping(
                    self.rev_input_indptr, self.per_token_offset
                )

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"self.seqlens: {self.seqlens}")
                logging.debug(f"self.indices: {self.indices}")
                logging.debug(f"self.qo_indices: {self.qo_indices}")
                logging.debug(f"self.start_req_idx: {self.start_req_idx}")
                logging.debug(f"self.end_req_idx: {self.end_req_idx}")
                logging.debug(f"self.per_token_offset: {self.per_token_offset}")
                logging.debug(f"self.rev_input_indptr: {self.rev_input_indptr}")
                logging.debug(f"self.slot_mapping: {self.slot_mapping}")

    def copy_nano(self, index: int):
        new_op = RopeAppendBatched(
            f"{self.name}{index}",
            self.device,
            self.rope_type,
            self.theta,
            self.factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.original_max_position_embeddings,
        )
        new_op.set_category(self.category)
        new_op.externals = self.externals
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.nano_ops.append(new_op)  # type: ignore

        return new_op

    def profile(self) -> None:
        raise NotImplementedError(
            "Profile method is not implemented for RopeAppendBatched."
        )


class RopeAppendBatched_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)

    def run(self):
        self.impl.run(
            self.layer,
            self.inputs["kqv"].tensor,
            self.externals["KVCache"],
            self.outputs["q"].tensor,
            offset=0,
        )

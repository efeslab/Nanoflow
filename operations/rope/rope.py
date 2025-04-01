import torch
import math
import time
import nvtx
import bind_ropeappend
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache

def rotate_half(x):
    """Rotates the last half of the last dimension."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RopeAppendTorchImpl(OperationImpl):
    category_tag = "torch"

    def apply_rope(self, rope_type, theta, original_max_position_embeddings, low_freq_factor, high_freq_factor, factor, x, output, offset=0):
        """
        Applies RoPE to the tensor `x` (of shape [seq_len, head_dim]). For llama3,
        we adjust the inverse frequency vector as described in the paper.
        
        Args:
            x (torch.Tensor): Input tensor with shape [seq_len, head_dim].
            offset (int): The starting position offset.
        
        Returns:
            torch.Tensor: The rotated tensor.
        """
        # print("using torch")
        seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)

        if rope_type == "llama3.1":
            # Compute the basic inverse frequency vector using theta.
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
            )
            # Compute the wavelengths.
            wavelen = 2 * math.pi / inv_freq
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor

            # For frequencies with wavelengths greater than the low bound, divide inv_freq by factor.
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

            # For values in between, interpolate smoothly.
            smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (1 - smooth_factor) * (inv_freq_llama / factor) + smooth_factor * inv_freq_llama

            # Identify indices where wavelengths are in the medium range.
            is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
            inv_freq_final = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        elif rope_type == "llama3":
            base = 500000.0
            dim = 128
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            # print("inv_freq:", inv_freq)
            inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
            position_ids_expanded = positions[None, None, :].float()
            with torch.autocast(device_type=device.type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            x = x.reshape(1, seq_len, -1, dim).transpose(1, 2)
            # print("x shape:", x.shape)
            # print("cos shape:", cos.shape)
            x = x * cos + rotate_half(x) * sin
            output.copy_(x.transpose(1, 2).reshape(positions.shape[0], -1).to(dtype=dtype))
            return
            
        else:
            # Default RoPE: simply use the base theta.
            inv_freq_final = 1.0 / (
                theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
            )

        # Compute the sinusoidal inputs.
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq_final)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()

        # Expand sin and cos to match x's dimension.
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)

        # Apply the RoPE transformation.
        output.copy_(x * cos + rotate_half(x) * sin)
        return

    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies, kv_indptr, kv_indices, kv_last_page_len, rev_input_indptr, per_token_offset, kqv, KVCache, rope_type, theta, original_max_position_embeddings, low_freq_factor, high_freq_factor, factor, output, decode_flag, offset=0):
        # Determine the number of elements for each slice.
        layout_strides = [
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
            num_qo_heads * head_dim,
        ]
        # Split kqv into key, query, and value (here assumed to be in the order: k, q, v).
        # print("kqv shape:", kqv.shape)
        k, v, q = torch.split(kqv, layout_strides, dim=1)
        k = k.contiguous()
        v = v.contiguous()
        q = q.contiguous()

        # Process each batch element.
        for i in range(len(qo_indicies) - 1):
            start = qo_indicies[i]
            end = qo_indicies[i + 1]
            sub_q = q[start:end, :]
            sub_k = k[start:end, :]
            if not decode_flag or KVCache[layer].get_indices(i) is None:
                last_offest = 0
            else:
                last_offest = KVCache[layer].get_indices(i)[0]
            self.apply_rope(
                rope_type,
                theta,
                original_max_position_embeddings,
                low_freq_factor,
                high_freq_factor,
                factor,
                sub_q,
                output=sub_q,
                offset=last_offest
            )
            self.apply_rope(
                rope_type,
                theta,
                original_max_position_embeddings,
                low_freq_factor,
                high_freq_factor,
                factor,
                sub_k,
                output=sub_k,
                offset=last_offest
            )

            # Write the updated values back.
            q[start:end, :] = sub_q
            k[start:end, :] = sub_k

            # Update the external KVCache with the new key and value.
            KVCache[layer].put(i, sub_k, v[start:end, :])

        output.copy_(q)
        

class RopeAppendCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, layer, page_size, head_dim, num_qo_heads, num_kv_heads, qo_indicies, rev_input_indptr, per_token_offset, kqv, k_data, v_data, rope_type, theta, original_max_position_embeddings, low_freq_factor, high_freq_factor, factor, output, decode_flag, offset=0):
        
        # with nvtx.annotate("RopeAppendCuda: GetKVCache"):
            # k_data, v_data = KVCache.get_whole_kv_data(layer)\
            # k_data = k_data_all[layer]
            # v_data = v_data_all[layer]

        with nvtx.annotate("RopeAppendCuda: SplitRopeAppend"):
            bind_ropeappend.splitRopeAppend(
                k_data,
                v_data,
                kqv,
                output,
                rev_input_indptr,
                per_token_offset,
                len(qo_indicies) - 1,
                page_size,
                num_kv_heads,
                num_qo_heads,
                head_dim,
                1.0,
                500000.0,
                0.0,
                0.0
            )

class RopeAppend(Operations):
    def __init__(
        self,
        name,
        rope_type="llama3",
        theta=10000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
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
        self.inputs = {"kqv": IOWrapper(self, "kqv", IOBufferType.FULL)}
        self.outputs = {"q": IOWrapper(self, "q", IOBufferType.FULL)}
        self.externals = {"KVCache": None, "k_data": None, "v_data": None}
        
        # Save RoPE configuration.
        self.rope_type = rope_type
        self.theta = theta  # typically config.rope_theta
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        self.impl_map = {}
        self.init_impl_map()

    def init_impl_map(self):
        self.add_impl(RopeAppendTorchImpl)
        self.add_impl(RopeAppendCudaImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        # The input tensor "kqv" is assumed to have a flattened layout:
        # [batch_size, (num_qo_heads + 2 * num_kv_heads) * head_dim]
        self.inputs["kqv"].shape = (
            self.batch_size,
            (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim,
        )
        # The output "q" has shape [batch_size, num_qo_heads * head_dim]
        self.outputs["q"].shape = (self.batch_size, self.num_qo_heads, self.head_dim)

    def update(self, page_size, qo_indicies, kv_indptr, kv_indices, kv_last_page_len, rev_input_indptr, per_token_offset, decode_flag=False):
        """Stores the starting indices for the query/key segments."""
        self.page_size = page_size
        self.qo_indicies = qo_indicies
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_len = kv_last_page_len
        self.rev_input_indptr = rev_input_indptr
        self.per_token_offset = per_token_offset
        self.decode_flag = decode_flag
        if self.impl.category_tag == "cuda":
            bind_ropeappend.updateKVCache(self.kv_indptr, self.kv_indices, self.kv_last_page_len, len(self.kv_last_page_len), self.page_size, self.num_kv_heads, self.num_qo_heads, self.head_dim)

    def profile(self):
        input_kqv = torch.randn(2, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for category_tag, impl in self.impl_map.items():
            out = torch.zeros((2, self.num_qo_heads * self.head_dim), dtype=torch.float16, device='cuda')
            # print("name:", category_tag)
            if category_tag == "torch":
                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [KVCacheNone()], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

                output_list.append(out)

                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [KVCacheTorch()], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

            elif category_tag == "cuda":
                kv_pool = DistKVPool(1, self.num_kv_heads, self.head_dim, 2048, 7, 1)
                batchde_kv = BatchedDistKVCache(kv_pool, 0)
                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [batchde_kv], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

            # print("out:", out)
            output_list.append(out)
        
        self.checkConsistencyBetweenImpl(output_list)
        # print("RopeAppend profile passed")
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            output = torch.zeros((batch_size, self.num_qo_heads * self.head_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl_instance.category_tag
                if category_tag == "torch":
                    kv_caches_choices = ['nokv', 'torch']
                elif category_tag == "cuda":
                    kv_caches_choices = ['flashinfer']

                total_latency = 0
                for kv_choice in kv_caches_choices:
                    for round in range(rounds):
                        input_kqv = torch.randn(batch_size, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim, dtype=torch.float16, device='cuda')
                        if kv_choice == 'nokv':
                            kv_caches = [KVCacheNone()]
                        elif kv_choice == 'torch':
                            kv_caches = [KVCacheTorch()]
                        elif kv_choice == 'flashinfer':
                            kv_pool = DistKVPool(1, self.num_kv_heads, self.head_dim, 2048, 7, 1)
                            batchde_kv = BatchedDistKVCache(kv_pool, 0)
                            kv_caches = [batchde_kv]

                        start_time = time.time()
                        impl_instance.run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, batch_size], dtype=torch.int32).cuda(), input_kqv, kv_caches, self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, output, False)
                        if round > 0:
                            total_latency += time.time() - start_time

                    average_time = total_latency / rounds
                    print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}" + f"with_{kv_caches[0].name}", batch_size, average_time))
                    self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}" + f"with_{kv_caches[0].name}", batch_size, average_time))
        self.conn.commit()

    def run(self, layer):
        """
        The run method splits the input `kqv` tensor into key, query, and value tensors,
        applies RoPE (using the llama3 variant if selected) to the query and key portions,
        and writes the updated keys to an external KV cache. The output "q" is set as a copy of v.
        """
        kqv = self.inputs["kqv"].tensor
        self.impl.run(layer, self.head_dim, self.num_qo_heads, self.num_kv_heads, self.qo_indicies, self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.rev_input_indptr, self.per_token_offset, kqv, self.externals["KVCache"], self.externals["k_data"], self.externals["v_data"], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, self.outputs["q"].tensor, self.decode_flag, offset=0)

class RopeAppend_Layer(Operations):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.k_data_ptr, self.v_data_ptr = operator_device.externals["KVCache"].get_whole_kv_data(self.layer)
        self.impl = operator_device.impl

    def run(self):
        self.operator_device.impl.run(self.layer, self.operator_device.page_size, self.operator_device.head_dim, self.operator_device.num_qo_heads, self.operator_device.num_kv_heads, self.operator_device.qo_indicies, self.operator_device.rev_input_indptr, self.operator_device.per_token_offset, self.inputs["kqv"].tensor, self.k_data_ptr, self.v_data_ptr, self.operator_device.rope_type, self.operator_device.theta, self.operator_device.original_max_position_embeddings, self.operator_device.low_freq_factor, self.operator_device.high_freq_factor, self.operator_device.factor, self.operator_device.outputs["q"].tensor, self.operator_device.decode_flag, offset=0)
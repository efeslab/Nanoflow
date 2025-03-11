import torch
import math
import time
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl

def rotate_half(x):
    """Rotates the last half of the last dimension."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RopeAppendTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, rope_type, theta, original_max_position_embeddings, low_freq_factor, high_freq_factor, factor, x, output, offset=0):
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

            x = x.reshape(1, seq_len, -1, 128).transpose(1, 2)
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

    def init_impl_map(self):
        self.add_impl(RopeAppendTorchImpl)

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
        self.outputs["q"].shape = (self.batch_size, self.num_qo_heads * self.head_dim)

    def update(self, qo_indicies):
        """Stores the starting indices for the query/key segments."""
        self.qo_indicies = qo_indicies

    def profile(self):
        pass

    def run(self, layer):
        """
        The run method splits the input `kqv` tensor into key, query, and value tensors,
        applies RoPE (using the llama3 variant if selected) to the query and key portions,
        and writes the updated keys to an external KV cache. The output "q" is set as a copy of v.
        """
        kqv = self.inputs["kqv"].tensor
        # Determine the number of elements for each slice.
        layout_strides = [
            self.num_qo_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ]
        # Split kqv into key, query, and value (here assumed to be in the order: k, q, v).
        q, k, v = torch.split(kqv, layout_strides, dim=1)
        k = k.contiguous()
        v = v.contiguous()
        q = q.contiguous()

        # Process each batch element.
        for i in range(len(self.qo_indicies) - 1):
            start = self.qo_indicies[i]
            end = self.qo_indicies[i + 1]
            sub_q = q[start:end, :]
            sub_k = k[start:end, :]

            # Apply RoPE to both the query and key sub-tensors.
            self.impl.run(
                self.rope_type,
                self.theta,
                self.original_max_position_embeddings,
                self.low_freq_factor,
                self.high_freq_factor,
                self.factor,
                sub_q,
                output=sub_q,
                offset=0
            )
            self.impl.run(
                self.rope_type,
                self.theta,
                self.original_max_position_embeddings,
                self.low_freq_factor,
                self.high_freq_factor,
                self.factor,
                sub_k,
                output=sub_k,
                offset=0,
            )

            # Write the updated values back.
            q[start:end, :] = sub_q
            k[start:end, :] = sub_k

            # Update the external KVCache with the new key and value.
            self.externals["KVCache"].put(layer, i, sub_k, v[start:end, :])

        self.outputs["q"].tensor.copy_(q)
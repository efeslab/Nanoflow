import torch
import math

def rotate_half(x):
    """Rotates the last half of the last dimension."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(rope_type, theta, head_dim, original_max_position_embeddings, low_freq_factor, high_freq_factor, factor, x, output, offset=0):
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
        dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
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
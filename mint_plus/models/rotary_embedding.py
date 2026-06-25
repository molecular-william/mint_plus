from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]

        return apply_rotary_pos_emb(x, cos, sin)


class RotaryEmbedding_flex(nn.Module):
    """
    Rotary Position Embedding module supporting arbitrary positions (Per‑Chain 1D RoPE).
    Fully compatible with torch.compile and validation-step shape variations.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim          # head dimension, must be even
        self.max_seq_len = max_seq_len

        # 1. Compute geometric progression of rotation frequencies
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        # 2. Pre‑compute complete cosine and sine tables for half dimension
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)          # (max_seq_len, dim//2)
        self.register_buffer("_cos_table", freqs.cos(), persistent=True)
        self.register_buffer("_sin_table", freqs.sin(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, T, D) [Flex Attention Head Format] 
               OR (B_heads, T, D) [3D Block Format]
        """
        original_shape = x.shape
        
        # 1. Normalize the primary tensor layout and extract dimensions dynamically
        if x.dim() == 4:
            bsz, num_heads, seq_len, head_dim = x.shape
            batch_heads = bsz * num_heads
            # Flatten to 3D to match indexing logic smoothly
            x_flat = x.transpose(1, 2).reshape(batch_heads, seq_len, head_dim)
        else:
            batch_heads, seq_len, head_dim = x.shape
            x_flat = x

        # 2. THE COMPILER SAFE FIX:
        # Dynamically construct clean, predictable 2D index arrays directly on the targets device.
        # This completely avoids tracking mutated FakeTensors passed from the graph compiler.
        device = x.device
        pos_range = torch.arange(seq_len, device=device, dtype=torch.long)
        safe_position_ids = pos_range.unsqueeze(0).expand(batch_heads, -1)

        # 3. Perform standard Rotary Math
        x1, x2 = x_flat.chunk(2, dim=-1)

        # Slice embedding tables via our safe 2D indices
        cos = self._cos_table[safe_position_ids]   # Shape: (batch_heads, seq_len, head_dim//2)
        sin = self._sin_table[safe_position_ids]   # Shape: (batch_heads, seq_len, head_dim//2)

        # Apply rotation transformations
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        out_flat = torch.cat([x1_rot, x2_rot], dim=-1)

        # 4. Revert tensor layout to match input expectations perfectly
        if original_shape != out_flat.shape:
            return out_flat.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            
        return out_flat
        
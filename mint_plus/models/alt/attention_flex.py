from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from mint_plus.models.rotary_embedding import RotaryEmbedding, RotaryEmbedding_flex
from mint_plus.models.munit_scaling import UnitScaledLinear

class MultiHeadAttentionFlex(nn.Module):
    """
    Optimized Multi-Head Attention using PyTorch FlexAttention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_rotary_embeddings: bool = False,
        no_proj: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = None if no_proj else nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = RotaryEmbedding_flex(dim=self.head_dim) if use_rotary_embeddings else None

    def forward(
        self,
        x: torch.Tensor,  # Expected shape: (B, T, E) for FlexAttention compatibility
        block_mask = None, 
        position_ids = None,
    ) -> torch.Tensor:
        bsz, tgt_len, embed_dim = x.shape
        
        # 1. Project and reshape to FlexAttention format: (B, H, T, D)
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scale queries ahead of time (or pass as scale parameter in flex)
        q = q * (self.head_dim ** -0.5)

        # 3. Apply rotary embeddings if required
        if self.rot_emb is not None:
            # Adjust shapes if your custom rotary embedding expects flat heads
            q = self.rot_emb(q)
            k = self.rot_emb(k)

        # 4. Invoke the compiled FlexAttention kernel
        # Flex handles softmax, masking, and head aggregation natively in fused Triton code
        attn_out = flex_attention(
            q, k, v, 
            block_mask=block_mask, 
        )

        # 5. Reshape back to (B, T, E)
        #attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        if self.out_proj is not None:
            attn_out = self.out_proj(attn_out)
            
        return attn_out


class MultiHeadAttentionFlex_fp8(nn.Module):
    """
    Optimized Multi-Head Attention using PyTorch FlexAttention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_rotary_embeddings: bool = False,
        no_proj: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = dropout

        self.q_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
        self.out_proj = None if no_proj else UnitScaledLinear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = RotaryEmbedding_flex(dim=self.head_dim) if use_rotary_embeddings else None

    def forward(
        self,
        x: torch.Tensor,  # Expected shape: (B, T, E) for FlexAttention compatibility
        block_mask = None, 
        position_ids = None,
    ) -> torch.Tensor:
        bsz, tgt_len, embed_dim = x.shape
        
        # 1. Project and reshape to FlexAttention format: (B, H, T, D)
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scale queries ahead of time (or pass as scale parameter in flex)
        q = q * (self.head_dim ** -0.5)

        # 3. Apply rotary embeddings if required
        if self.rot_emb is not None:
            # Adjust shapes if your custom rotary embedding expects flat heads
            q = self.rot_emb(q)
            k = self.rot_emb(k)

        # 4. Invoke the compiled FlexAttention kernel
        # Flex handles softmax, masking, and head aggregation natively in fused Triton code
        attn_out = flex_attention(
            q, k, v, 
            block_mask=block_mask, 
        )

        # 5. Reshape back to (B, T, E)
        #attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        if self.out_proj is not None:
            attn_out = self.out_proj(attn_out)
            
        return attn_out
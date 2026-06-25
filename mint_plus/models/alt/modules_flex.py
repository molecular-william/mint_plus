from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mint_plus.models.attention_flex import MultiHeadAttentionFlex, MultiHeadAttentionFlex_fp8
from torch.nn.attention.flex_attention import create_block_mask

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

        
class TransformerLayer_flex(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rotary_embeddings: bool = True,
        fp8: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        # Intra-chain channel
        self.self_attn = MultiHeadAttentionFlex(
            embed_dim, attention_heads, use_rotary_embeddings=use_rotary_embeddings
        )

        # Inter-chain channel
        if fp8:
            self.multimer_attn = MultiHeadAttentionFlex_fp8(
                embed_dim, attention_heads, use_rotary_embeddings=False,
            )
        else:
             self.multimer_attn = MultiHeadAttentionFlex(
                embed_dim, attention_heads, use_rotary_embeddings=False,
            )
            
        self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim)
        self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
        self.final_layer_norm = ESM1bLayerNorm(embed_dim)

        
    # Diverge execution paths depending on block mask layout
    def forward(
        self,
        x: torch.Tensor,
        intra_block_mask=None,
        inter_block_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        residual = x
        x = self.self_attn_layer_norm(x)
        
        if intra_block_mask is not None:
            # Removed the _adjust() logic entirely
            intra_out = self.self_attn(x, block_mask=intra_block_mask)
            inter_out = self.multimer_attn(x, block_mask=inter_block_mask)
            x = intra_out + inter_out
        else:
            x = self.self_attn(x, block_mask=None)

        x = residual + x

        # Feed Forward Block
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        # Transpose back to original (T, B, E) tracking format
        return x

        
class VanillaFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
    def forward(self, x):
        x = F.gelu(self.fc1(x))  # uses GELU activation
        x = self.fc2(x)
        return x


class RobertaLMHead(nn.Module):
    """
    Language modeling head, maps transformer hidden states to token predictions.
    For MLM objective, where model predicts original amino acid at positions masked.
    Has weight tying, output projection matrix W_out is transpose of input embedding matrix.
    Saves vocab_size * embed_dim params, constraints output space, and has good empirial performance.
    Bias term isn't tied, separate learnable parameter
    ===================================
    Example
    ===================================
        >>> embed_dim = 320
        >>> vocab_size = 33
        >>> embed_weights = torch.randn(vocab_size, embed_dim)
        >>> head = RobertaLMHead(embed_dim, vocab_size, embed_weights)
        >>> hidden = torch.randn(2, 50, embed_dim)  # batch=2, seq=50
        >>> logits = head(hidden)
        >>> logits.shape
        torch.Size([2, 50, 33])
    """
    def __init__(self, embed_dim: int, output_dim: int, weight: torch.Tensor):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)  # (batch, seq, embed_dim)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias  # F.linear applies input @ weight^T
        return x


def make_multimer_masks(chain_ids_2d: torch.Tensor, current_seq_len: int):
    device = chain_ids_2d.device
    
    def intra_chain_mask_fn(b, h, q_idx, kv_idx):
        return chain_ids_2d[b, q_idx] == chain_ids_2d[b, kv_idx]

    def inter_chain_mask_fn(b, h, q_idx, kv_idx):
        return chain_ids_2d[b, q_idx] != chain_ids_2d[b, kv_idx]

    # Directly use current_seq_len; FlexAttention natively handles lengths < 128
    intra_block_mask = create_block_mask(
        intra_chain_mask_fn, 
        B=1, 
        H=1, 
        Q_LEN=current_seq_len, 
        KV_LEN=current_seq_len, 
        device=device
    )

    inter_block_mask = create_block_mask(
        inter_chain_mask_fn, 
        B=1, 
        H=1, 
        Q_LEN=current_seq_len, 
        KV_LEN=current_seq_len, 
        device=device
    )

    return intra_block_mask, inter_block_mask
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te  # Added TE import
from mint_plus.models.attention_fp8 import MultiHeadAttention

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


class TransformerLayer_MINT(nn.Module):
    def __init__(self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rotary_embeddings: bool = True,
        use_multimer: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_multimer = use_multimer

        self.self_attn = MultiHeadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        if self.use_multimer:
            self.multimer_attn = MultiHeadAttention(
                embed_dim,
                attention_heads,
                use_rotary_embeddings=False,
                no_proj=True,
            )

        self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
        
        # Swapped inner architecture component to the FP8 compatible subclass below
        self.feed_forward = FeedForwardNetwork_MINT(embed_dim, ffn_embed_dim)
        self.final_layer_norm = ESM1bLayerNorm(embed_dim)

    def _standard_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """
        Standard transformer self‑attention.

        Args:
            x: Input tensor (T, B, E)
            attn_mask: Optional attention mask (e.g., for causality)
            padding_mask: Padding mask for key positions
            need_head_weights: If True, return per‑head weights

        Returns:
            x: Output after attention projection (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        x, attn = self.self_attn(
            x=x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        return x, attn

    def _multimer_attention(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """
        Multimer‑aware attention that uses different pathways for intra‑chain
        and inter‑chain interactions.

        Args:
            x: Input tensor (T, B, E)
            chain_mask: Boolean mask (B, T, T) where True means the two positions
                        belong to different chains → use inter‑chain attention.
            padding_mask: Padding mask for key positions
            need_head_weights: If True, return per‑head weights

        Returns:
            x: Output after attention and projection (T, B, E)
            attn: Attention probabilities (combined distribution)
        """
        # 1. Compute logits and values for both pathways (before softmax)
        intra_logits, intra_values = self.self_attn(
            x=x,
            key_padding_mask=padding_mask,
            before_softmax=True,
        )
        inter_logits, inter_values = self.multimer_attn(
            x=x,
            key_padding_mask=padding_mask,
            before_softmax=True,
        )

        # 2. Expand chain mask to broadcast over attention heads
        #    chain_mask: (B, T, T) -> (B, 1, T, T)
        mask_expanded = chain_mask.unsqueeze(1)
        
        # 3. Combine logits: use inter where different chains, intra elsewhere
        combined_logits = torch.where(mask_expanded, inter_logits, intra_logits)

        # 4. Softmax and dropout on the combined distribution
        attn_probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32)
        attn_probs = attn_probs.type_as(combined_logits)
        attn_probs = F.dropout(attn_probs, p=self.self_attn.dropout, training=self.training)

        # 5. Separate probabilities for each pathway (zero out the other)
        intra_probs = attn_probs.masked_fill(mask_expanded, 0.0)
        inter_probs = attn_probs.masked_fill(~mask_expanded, 0.0)

        # 6. Weighted sum with the respective value matrices
        attn_out = torch.matmul(intra_probs, intra_values) + torch.matmul(inter_probs, inter_values)

        # 7. Output projection (shared from intra‑chain attention)
        #    attn_out: (B, heads, T, head_dim) -> (T, B, E)
        attn_out = attn_out.transpose(1, 2).contiguous()      # (B, T, heads, head_dim)
        attn_out = attn_out.view(*attn_out.shape[:2], -1)     # (B, T, E)
        x = self.self_attn.out_proj(attn_out).transpose(0, 1).contiguous()  # (T, B, E)

        # 8. Return attention weights in required format
        #attn = attn_probs.mean(1)  # Average over heads
        return x, None#attn
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        chain_ids = None,
    ):
        """
        Forward pass of the transformer layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, embed_dim)
            self_attn_mask:
                - Standard mode: optional attention mask (e.g., causal)
                - Multimer mode: chain separation mask (B, T, T) where True = different chains
            self_attn_padding_mask: Padding mask for key positions (B, T)
            need_head_weights: If True, return per‑head attention weights

        Returns:
            x: Output tensor of shape (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        # --- Attention block (pre‑norm) ---
        residual = x
        x = self.self_attn_layer_norm(x)

        if self.use_multimer:
            # Multimer mode: self_attn_mask must be the chain mask
            x, attn = self._multimer_attention(x, self_attn_mask, self_attn_padding_mask)
        else:
            # Standard mode
            x, attn = self._standard_attention(x, self_attn_mask, self_attn_padding_mask)

        x = residual + x

        # --- Feed‑forward block (pre‑norm) ---
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, attn


class FeedForwardNetwork_MINT(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim):
        super().__init__()
        # REPLACED: nn.Linear -> te.Linear with torch backend
        self.fc1 = te.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = te.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class RobertaLMHead(nn.Module):
    """Kept intentionally in high precision to maintain token logit distribution safety"""
    def __init__(self, embed_dim, output_dim, weight: torch.Tensor):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


def compute_per_chain_positions(chain_ids: torch.Tensor) -> torch.Tensor:
    is_new_chain = torch.cat([
        torch.ones_like(chain_ids[:, :1]), 
        (chain_ids[:, 1:] != chain_ids[:, :-1]).long()
    ], dim=1)
    
    positions = torch.cumsum(torch.ones_like(chain_ids), dim=1) - 1
    return positions
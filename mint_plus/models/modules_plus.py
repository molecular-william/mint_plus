"""
MINT Plus -- transformer layer with fused multimer attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint_plus.models.attention import MultiHeadAttention, MultimerAttention
from mint_plus.models.modules import (
    ESM1bLayerNorm,
    VanillaFeedForward,
    compute_per_chain_positions,
)
from mint_plus.models.kernels import fused_multimer_combine


class TransformerLayer_MINT_plus(nn.Module):
    """Transformer layer with fused multimer combine kernel.

    Same constructor as TransformerLayer_MINT. The only difference is
    _multimer_attention_plus uses the fused Triton kernel for the
    where + softmax + dropout + masked_fill + weighted-sum pipeline.
    """

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rotary_embeddings: bool = True,
        use_multimer: bool = False,
        use_rmsnorm: bool = False,
        use_erf_gelu: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_multimer = use_multimer
        self.use_erf_gelu = use_erf_gelu
        self._use_fused_multi_pathway = False  # Dynamo-visible flag

        self.self_attn = MultiHeadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        if use_multimer:
            self.multimer_attn = MultimerAttention(
                embed_dim,
                attention_heads,
            )
        self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim, use_erf_gelu=use_erf_gelu)
        self.self_attn_layer_norm = (
            nn.RMSNorm(embed_dim) if use_rmsnorm else ESM1bLayerNorm(embed_dim)
        )
        self.final_layer_norm = (
            nn.RMSNorm(embed_dim) if use_rmsnorm else ESM1bLayerNorm(embed_dim)
        )

    def _standard_attention(self, x, attn_mask, padding_mask):
        x, attn = self.self_attn(
            x=x, key_padding_mask=padding_mask, attn_mask=attn_mask,
        )
        return x, attn

    def _multimer_attention_plus(self, x, chain_mask, padding_mask):
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        # --- Super-fused path: no (B, H, T, T) logit materialization ---
        if self._use_fused_multi_pathway:
            return self._multimer_attention_superfused(x, chain_mask, padding_mask)

        # --- Fused combine path (materializes logits, then combines) ---
        # QKV projections via existing before_softmax path
        intra_logits, intra_values = sa(
            x=x, key_padding_mask=padding_mask, before_softmax=True)
        inter_logits, inter_values = ma(
            x=x, key_padding_mask=padding_mask, before_softmax=True)

        # --- Single fused kernel: where + softmax + dropout + mask + weighted sum ---
        attn_out = fused_multimer_combine(
            intra_logits, inter_logits, chain_mask,
            intra_values, inter_values,
            dropout_p=sa.dropout if self.training else 0.0,
        )
        # attn_out: (B, H, T, D)

        # --- Output projection (unchanged from original) ---
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        x = sa.out_proj(attn_out).transpose(0, 1).contiguous()

        return x, None

    def _multimer_attention_superfused(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """Multi-pathway fused attention: no (B, H, T, T) logit materialization.

        Uses fused_multi_pathway_attention kernel which does 2xbmm + combine
        + softmax + weighted sum in one pass via Flash-Attention-style tiling.
        """
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        # 1. Project Q/K/V for both pathways (uses fused_qkv when available)
        q_self, k_self, v_self = sa.project_qkv_4d(x)
        q_multi, k_multi, v_multi = ma.project_qkv_4d(x)

        # 2. Apply scaling to Q
        scaling = D ** -0.5
        q_self = q_self * scaling
        q_multi = q_multi * scaling

        # 3. Apply RoPE to Q_self and K_self (multimer has no RoPE)
        if sa.rot_emb is not None:
            q_rope = sa.rot_emb(q_self.reshape(B * H, T, D))
            k_rope = sa.rot_emb(k_self.reshape(B * H, T, D))
            q_self = q_rope.view(B, H, T, D).contiguous()
            k_self = k_rope.view(B, H, T, D).contiguous()

        # 4. Multi-pathway fused attention (no logit materialization)
        from mint_plus.models.kernels.differentiable_attention import (
            differentiable_multi_pathway_attention as fused_attn)
        attn_out = fused_attn(
            q_self, k_self, v_self,
            q_multi, k_multi, v_multi,
            chain_mask,
            dropout_p=sa.dropout if self.training else 0.0,
            training=self.training,
        )
        # attn_out: (B, H, T, D)

        # 5. Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        x = sa.out_proj(attn_out).transpose(0, 1).contiguous()

        return x, None

    def _multimer_attention(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """Native PyTorch multimer attention. Fully differentiable.

        Uses torch.where + F.softmax + torch.matmul instead of Triton kernels,
        so gradients flow correctly through the QKV projections.
        """
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        # 1. Compute logits and values for both pathways (before softmax)
        intra_logits, intra_values = sa(
            x=x, key_padding_mask=padding_mask, before_softmax=True)
        inter_logits, inter_values = ma(
            x=x, key_padding_mask=padding_mask, before_softmax=True)

        # 2. Expand chain mask to broadcast over attention heads
        mask_expanded = chain_mask.unsqueeze(1)

        # 3. Combine logits: use inter where different chains, intra elsewhere
        combined_logits = torch.where(mask_expanded, inter_logits, intra_logits)

        # 4. Softmax and dropout on the combined distribution
        attn_probs = F.softmax(combined_logits, dim=-1, dtype=torch.bfloat16)
        attn_probs = F.dropout(attn_probs, p=sa.dropout, training=self.training)

        # 5. Separate probabilities for each pathway (zero out the other)
        intra_probs = attn_probs.masked_fill(mask_expanded, 0.0)
        inter_probs = attn_probs.masked_fill(~mask_expanded, 0.0)

        # 6. Weighted sum with the respective value matrices
        attn_out = torch.matmul(intra_probs, intra_values) + torch.matmul(
            inter_probs, inter_values)

        # 7. Output projection (shared from intra-chain attention)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(*attn_out.shape[:2], -1)
        x = sa.out_proj(attn_out).transpose(0, 1).contiguous()

        return x, None

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        chain_ids=None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)

        if self.use_multimer:
            # Use native PyTorch attention during training (differentiable,
            # gradients flow to QKV projections). Use Triton-accelerated
            # path during inference (fast, no autograd needed).
            if self.training:
                x, attn = self._multimer_attention(
                    x, self_attn_mask, self_attn_padding_mask,
                )
            else:
                x, attn = self._multimer_attention_plus(
                    x, self_attn_mask, self_attn_padding_mask,
                )
        else:
            x, attn = self._standard_attention(
                x, self_attn_mask, self_attn_padding_mask,
            )

        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, attn


def build_mint_plus(model: nn.Module) -> nn.Module:
    """Replace every TransformerLayer_MINT with the _plus variant.

    For self_attn and multimer_attn, the separate q_proj/k_proj/v_proj weights
    are concatenated into a fused_qkv weight matrix, reducing 3 GEMMs to 1
    per projection. This lowers kernel launch overhead and improves tensor core
    utilization for the backward pass (6 backward GEMMs -> 2).
    """
    for i, old_layer in enumerate(model.layers):
        new_layer = TransformerLayer_MINT_plus(
            embed_dim=old_layer.embed_dim,
            ffn_embed_dim=old_layer.ffn_embed_dim,
            attention_heads=old_layer.attention_heads,
            use_rotary_embeddings=old_layer.use_rotary_embeddings,
            use_multimer=old_layer.use_multimer,
            use_rmsnorm=getattr(old_layer, 'use_rmsnorm', False),
            use_erf_gelu=getattr(old_layer, 'use_erf_gelu', False),
        )

        # ---- Fuse self_attn QKV ----
        old_sa = old_layer.self_attn
        # Create a fused MultiHeadAttention, transfer weights by concatenation
        fused_sa = MultiHeadAttention(
            new_layer.embed_dim, new_layer.attention_heads,
            use_rotary_embeddings=old_sa.rot_emb is not None,
            use_fused_qkv=True,
        )
        # Copy weights: cat q, k, v into fused_qkv
        w_q = old_sa.q_proj.weight
        w_k = old_sa.k_proj.weight
        w_v = old_sa.v_proj.weight
        fused_sa.fused_qkv.weight.data.copy_(
            torch.cat([w_q, w_k, w_v], dim=0))
        if old_sa.q_proj.bias is not None:
            b_q = old_sa.q_proj.bias
            b_k = old_sa.k_proj.bias
            b_v = old_sa.v_proj.bias
            fused_sa.fused_qkv.bias.data.copy_(
                torch.cat([b_q, b_k, b_v], dim=0))
        # Transfer out_proj and RoPE by reference
        fused_sa.out_proj = old_sa.out_proj
        fused_sa.rot_emb = old_sa.rot_emb
        new_layer.self_attn = fused_sa

        # ---- Fuse multimer_attn QKV ----
        if hasattr(old_layer, 'multimer_attn'):
            old_ma = old_layer.multimer_attn
            # Transfer weights by concatenating q, k, v into fused_qkv
            w_q = old_ma.q_proj.weight
            w_k = old_ma.k_proj.weight
            w_v = old_ma.v_proj.weight
            new_layer.multimer_attn.fused_qkv.weight.data.copy_(
                torch.cat([w_q, w_k, w_v], dim=0)
            )
            if old_ma.q_proj.bias is not None:
                b_q = old_ma.q_proj.bias
                b_k = old_ma.k_proj.bias
                b_v = old_ma.v_proj.bias
                new_layer.multimer_attn.fused_qkv.bias.data.copy_(
                    torch.cat([b_q, b_k, b_v], dim=0)
                )
        new_layer.feed_forward = old_layer.feed_forward
        new_layer.self_attn_layer_norm = old_layer.self_attn_layer_norm
        new_layer.final_layer_norm = old_layer.final_layer_norm
        model.layers[i] = new_layer
    return model

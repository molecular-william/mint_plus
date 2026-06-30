"""Transformer layer with mu-Scaling (muS) architecture.

Architecture changes vs standard Pre-LN (TransformerLayer_MINT):

  Pre-LN (current):
    x -> LN -> Attn -> + residual -> LN -> FFN -> + residual

  muS Res-Post-LN:
    x -> Attn -> weighted residual (tau) -> FFN -> weighted residual (tau) -> LN

Key differences:
  1. No self_attn_layer_norm -- normalization happens ONCE at the end
  2. Weighted residuals with tau coefficient
  3. Optional square-root softmax for variance preservation
  4. Reuses existing MultiHeadAttention, MultimerAttention, VanillaFeedForward

The layer is a drop-in replacement for TransformerLayer_MINT with the same
constructor signature. It is meant to be used with the builder function
build_mint_mu() in __init__.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint_plus.models.attention import MultiHeadAttention
from mint_plus.models.modules import VanillaFeedForward
from torch.nn import LayerNorm as ESM1bLayerNorm


class TransformerLayer_MINT_mu(nn.Module):
    """Transformer layer with muS Res-Post-LN architecture.

    Reuses existing attention and FFN submodules. The constructor
    signature matches TransformerLayer_MINT for drop-in replacement.

    Args:
        embed_dim: Model dimension (E).
        ffn_embed_dim: FFN hidden dimension (typically 4*E).
        attention_heads: Number of attention heads.
        use_rotary_embeddings: Apply RoPE to self-attention Q/K.
        use_multimer: Enable dual-pathway (self + cross-chain) attention.
        use_rmsnorm: Not used in muS, kept for API compatibility.
        use_erf_gelu: Use erf-based GELU (matching original ESM-2).
        tau: Residual coefficient for weighted skip connections.
             If None, computed from depth via compute_tau().
        num_layers: Total number of layers in the model (used for tau).
        use_sqrt_softmax: Apply square-root to softmax outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        use_rotary_embeddings: bool = True,
        use_multimer: bool = False,
        use_rmsnorm: bool = False,
        use_erf_gelu: bool = False,
        tau: Optional[float] = None,
        num_layers: Optional[int] = None,
        use_sqrt_softmax: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_multimer = use_multimer
        self.use_erf_gelu = use_erf_gelu
        self._use_fused_multi_pathway = False
        self._use_sqrt_softmax = use_sqrt_softmax

        # Compute tau if not provided
        if tau is None:
            from mint_plus.models.mu_scaling.optim import compute_tau
            tau = compute_tau(num_layers or 30)

        # Reuse existing attention modules (NO norm before attention)
        self.self_attn = MultiHeadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        # Multimer attention -- same as base: MultiHeadAttention(no_proj=True)
        self.multimer_attn: Optional[MultiHeadAttention] = None
        if use_multimer:
            self.multimer_attn = MultiHeadAttention(
                embed_dim,
                attention_heads,
                use_rotary_embeddings=False,
                no_proj=True,  # No output projection; we combine manually
            )

        self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim, use_erf_gelu=use_erf_gelu)

        # Single post-norm (Res-Post-LN)
        self.final_layer_norm = ESM1bLayerNorm(embed_dim)

        # Weighted residual coefficient (registered as buffer for checkpointing)
        self.register_buffer("_tau", torch.tensor(tau, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Standard self-attention (non-multimer path)
    # ------------------------------------------------------------------

    def _standard_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        x, attn = self.self_attn(
            x=x, key_padding_mask=padding_mask, attn_mask=attn_mask,
        )
        return x, None

    # ------------------------------------------------------------------
    # Multimer attention (dual-pathway: intra + inter)
    # ------------------------------------------------------------------

    def _multimer_attention(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Multimer-aware attention with sqrt-softmax support.

        Same logic as TransformerLayer_MINT._multimer_attention but
        with optional sqrt-softmax for variance preservation.
        """
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        # Check if super-fused kernel is active
        if self._use_fused_multi_pathway:
            return self._multimer_attention_superfused(x, chain_mask, padding_mask)

        # Standard path: before_softmax + combine
        intra_logits, intra_values = sa(
            x=x, key_padding_mask=padding_mask, before_softmax=True,
        )
        inter_logits, inter_values = ma(
            x=x, key_padding_mask=padding_mask, before_softmax=True,
        )

        mask_expanded = chain_mask.unsqueeze(1)  # (B, 1, T, T)
        combined_logits = torch.where(mask_expanded, inter_logits, intra_logits)

        # Softmax (fp32 for stability, as recommended by paper)
        attn_probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32)

        # muS: square-root softmax for variance preservation
        if self._use_sqrt_softmax:
            attn_probs = attn_probs.sqrt()

        attn_probs = attn_probs.type_as(combined_logits)
        attn_probs = F.dropout(attn_probs, p=sa.dropout, training=self.training)

        # Split by pathway and weighted sum
        intra_probs = attn_probs.masked_fill(mask_expanded, 0.0)
        inter_probs = attn_probs.masked_fill(~mask_expanded, 0.0)
        attn_out = torch.matmul(intra_probs, intra_values) + torch.matmul(
            inter_probs, inter_values
        )

        # Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        x = sa.out_proj(attn_out).transpose(0, 1).contiguous()

        return x, None

    # ------------------------------------------------------------------
    # Super-fused kernel path (when enabled via enable_fused_multi_pathway)
    # ------------------------------------------------------------------

    def _multimer_attention_superfused(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Multi-pathway fused attention with optional sqrt-softmax.

        When sqrt-softmax is enabled, passes the flag to the Triton kernel
        which uses sqrt(softmax) coefficients with the correct online
        rescaling math (sqrt(old_scale) for acc, sqrt(d) for normalization)
        and the matching backward formula (gradient through sqrt).
        """
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        q_self, k_self, v_self = sa.project_qkv_4d(x)
        q_multi, k_multi, v_multi = ma.project_qkv_4d(x)

        scaling = D ** -0.5
        q_self = q_self * scaling
        q_multi = q_multi * scaling

        if sa.rot_emb is not None:
            q_rope = sa.rot_emb(q_self.reshape(B * H, T, D))
            k_rope = sa.rot_emb(k_self.reshape(B * H, T, D))
            q_self = q_rope.view(B, H, T, D).contiguous()
            k_self = k_rope.view(B, H, T, D).contiguous()

        from mint_plus.models.kernels.differentiable_attention import (
            differentiable_multi_pathway_attention as fused_attn,
        )

        attn_out = fused_attn(
            q_self, k_self, v_self,
            q_multi, k_multi, v_multi,
            chain_mask,
            dropout_p=sa.dropout if self.training else 0.0,
            training=self.training,
            sqrt_softmax=self._use_sqrt_softmax,
        )

        # Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        x = sa.out_proj(attn_out).transpose(0, 1).contiguous()

        return x, None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        chain_ids=None,
    ) -> Tuple[torch.Tensor, None]:
        """muS forward: No pre-norm, weighted residuals, post-norm.

        Args:
            x: Input (T, B, E).
            self_attn_mask: Chain mask (B, T, T) in multimer mode.
            self_attn_padding_mask: Padding mask (B, T).
            chain_ids: Ignored (kept for API compatibility).

        Returns:
            x: Output (T, B, E).
            attn: Always None (kept for API compatibility).
        """
        tau = self._tau.item()
        sqrt_1mt = math.sqrt(1.0 - tau)
        sqrt_t = math.sqrt(tau)

        # --- Attention block (muS: no pre-norm, weighted residual) ---
        if self.use_multimer:
            attn_out, _ = self._multimer_attention(
                x, self_attn_mask, self_attn_padding_mask,
            )
        else:
            attn_out, _ = self._standard_attention(
                x, self_attn_mask, self_attn_padding_mask,
            )

        x = sqrt_1mt * x + sqrt_t * attn_out

        # --- Feed-forward block (muS: no pre-norm, weighted residual) ---
        ffn_out = self.feed_forward(x)
        x = sqrt_1mt * x + sqrt_t * ffn_out

        # --- Post-norm (muS: single norm at end) ---
        x = self.final_layer_norm(x)

        return x, None

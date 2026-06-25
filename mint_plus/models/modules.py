from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mint_plus.models.attention import MultiHeadAttention
from mint_plus.models.kernels.multi_pathway_attention import fused_multi_pathway_attention
from mint_plus.models.kernels.differentiable_attention import (
    differentiable_multi_pathway_attention,
)

from torch.nn import LayerNorm as ESM1bLayerNorm


def gelu_erf(x):
    """Erf-based GELU matching the original MINT paper and ESM-2 pretrained weights.

    PyTorch's F.gelu() uses the tanh approximation by default; this matches the
    original implementation used by fairseq/esm which is:
        x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerLayer_MINT(nn.Module):
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
        self._use_fused_multi_pathway = False  # Dynamo-visible flag, set by enable_fused_multi_pathway()


        # Main self‑attention (intra‑chain in multimer mode)
        self.self_attn = MultiHeadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        # Optional multimer (inter‑chain) attention
        if use_multimer:
            self.multimer_attn = MultiHeadAttention(
                embed_dim,
                attention_heads,
                use_rotary_embeddings=False,
                no_proj=True,          # No output projection; we combine manually
            )
        self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim, use_erf_gelu=use_erf_gelu)
        # Layer norms
        self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
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
        attn_probs = F.softmax(combined_logits, dim=-1, dtype=torch.bfloat16)#torch.float32)
        # attn_probs = attn_probs.type_as(combined_logits)
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
            x, attn = self._multimer_attention_plus(x, self_attn_mask, self_attn_padding_mask)
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

        
    def _multimer_attention_plus(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        sa = self.self_attn
        ma = self.multimer_attn

        # --- Check if multi-pathway fused kernel is enabled ---
        # Direct attribute access (not getattr) so torch._dynamo can trace
        # through this branch without inserting a graph break.
        if self._use_fused_multi_pathway:
            return self._multimer_attention_superfused(x, chain_mask, padding_mask)

        # --- Original fused combine path (via before_softmax + fused kernel) ---
        intra_logits, intra_values = sa(
            x=x, key_padding_mask=padding_mask, before_softmax=True)
        inter_logits, inter_values = ma(
            x=x, key_padding_mask=padding_mask, before_softmax=True)

        # Use differentiable wrapper when gradients are needed
        needs_grad = any(
            p.requires_grad for p in sa.parameters()
        ) or any(
            p.requires_grad for p in ma.parameters()
        )
        if needs_grad:
            from mint_plus.models.kernels.differentiable_attention import (
                differentiable_multimer_combine)
            attn_out = differentiable_multimer_combine(
                intra_logits, inter_logits, chain_mask,
                intra_values, inter_values,
                dropout_p=sa.dropout if self.training else 0.0,
            )
        else:
            attn_out = fused_multimer_combine(
                intra_logits, inter_logits, chain_mask,
                intra_values, inter_values,
                dropout_p=sa.dropout if self.training else 0.0,
            )
        # attn_out: (B, H, T, D)

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

        # 1. Project Q/K/V for both pathways (separate GEMMs via cuBLAS)
        q_self, k_self, v_self = sa.project_qkv_4d(x)
        q_multi, k_multi, v_multi = ma.project_qkv_4d(x)

        # 2. Apply scaling to Q
        scaling = D ** -0.5
        q_self = q_self * scaling
        q_multi = q_multi * scaling

        # 3. Apply RoPE to Q_self and K_self (multimer has no RoPE)
        # Collapse (B, H, T, D) -> (B*H, T, D) for RotaryEmbedding
        from mint_plus.models.rotary_embedding import apply_rotary_pos_emb
        if sa.rot_emb is not None:
            q_rope, k_rope = sa.rot_emb(
                q_self.reshape(B * H, T, D)
            ), sa.rot_emb(
                k_self.reshape(B * H, T, D)
            )
            q_self = q_rope.view(B, H, T, D).contiguous()
            k_self = k_rope.view(B, H, T, D).contiguous()

        # 4. Multi-pathway fused attention (no logit materialization)
        # Choose kernel based on whether gradients are needed.
        # If all attention params are frozen, use the fast raw Triton kernel.
        # Otherwise, use the differentiable wrapper (native PyTorch backward).
        needs_grad = any(
            p.requires_grad for p in sa.parameters()
        ) or any(
            p.requires_grad for p in ma.parameters()
        )
        if needs_grad:
            attn_out = differentiable_multi_pathway_attention(
                q_self, k_self, v_self,
                q_multi, k_multi, v_multi,
                chain_mask,
                dropout_p=sa.dropout if self.training else 0.0,
                training=self.training,
            )
        else:
            attn_out = fused_multi_pathway_attention(
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

        
class VanillaFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, use_erf_gelu: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.use_erf_gelu = use_erf_gelu
    def forward(self, x):
        if self.use_erf_gelu:
            x = gelu_erf(self.fc1(x))
        else:
            x = F.gelu(self.fc1(x))
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


def compute_per_chain_positions(chain_ids: torch.Tensor) -> torch.Tensor:
    """
    Transforms chain IDs into per-chain relative 1D position vectors.
    Example Input:  [0, 0, 0, 1, 1, 1] (Chain 0 followed by Chain 1)
    Example Output: [0, 1, 2, 0, 1, 2]
    """
    # Shift chain IDs to detect boundaries
    # Compares each element with its neighbor to mark where a new chain starts
    is_new_chain = torch.cat([
        torch.ones_like(chain_ids[:, :1]), 
        (chain_ids[:, 1:] != chain_ids[:, :-1]).long()
    ], dim=1)
    
    # Cumulative sum creates unique tracking indices that reset per chain boundary
    positions = torch.cumsum(torch.ones_like(chain_ids), dim=1) - 1
    
    # Use cumulative sum of boundary markers to calculate resets
    # This aligns the index system back to zero at each transition point
    resets = torch.where(is_new_chain == 1, positions, torch.zeros_like(positions))
    reset_offsets = torch.cummax(resets, dim=1)[0]
    
    return positions - reset_offsets


# ---- Convenience: enable multi-pathway fused kernel on all layers ----

def enable_fused_multi_pathway(model: nn.Module, enabled: bool = True):
    """Toggle the multi-pathway fused attention kernel on all layers.

    When enabled, the super-fused kernel replaces the per-layer
    before_softmax + fused_multimer_combine pipeline with a single
    Flash-Attention-style kernel that avoids materializing (B, H, T, T)
    logits. Set to False (default) to keep the original pipeline.

    Usage:
        model = MINT.from_config('150M')
        enable_fused_multi_pathway(model, True)  # enable super-fused
        model = torch.compile(model, mode='default')
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # CheckpointedBlock wrapping multiple layers
            for sub in layer.layers:
                sub._use_fused_multi_pathway = enabled
        else:
            layer._use_fused_multi_pathway = enabled


# ---- Block-level gradient checkpointing ----

class CheckpointedBlock(nn.Module):
    """Container for a contiguous block of transformer layers,
    checkpointed as a single segment during training backward pass.

    Drop-in compatible with the existing per-layer checkpoint loop
    in MINT.forward(). Returns (x, None) to match the interface of
    a single TransformerLayer_MINT.
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
    ):
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_mask=self_attn_mask,
                chain_ids=chain_ids,
            )
        # Return (x, None) to match individual layer return signature
        # so the existing x, attn = checkpoint(...) unpacking works.
        return x, None

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]


def build_checkpointed_model(
    model: nn.Module,
    block_size: int = 3,
) -> nn.Module:
    """Group model.layers into CheckpointedBlocks.

    Each block of ``block_size`` contiguous layers is wrapped in a
    single CheckpointedBlock and checkpointed as one segment during
    backward. The existing per-layer checkpoint loop in MINT.forward()
    works unchanged because CheckpointedBlock.forward() returns the
    same (x, attn) signature.

    Args:
        model: A MINT model whose .layers is a ModuleList of transformer
               layers. Should already be converted to _plus variants if
               using the fused kernel.
        block_size: Number of layers per checkpoint block. Must evenly
                    divide len(model.layers).

    Returns:
        The same model with model.layers replaced by CheckpointedBlocks.

    Raises:
        ValueError: If block_size does not evenly divide the layer count.

    Usage:
        >>> from mint_plus.models.modules import build_checkpointed_model
        >>> model = MINT.from_config('150M')
        >>> model = build_mint_plus(model)            # optional: fused kernel
        >>> model = build_checkpointed_model(model, 3) # block-3 checkpointing
        >>> model = torch.compile(model, mode='default')
    """
    layers = list(model.layers)
    num_layers = len(layers)

    if num_layers % block_size != 0:
        raise ValueError(
            f"block_size={block_size} does not evenly divide "
            f"{num_layers} layers. Try a different block_size."
        )

    blocks = []
    for i in range(0, num_layers, block_size):
        block_layers = layers[i:i + block_size]
        blocks.append(CheckpointedBlock(nn.ModuleList(block_layers)))

    model.layers = nn.ModuleList(blocks)
    # Store block size so MINT.forward() can adapt if needed
    model._checkpoint_block_size = block_size

    return model
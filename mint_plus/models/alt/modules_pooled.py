from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mint_plus.models.attention import MultiHeadAttention

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


class TransformerLayer_MINT_pooled(nn.Module):
    def __init__(
        self,
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


        # Main self‑attention (intra‑chain in multimer mode)
        self.self_attn = MultiHeadAttention(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        # Optional multimer (inter‑chain) attention
        # Replace nn.MultiheadAttention with explicit projections for SDPA
        if use_multimer:
            # We project input x (T, B, E) to Q, K, V
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
        self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim)
        # Layer norms
        self.self_attn_layer_norm = nn.RMSNorm(embed_dim) if use_rmsnorm else ESM1bLayerNorm(embed_dim)
        self.final_layer_norm = nn.RMSNorm(embed_dim) if use_rmsnorm else ESM1bLayerNorm(embed_dim)


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

    def _multimer_attention_pooled(
        self,
        x: torch.Tensor,                # (T, B, E)
        padding_mask: torch.Tensor,     # (B, T)
        chain_ids: torch.Tensor,        # (B, T)
        max_chain: int = 2,             # Keep as scalar variable ceiling to avoid graph breaks
    ):
        T, B, E = x.shape
        H = self.attention_heads
        D = E // H
        
        # 1. View-transpose (B, T, E) without copying data
        x_t = x.transpose(0, 1)   
    
        # 2. Extract Chain Pool Matrices
        sums = torch.zeros(B, max_chain, E, device=x.device, dtype=x.dtype)
        sums.scatter_add_(1, chain_ids.unsqueeze(-1).expand(-1, -1, E), x_t)
        
        counts = torch.zeros(B, max_chain, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, chain_ids.unsqueeze(-1), x_t[:, :, :1].fill_(1.0))
        
        chain_pool = sums / torch.clamp_(counts, min=1.0) # In-place clamp
        chain_pool = chain_pool.transpose(0, 1)          # (max_chain, B, E)
    
        # 3. Project Q from the full sequence, and K, V from the concatenated states
        # Shape of combined_kv: (T + max_chain, B, E)
        combined_kv = torch.cat([x, chain_pool], dim=0) 
        
        # Linear projections
        q = self.q_proj(x)            # (T, B, E)
        k = self.k_proj(combined_kv)  # (T + max_chain, B, E)
        v = self.v_proj(combined_kv)  # (T + max_chain, B, E)
        
        # Reshape to SDPA required layout: (B, H, SeqLen, HeadDim)
        q = q.transpose(0, 1).view(B, T, H, D).transpose(1, 2)
        k = k.transpose(0, 1).view(B, T + max_chain, H, D).transpose(1, 2)
        v = v.transpose(0, 1).view(B, T + max_chain, H, D).transpose(1, 2)
        
        # 4. Generate the Boolean Attention Mask (Very memory efficient)
        # same_chain specifies who can talk to who within the sequence: (B, T, T)
        same_chain = (chain_ids.unsqueeze(-1) == chain_ids.unsqueeze(-2)) 
        
        # Every token can see the trailing cross-attention pooled vectors: (B, T, max_chain)
        cross_visible = torch.ones(B, T, max_chain, device=x.device, dtype=torch.bool)
        
        # Combine them: (B, T, T + max_chain)
        attn_allowed = torch.cat([same_chain, cross_visible], dim=-1) 
        
        # SDPA expects the mask to be broadcastable to (B, H, T, T + max_chain)
        # We unsqueeze the Head dimension (1) so it broadcasts over all heads automatically
        attn_mask = attn_allowed.unsqueeze(1) 
    
        # 5. Native SDPA Execution 
        # This automatically dispatches to FlashAttention-3 or Memory-Efficient Attention kernels!
        # It handles the boolean mask natively without converting it into a giant float matrix.
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        ]):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False
            ) # Output shape: (B, H, T, D)
    
        # 6. Reshape back and apply final output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        combined_out = self.out_proj(attn_out).transpose(0, 1) # (T, B, E)
    
        return combined_out, None
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        chain_ids: Optional[torch.Tensor] = None,   # new: (B, T) long tensor
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
    
        if self.use_multimer and chain_ids is not None:
            # New pooled inter‑chain attention
            x, attn = self._multimer_attention_pooled(x, self_attn_padding_mask, chain_ids)
        elif self.use_multimer:
            # Old dense inter‑chain attention (needs self_attn_mask as chain mask)
            x, attn = self._multimer_attention(x, self_attn_mask, self_attn_padding_mask)
        else:
            x, attn = self._standard_attention(x, self_attn_mask, self_attn_padding_mask)
    
        x = residual + x
    
        # Feed‑forward block (unchanged)
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x
    
        return x, attn

        
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
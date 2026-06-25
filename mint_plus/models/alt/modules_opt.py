from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mint_plus.models.attention_opt import MultiHeadAttention_Opt
from mint_plus.models.munit_scaling import UnitScaledLinear

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

class TransformerLayer_Opt(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rotary_embeddings: bool = True,
        use_rmsnorm: bool = True,
        use_multimer: bool = True,
        use_swiglu: bool = True,
        layer_type: str = "both",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_multimer = use_multimer
        self.layer_type = layer_type

        self.self_attn = MultiHeadAttention_Opt(
            self.embed_dim,
            self.attention_heads,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        if self.use_multimer and layer_type in ("both", "cross"):
            self.multimer_attn = MultiHeadAttention_Opt(
                self.embed_dim,
                self.attention_heads,
                use_rotary_embeddings=False,  # multimer doesn't use RoPE
                no_proj=True if layer_type == "both" else False,
            )
        # pre layer norm, more stable training than post layer norm
        if use_rmsnorm:
            self.self_attn_layer_norm = nn.RMSNorm(self.embed_dim)
            self.final_layer_norm = nn.RMSNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = ESM1bLayerNorm(self.embed_dim)
            self.final_layer_norm = ESM1bLayerNorm(self.embed_dim)

        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(self.embed_dim, self.ffn_embed_dim) 
        else:
            self.feed_forward = VanillaFeedForward(self.embed_dim, self.ffn_embed_dim)

    def _standard_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        need_head_weights: bool,
        position_ids: Optional[torch.Tensor],
        flash: bool = False,
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
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            flash=flash,
        )
        return x, attn

    def _multimer_attention(
        self,
        x: torch.Tensor,
        chain_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        need_head_weights: bool,
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
        if need_head_weights:
            attn = attn_probs.transpose(0, 1).contiguous()    # (T, B, heads, T)
        else:
            attn = attn_probs.mean(1)                         # Average over heads

        return x, attn
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,  # chain mask (B,T,T) for multimer
        self_attn_padding_mask: torch.Tensor = None,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # per chain positions for rotary
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
           # print('multimer mode****************************8')
            # Multimer mode: self_attn_mask must be the chain mask
            if self.layer_type == "self":
                # intra chain only, standard attn with rotary, mask out cross-chain positions
                intra_mask = (self_attn_mask == 0)  # now is true is same chain
                attn_mask = torch.where(intra_mask, 0.0, float("-inf"))
                x, attn = self._standard_attention(
                    x, attn_mask, 
                    self_attn_padding_mask, 
                    need_head_weights,
                    position_ids,
                    flash=True,
                )
            elif self.layer_type == "cross":
                # inter-chain only, attention only on cross pairs, mask that disables intra-chain
                cross_mask = (self_attn_mask == 1)
                attn_mask = torch.where(cross_mask, 0.0, float("-inf"))
                x, attn = self._standard_attention(
                    x, attn_mask, 
                    self_attn_padding_mask, 
                    need_head_weights,
                    position_ids,
                    flash=True
                )
            else:  # original design
                x, attn = self._multimer_attention(x, self_attn_mask, self_attn_padding_mask, need_head_weights, position_ids,)
        else:
            # Standard mode
            x, attn = self._standard_attention(x, self_attn_mask, self_attn_padding_mask, need_head_weights, position_ids, True)

        x = residual + x

        # --- Feed‑forward block (pre‑norm) ---
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, attn

        
class VanillaFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, fp8=False):
        super().__init__()
        if fp8:
            self.fc1 = UnitScaledLinear(embed_dim, ffn_embed_dim)
            self.fc2 = UnitScaledLinear(ffn_embed_dim, embed_dim)
        else:
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


class SwiGLUFeedForward(nn.Module):  # suggested by Gemini
    def __init__(self, embed_dim, ffn_embed_dim, fp8=False):
        super().__init__()
        # SwiGLU needs splits projections (gate state and value state)
        # cant use pre-trained for this
        if fp8:
            self.w12 = UnitScaledLinear(embed_dim, ffn_embed_dim * 2, bias=False)
            self.w3 = UnitScaledLinear(ffn_embed_dim, embed_dim, bias=False)
        else:
            self.w12 = nn.Linear(embed_dim, ffn_embed_dim * 2, bias=False)
            self.w3 = nn.Linear(ffn_embed_dim, embed_dim, bias=False)
            
            gain = 1 / math.sqrt(2)
            nn.init.xavier_uniform_(self.w12.weight, gain=gain)
            nn.init.xavier_uniform_(self.w3.weight, gain=gain)

    def forward(self, x):  # this trick is said to be more efficieny according to Gemini
        # Swish(W1x) * W2x
        fused_proj = self.w12(x)  # compute both and value states
        gate_state, value_state = torch.chunk(fused_proj, chunks=2, dim=-1)  # split
        gated_hidden = F.silu(gate_state) * value_state  # apply Swish to gate, and multiply element wise
        return self.w3(gated_hidden)


class CrossAttentionLayer(nn.Module):
    def __init__(  # this is specifically for alternating layers
        self,  # accepts mask for same seq, attends to cross seq
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        use_rotary_embeddings: bool = True,
        fp8: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.cross_attn = MultiHeadAttention_Opt(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
            fp8=fp8
        )

        # pre layer norm, more stable training than post layer norm
        if use_rmsnorm:
            self.self_attn_layer_norm = nn.RMSNorm(embed_dim)
            self.final_layer_norm = nn.RMSNorm(embed_dim)
        else:
            self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
            self.final_layer_norm = ESM1bLayerNorm(embed_dim)

        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(embed_dim, ffn_embed_dim) 
        else:
            self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,  # chain mask (B,T,T) for multimer
        self_attn_padding_mask: torch.Tensor = None,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # per chain positions for rotary
    ):
        """
        Forward pass of the transformer layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, embed_dim)
            self_attn_mask:
                - Standard mode: optional attention mask (e.g., causal)
            self_attn_padding_mask: Padding mask for key positions (B, T)
            need_head_weights: If True, return per‑head attention weights

        Returns:
            x: Output tensor of shape (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        # --- Attention block (pre‑norm) ---
        residual = x
        x = self.self_attn_layer_norm(x)

        # inter-chain only, attention only on cross pairs, mask that disables intra-chain
        cross_mask = (self_attn_mask == 1)
        attn_mask = torch.where(cross_mask, 0.0, float("-inf"))
        x_attn, attn = self.cross_attn(
            x=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            flash=True,
        )
        
        x = residual + x_attn
        # --- Feed‑forward block (pre‑norm) ---
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, attn


class SelfAttentionLayer(nn.Module):
    def __init__(  # this is specifically for alternating layers
        self,  # accepts mask for cross seq, only attends to same seq
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        use_rotary_embeddings: bool = True,
        fp8: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.self_attn = MultiHeadAttention_Opt(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
            fp8 = False,
        )

        # pre layer norm, more stable training than post layer norm
        if use_rmsnorm:
            self.self_attn_layer_norm = nn.RMSNorm(embed_dim)
            self.final_layer_norm = nn.RMSNorm(embed_dim)
        else:
            self.self_attn_layer_norm = ESM1bLayerNorm(embed_dim)
            self.final_layer_norm = ESM1bLayerNorm(embed_dim)

        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(embed_dim, ffn_embed_dim) 
        else:
            self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,  # chain mask (B,T,T) for multimer
        self_attn_padding_mask: torch.Tensor = None,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # per chain positions for rotary
    ):
        """
        Forward pass of the transformer layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, embed_dim)
            self_attn_mask:
                - Standard mode: optional attention mask (e.g., causal)
            self_attn_padding_mask: Padding mask for key positions (B, T)
            need_head_weights: If True, return per‑head attention weights

        Returns:
            x: Output tensor of shape (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        # --- Attention block (pre‑norm) ---
        residual = x
        x = self.self_attn_layer_norm(x)

        # inter-chain only, attention only on cross pairs, mask that disables intra-chain
        intra_mask = (self_attn_mask == 0)  # now is true is same chain
        attn_mask = torch.where(intra_mask, 0.0, float("-inf"))
        x_attn, attn = self.self_attn(
            x=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            flash=True,
        )
        x = residual + x_attn
        # --- Feed‑forward block (pre‑norm) ---
        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, attn


class CrossAttentionLayer_fp8(nn.Module):
    def __init__(  # this is specifically for alternating layers
        self,  # accepts mask for same seq, attends to cross seq
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_swiglu: bool = False,
        use_rotary_embeddings: bool = True,
        tau: float = 0.1,  # default for munit scaling
        fp8: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.cross_attn = MultiHeadAttention_Opt(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
            fp8=True
        )

        # munit scaling calls for rees post norm
        self.self_attn_layer_norm = nn.RMSNorm(embed_dim)
        self.final_layer_norm = nn.RMSNorm(embed_dim)

        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(embed_dim, ffn_embed_dim, fp8) 
        else:
            self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim, fp8)
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,  # chain mask (B,T,T) for multimer
        self_attn_padding_mask: torch.Tensor = None,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # per chain positions for rotary
    ):
        """
        Forward pass of the transformer layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, embed_dim)
            self_attn_mask:
                - Standard mode: optional attention mask (e.g., causal)
            self_attn_padding_mask: Padding mask for key positions (B, T)
            need_head_weights: If True, return per‑head attention weights

        Returns:
            x: Output tensor of shape (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        # --- res post layernorm ---
        residual = x

        # inter-chain only, attention only on cross pairs, mask that disables intra-chain
        cross_mask = (self_attn_mask == 1)
        attn_mask = torch.where(cross_mask, 0.0, float("-inf"))
        x_attn, _ = self.cross_attn(
            x=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            flash=True,
        )
        # fixed residual modification
        x = (1.0 - self.tau)**0.5 * residual + (self.tau**0.5) * x_attn
        x = self.self_attn_layer_norm(x)
        residual = x
        x_ffn = self.feed_forward(x)
        x = (1.0 - self.tau)**0.5 * residual + (self.tau**0.5) * x_ffn
        x = self.final_layer_norm(x)  # post layer norm
        return x, None


class SelfAttentionLayer_fp8(nn.Module):
    def __init__(  # this is specifically for alternating layers
        self,  # accepts mask for cross seq, only attends to same seq
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_swiglu: bool = False,
        use_rotary_embeddings: bool = True,
        tau: float = 0.1,  # default for munit scaling
        fp8: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.self_attn = MultiHeadAttention_Opt(
            embed_dim,
            attention_heads,
            use_rotary_embeddings=use_rotary_embeddings,
            fp8=True,
        )

        # munit scaling calls for rees post norm
        self.self_attn_layer_norm = nn.RMSNorm(embed_dim)
        self.final_layer_norm = nn.RMSNorm(embed_dim)

        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(embed_dim, ffn_embed_dim, fp8) 
        else:
            self.feed_forward = VanillaFeedForward(embed_dim, ffn_embed_dim, fp8)
        
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,  # chain mask (B,T,T) for multimer
        self_attn_padding_mask: torch.Tensor = None,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,  # per chain positions for rotary
    ):
        """
        Forward pass of the transformer layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, embed_dim)
            self_attn_mask:
                - Standard mode: optional attention mask (e.g., causal)
            self_attn_padding_mask: Padding mask for key positions (B, T)
            need_head_weights: If True, return per‑head attention weights

        Returns:
            x: Output tensor of shape (T, B, E)
            attn: Attention weights (shape depends on need_head_weights)
        """
        # --- Attention block (pre‑norm) ---
        residual = x

        # inter-chain only, attention only on cross pairs, mask that disables intra-chain
        intra_mask = (self_attn_mask == 0)  # now is true is same chain
        attn_mask = torch.where(intra_mask, 0.0, float("-inf"))
        x_attn, _ = self.self_attn(
            x=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            flash=True,
        )
        # fixed residual modification
        x = (1.0 - self.tau)**0.5 * residual + (self.tau**0.5) * x_attn
        x = self.self_attn_layer_norm(x)  # post layer norm
        residual = x
        x_ffn = self.feed_forward(x)
        x = (1.0 - self.tau)**0.5 * residual + (self.tau**0.5) * x_ffn
        x = self.final_layer_norm(x)  # post layer norm
        return x, None
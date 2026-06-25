import math
import uuid
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mint_plus.models.rotary_embedding import RotaryEmbedding
from mint_plus.models.attention import IncrementalStateMixin
from mint_plus.models.munit_scaling import UnitScaledLinear


class MultiHeadAttention_Opt(nn.Module, IncrementalStateMixin):
    """
    Multi‑Head self‑attention with rotary embeddings and optional caching.
    Designed for ESM2 – no encoder‑decoder, no bias_kv, no add_zero_attn.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_rotary_embeddings: bool = False,
        no_proj: bool = False,
        fp8: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout

        if fp8:
            self.q_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
            self.k_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
            self.v_proj = UnitScaledLinear(embed_dim, embed_dim, bias=bias)
            self.out_proj = None if no_proj else UnitScaledLinear(embed_dim, embed_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = None if no_proj else nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = RotaryEmbedding(dim=self.head_dim) if use_rotary_embeddings else None
        self._init_parameters()

    def _init_parameters(self):
        gain = 1 / math.sqrt(2)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=gain)
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    def _flatten_heads(self, x: Tensor, seq_len: int, batch_size: int) -> Tensor:
        """
        Convert (seq_len, batch, embed_dim) -> (batch * num_heads, seq_len, head_dim)
        """
        return x.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        before_softmax: bool = False,
        position_ids: Optional[Tensor] = None,
        flash: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Setup shapes
        tgt_len, bsz, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        # 1. Project Q, K, V
        q = self.q_proj(x) #* self.scaling   # (T, B, E)
        k = self.k_proj(x)                    # (S, B, E)  (S = src_len)
        v = self.v_proj(x)

        # 2. Handle incremental caching (if used)
        if incremental_state is not None:
            saved = self.get_incremental_state(incremental_state, "attn_state")
            if saved is not None:
                prev_k = saved["prev_key"]  # (B, H, prev_len, head_dim)
                prev_v = saved["prev_value"]
                # Flatten current K/V and concatenate with cached
                cur_k = self._flatten_heads(k, tgt_len, bsz)          # (B*H, T, head_dim)
                cur_v = self._flatten_heads(v, tgt_len, bsz)
                prev_k_flat = prev_k.view(bsz * self.num_heads, -1, self.head_dim)
                prev_v_flat = prev_v.view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_k_flat, cur_k], dim=1)
                v = torch.cat([prev_v_flat, cur_v], dim=1)
            else:
                k = self._flatten_heads(k, -1, bsz)
                v = self._flatten_heads(v, -1, bsz)
            # Save for next step
            new_state = {
                "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
                "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            }
            self.set_incremental_state(incremental_state, "attn_state", new_state)
        else:
            # 3. Flatten batch and heads for all tensors
            q = self._flatten_heads(q, tgt_len, bsz)   # (B*H, T, head_dim)
            k = self._flatten_heads(k, -1, bsz)        # (B*H, S, head_dim)
            v = self._flatten_heads(v, -1, bsz)

        # 4. Apply rotary embeddings (in place, shapes unchanged)
        if self.rot_emb is not None:
            # position_ids: (B, T) -> expand to (B, H, T) -> flatten
            if position_ids is not None:
                # Expand to heads: (B, T) -> (B, 1, T) -> (B*H, T)
                pos_ids = position_ids[:, None, :].expand(-1, self.num_heads, -1)
                pos_ids = pos_ids.reshape(bsz * self.num_heads, tgt_len)
            else:
                # Fallback to default positions (0..T-1) per batch
                pos_ids = torch.arange(tgt_len, device=x.device)
                pos_ids = pos_ids[None, :].expand(bsz * self.num_heads, -1)
            q = self.rot_emb(q, pos_ids)
            k = self.rot_emb(k, pos_ids)
        
        if flash:
            attn_out, attn_probs = self.sdpa(q, k, v, attn_mask, key_padding_mask, tgt_len, bsz, embed_dim)

        else:
            attn_out, attn_probs = self.vanilla_attn(q, k, v, attn_mask, key_padding_mask, tgt_len, bsz, embed_dim, before_softmax)
            
        if self.out_proj is not None:
            attn_out = self.out_proj(attn_out)

        # 10. Return attention weights if requested
        attn_weights = None
        if need_weights:
            attn_weights = None if attn_probs is None else attn_probs.view(bsz, self.num_heads, tgt_len, -1)
            if not need_head_weights:
                attn_weights = None if attn_weights is None else attn_weights.mean(dim=1)
        return attn_out, attn_weights

    def vanilla_attn(self, q, k, v, attn_mask, key_padding_mask, tgt_len, bsz, embed_dim, before_softmax):
        # q,k,v are (B*H, T, head_dim)
        attn_logits = torch.bmm(q, k.transpose(1, 2))   # (B*H, T, S)
    
        # Reshape to separate batch and heads for easier masking
        attn_logits = attn_logits.view(bsz, self.num_heads, tgt_len, -1)   # (B, H, T, S)
    
        # Apply attention mask (if any)
        if attn_mask is not None:
            # attn_mask can be (B, T, S) or (T, S) or (B, H, T, S)
            if attn_mask.dim() == 3 and attn_mask.size(0) == bsz:
                # (B, T, S) -> (B, 1, T, S)  (broadcast over heads)
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.dim() == 2:
                # (T, S) -> (1, 1, T, S)
                attn_mask = attn_mask[None, None, :, :]
            # If already (B, H, T, S), use as is.
            attn_logits = attn_logits + attn_mask
    
        # Apply key padding mask (if any)
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> (B, 1, 1, S)
            attn_logits = attn_logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
    
        # Flatten back for softmax
        attn_logits = attn_logits.view(bsz * self.num_heads, tgt_len, -1)
    
        # Early return for before_softmax
        if before_softmax:
            # Reshape v similarly for consistency
            v = v.view(bsz, self.num_heads, -1, self.head_dim)
            return attn_logits.view(bsz, self.num_heads, tgt_len, -1), v
    
        # Softmax, dropout, weighted sum
        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).type_as(attn_logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_out = torch.bmm(attn_probs, v)          # (B*H, T, head_dim)
    
        # Output projection shape
        attn_out = attn_out.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return attn_out, attn_probs
        
    def sdpa(self, q, k, v, attn_mask, key_padding_mask, tgt_len, bsz, embed_dim):
        # Reshape for SDPA: (B, H, T, head_dim)
        q = q.view(bsz, self.num_heads, -1, self.head_dim)
        k = k.view(bsz, self.num_heads, -1, self.head_dim)
        v = v.view(bsz, self.num_heads, -1, self.head_dim)

        # Build boolean mask (True = mask out)
        mask = None
    
        # Convert additive attention mask (if any) to boolean and add head dimension
        if attn_mask is not None:
            # attn_mask could be (B, T, T) or (B, 1, T, T) or (B, H, T, T)
            if attn_mask.dim() == 3:                 # (B, T, T) -> (B, 1, T, T)
                attn_mask = attn_mask.unsqueeze(1)
            # Now attn_mask is (B, *, T, T) where * is 1 or H
            mask = (attn_mask == float('-inf'))      # boolean, same shape
    
        # Convert key padding mask (if any) to (B, 1, 1, T)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask[:, None, None, :]   # (B, 1, 1, T)
            if mask is None:
                mask = pad_mask
            else:
                # Broadcast pad_mask to match mask's dimensions
                # mask may have shape (B, 1, T, T) or (B, H, T, T)
                # pad_mask is (B, 1, 1, T) -> broadcastable to (B, H, T, T)
                mask = mask | pad_mask
        # Flash attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,#attn_mask_combined,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )
    
        # Merge heads and output shape (T, B, E)
        attn_out = attn_out.transpose(1, 2).contiguous().view(tgt_len, bsz, -1)
        return attn_out, None
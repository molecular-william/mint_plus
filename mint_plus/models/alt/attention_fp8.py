import math
import uuid
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import transformer_engine.pytorch as te  # Added TE import

from .rotary_embedding import RotaryEmbedding


class IncrementalStateMixin:
    """Lightweight incremental state management."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_key(self, key: str) -> str:
        return f"{self._incremental_state_id}.{key}"

    def get_incremental_state(self, incremental_state: Optional[Dict], key: str):
        if incremental_state is None:
            return None
        return incremental_state.get(self._get_full_key(key))

    def set_incremental_state(self, incremental_state: Optional[Dict], key: str, value):
        if incremental_state is not None:
            incremental_state[self._get_full_key(key)] = value
        return incremental_state


class MultiHeadAttention(nn.Module, IncrementalStateMixin):
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = te.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = te.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = te.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = None if no_proj else te.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = RotaryEmbedding(dim=self.head_dim) if use_rotary_embeddings else None
        self._init_parameters()

    def _init_parameters(self):
        gain = 2 ** -0.5
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
        return x.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1).contiguous()  # contiguous may lower peak memory

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        before_softmax: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Setup shapes
        tgt_len, bsz, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        # 1. Project Q, K, V
        q = self.q_proj(x) * self.scaling   # (T, B, E)
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
            q, k = self.rot_emb(q), self.rot_emb(k)

        # 5. Compute raw attention logits (now 3D tensors)
        attn_logits = torch.bmm(q.squeeze(), k.squeeze().transpose(1, 2))   # (B*H, T, S)

        # 6. Apply masks (if any)
        if attn_mask is not None:
            # attn_mask shape: (T, S) or (B*H, T, S) – broadcast to (B*H, T, S)
            attn_logits = attn_logits + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> expand to (B*H, 1, S) then to (B*H, T, S)
            attn_logits = attn_logits.view(bsz, self.num_heads, tgt_len, -1)
            attn_logits = attn_logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
            attn_logits = attn_logits.view(bsz * self.num_heads, tgt_len, -1)

        # 7. Early return for before_softmax (used by multimer attention)
        if before_softmax:
            attn_logits = attn_logits.view(bsz, self.num_heads, tgt_len, -1)
            v_heads = v.view(bsz, self.num_heads, -1, self.head_dim)
            return attn_logits, v_heads

        # 8. Softmax, dropout, weighted sum
        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).type_as(attn_logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_out = torch.bmm(attn_probs, v)          # (B*H, T, head_dim)

        # 9. Output projection
        attn_out = attn_out.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if self.out_proj is not None:
            attn_out = self.out_proj(attn_out)

        # 10. Return attention weights if requested
        attn_weights = None
        if need_weights:
            attn_weights = attn_probs.view(bsz, self.num_heads, tgt_len, -1)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=1)
        return attn_out, attn_weights
"""
Multi-pathway fused attention kernel with K/V reuse across query blocks.

Flash-Attention-2-style design: each program processes multiple query blocks
(QPP * BLOCK_TQ queries total) and reuses K/V tiles across all of them.
This reduces global memory reads of K_self, K_multi, V_self, V_multi by a
factor of QPP.

Architecture:
  Grid: (B, H, ceil(T / (BLOCK_TQ * QPP))) -- each program owns a query group
  Outer loop: over key blocks (T / BLOCK_TK iterations)
    Load K_self, K_multi, V_self, V_multi + mask for this key block
    For each query in group (vectorized via tl.dot):
      logits = Q @ K^T (both pathways)
      combine + online softmax + weighted sum
  Inner: vectorized tl.dot over all QPP * BLOCK_TQ queries simultaneously

RoPE is applied BEFORE calling this kernel.
QKV projection is done BEFORE calling this kernel.
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _fused_multi_pathway_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr,
    output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,  # query blocks per program
    DROPOUT_P: tl.constexpr,
    seed: int,  # NOT constexpr -- avoids recompilation when seed changes
    SQRT_SOFTMAX: tl.constexpr = False,  # muS: sqrt(softmax) for variance preservation
):
    """FA-2 style: one program handles QPP * BLOCK_TQ queries.

    K/V tiles are loaded once per key-block iteration and shared across
    all queries in the group via tl.dot. This eliminates the redundant
    K/V global reads that occur when each program handles 1 query block.
    """
    pid = tl.program_id(0)
    num_q_groups = tl.cdiv(T, BLOCK_TQ * QPP)
    qg_start = (pid % num_q_groups) * BLOCK_TQ * QPP
    h_idx = (pid // num_q_groups) % H
    b_idx = pid // (H * num_q_groups)

    offsets_tq = qg_start + tl.arange(0, BLOCK_TQ * QPP)
    mask_tq = offsets_tq < T
    offsets_tk_base = tl.arange(0, BLOCK_TK)
    offsets_d = tl.arange(0, D)
    mask_d = offsets_d < D

    # ---- 1. Load Q_self and Q_multi for ALL queries in group (loaded once) ----
    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )  # (BLOCK_TQ * QPP, D)

    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )  # (BLOCK_TQ * QPP, D)

    # ---- 2. Online-softmax state for ALL queries ----
    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    # ---- 3. Tile over key dimension (K/V loaded ONCE per tile for all NQ queries) ----
    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        # --- Load K, V for both pathways ---
        k_self = tl.load(
            k_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D)

        k_multi = tl.load(
            k_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D)

        v_self = tl.load(
            v_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D)

        v_multi = tl.load(
            v_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D)

        # --- Load chain mask for ALL queries against this key block ---
        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)  # (NQ, BLOCK_TK)

        # --- Vectorized dot products directly into FP32 accumulators (FMA) ---
        intra_logits = tl.dot(q_self, tl.trans(k_self),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))
        inter_logits = tl.dot(q_multi, tl.trans(k_multi),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))

        # Combine logits: mask=True -> inter, mask=False -> intra
        combined = tl.where(mask_block, inter_logits, intra_logits)  # (NQ, BLOCK_TK) fp32

        # --- Online softmax (all fp32, no conversion needed) ---
        block_max = tl.max(combined, axis=1)                     # (NQ,) fp32
        new_m = tl.maximum(m, block_max)                         # (NQ,) fp32
        old_scale = tl.exp(m - new_m)                            # (NQ,) fp32

        exp_corrected = tl.exp(combined - new_m[:, None])        # (NQ, BLOCK_TK) fp32
        sum_exp = tl.sum(exp_corrected, axis=1)                  # (NQ,)

        # Scale old accumulators
        if SQRT_SOFTMAX:
            # muS sqrt-softmax: acc = acc * sqrt(old_scale), d = d * old_scale + sum_exp
            sqrt_old_scale = tl.sqrt(old_scale)
            acc = acc * sqrt_old_scale[:, None]
        else:
            acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        # Split probs/coeffs by pathway and fuse weighted sum directly into acc
        if SQRT_SOFTMAX:
            # muS: coeffs = sqrt(softmax) = sqrt(exp_corrected) / sqrt(d)
            # But we accumulate sqrt(exp_corrected) and normalize at the end.
            sqrt_exp = tl.sqrt(exp_corrected)
            intra_coeffs = tl.where(~mask_block, sqrt_exp, 0.0).to(tl.bfloat16)
            inter_coeffs = tl.where(mask_block, sqrt_exp, 0.0).to(tl.bfloat16)
        else:
            intra_coeffs = tl.where(~mask_block, exp_corrected, 0.0).to(tl.bfloat16)
            inter_coeffs = tl.where(mask_block, exp_corrected, 0.0).to(tl.bfloat16)

        # Fused FMA: acc = tl.dot(probs, V, acc) saves one add instruction
        acc = tl.dot(intra_coeffs, v_self, acc)
        acc = tl.dot(inter_coeffs, v_multi, acc)
        m = new_m

    # ---- 4. Normalize ----
    d_safe = tl.where(d > 0, d, 1.0)
    if SQRT_SOFTMAX:
        output = acc / tl.sqrt(d_safe[:, None])
    else:
        output = acc / d_safe[:, None]

    # ---- 5. Dropout (fused) ----
    if DROPOUT_P > 0:
        phi = tl.rand(seed, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    # ---- 6. Store ----
    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def fused_multi_pathway_attention(
    q_self: torch.Tensor,
    k_self: torch.Tensor,
    v_self: torch.Tensor,
    q_multi: torch.Tensor,
    k_multi: torch.Tensor,
    v_multi: torch.Tensor,
    chain_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = True,
    qpp: int = None,
    sqrt_softmax: bool = False,
) -> torch.Tensor:
    """Multi-pathway fused attention with K/V reuse across query blocks.

    Takes pre-projected Q/K/V for both pathways. Caller is responsible for:
      1. Projecting Q/K/V via MultiHeadAttention.project_qkv_4d()
      2. Applying scaling to Q: q = q * (1/sqrt(D))
      3. Applying RoPE to Q_self and K_self

    When sqrt_softmax=True (muS mode), uses sqrt(softmax) coefficients for
    the attention-weighted sum. This preserves activation variance across
    sequence positions (see Proposition 2.1 in the muS paper).

    The `qpp` parameter controls how many BLOCK_TQ blocks each program
    processes. Higher QPP = fewer K/V global reads, more registers consumed.
    Default: QPP=8 at T>=1024, else QPP=4.

    Args:
        q/k/v_self: (B, H, T, D) contiguous bf16. RoPE applied to Q,K.
        q/k/v_multi: (B, H, T, D) contiguous bf16. No RoPE.
        chain_mask: (B, T, T) bool. True = different chains.
        dropout_p: probability (0.0 = disabled).
        training: whether in training mode.
        qpp: query blocks per program override.
        sqrt_softmax: use sqrt(softmax) instead of softmax (muS mode).

    Returns:
        (B, H, T, D) bf16, ready for output projection.
    """
    B, H, T, D = q_self.shape
    assert all(t.shape == q_self.shape for t in [k_self, v_self, q_multi, k_multi, v_multi])
    assert chain_mask.shape == (B, T, T)
    assert q_self.is_contiguous() and q_self.dtype == torch.bfloat16

    output = torch.empty_like(q_self)

    # Block sizes — ensure NQ >= 16 for BF16 Tensor Core constraint (M >= 16)
    if T >= 1024:
        BLOCK_TQ = 8
    elif T >= 512:
        BLOCK_TQ = 8
    else:
        BLOCK_TQ = 8

    if D > 32:
        BLOCK_TK = _next_power_of_2(min(64, T))
    else:
        BLOCK_TK = _next_power_of_2(min(128, T))

    # Query blocks per program: higher = fewer K/V reloads, more register pressure
    if qpp is not None:
        QPP = qpp
    elif T >= 1024:
        QPP = 8
    elif T >= 256:
        QPP = 4
    else:
        QPP = 2

    # Enforce minimum M=16 for BF16 Tensor Cores
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    NQ = BLOCK_TQ * QPP  # total queries per program
    num_q_groups = triton.cdiv(T, NQ)
    grid = (B * H * num_q_groups,)
    # Generate seed once per host call.
    # NOTE: Using deterministic seed 0 because (a) dropout is never enabled in
    # current training (dropout_p=0.0), and (b) torch.rand().item() causes a
    # torch._dynamo graph break inside torch.compile. If dropout is enabled
    # later, move seed generation outside the compiled region (e.g., generate
    # in the training loop and pass as a parameter).
    seed = 0

    _fused_multi_pathway_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0,
        seed=seed,
        SQRT_SOFTMAX=sqrt_softmax,
    )
    return output

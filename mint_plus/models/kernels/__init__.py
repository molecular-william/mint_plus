"""
Fused Triton kernel for the multimer combine step.

Replaces the 7-kernel sequence:
    torch.where + F.softmax + F.dropout + 2x masked_fill + 2x matmul

with a single fused kernel that reads intra/ inter logits and values,
computes the combined softmax, applies dropout, splits by pathway,
and produces the weighted sum -- all in registers / shared memory.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_multimer_combine_kernel(
    # Input pointers
    intra_logits_ptr, inter_logits_ptr, mask_ptr,
    intra_values_ptr, inter_values_ptr,
    # Output pointer
    output_ptr,
    # Strides for intra/inter logits: (B, H, T, T)
    stride_il_b, stride_il_h, stride_il_tq, stride_il_tk,
    stride_el_b, stride_el_h, stride_el_tq, stride_el_tk,
    # Strides for mask: (B, T, T)
    stride_m_b, stride_m_tq, stride_m_tk,
    # Strides for values: (B, H, T, D)
    stride_iv_b, stride_iv_h, stride_iv_t, stride_iv_d,
    stride_ev_b, stride_ev_h, stride_ev_t, stride_ev_d,
    # Strides for output: (B, H, T, D)
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    # Dropout (0.0 = disabled during validation)
    dropout_p: tl.constexpr,
    # Random seed for dropout
    seed: tl.constexpr,
    # Block sizes (compile-time)
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program handles one (batch, head, query_position) tuple.

    Each program:
      1. Loads the logit rows and mask row for this query position
      2. Combines via where (same-chain -> intra, cross-chain -> inter)
      3. Online softmax over the T dimension
      4. Dropout (fused)
      5. Splits probabilities by pathway (no extra mask write)
      6. Weighted sum over T, tiled over D
      7. Stores one output row (B, H, Tq, d_block)
    """
    pid = tl.program_id(0)

    # Decode pid -> (batch, head, query_index)
    # pid = B * H * tq + H * b + h  (column-major over tq)
    pid_tq = pid % T
    pid_h = (pid // T) % H
    pid_b = pid // (H * T)

    offsets_k = tl.arange(0, BLOCK_T)
    mask_k = offsets_k < T

    # ---- 1. Load mask row for this query position ----
    # mask is (B, T, T), bool
    mask_base = (
        mask_ptr
        + pid_b * stride_m_b
        + pid_tq * stride_m_tq
    )
    mask_row = tl.load(
        mask_base + offsets_k * stride_m_tk,
        mask=mask_k,
    ).to(tl.int1)

    # ---- 2. Load logit rows from both pathways ----
    # intra_logits: (B, H, T, T)
    il_base = (
        intra_logits_ptr
        + pid_b * stride_il_b
        + pid_h * stride_il_h
        + pid_tq * stride_il_tq
    )
    intra_row = tl.load(
        il_base + offsets_k * stride_il_tk,
        mask=mask_k,
    )

    # inter_logits: (B, H, T, T)
    el_base = (
        inter_logits_ptr
        + pid_b * stride_el_b
        + pid_h * stride_el_h
        + pid_tq * stride_el_tq
    )
    inter_row = tl.load(
        el_base + offsets_k * stride_el_tk,
        mask=mask_k,
    )

    # ---- 3. Combine: where same-chain use intra, cross-chain use inter ----
    combined = tl.where(mask_row, inter_row, intra_row)

    # ---- 4. Online softmax (in fp32 for numerical stability) ----
    m = tl.max(combined, axis=0)
    safe = combined - m
    exp = tl.exp(safe.to(tl.float32))
    sum_exp = tl.sum(exp, axis=0)
    probs = exp / sum_exp
    probs = probs.to(combined.dtype)

    # ---- 5. Fused dropout ----
    if dropout_p > 0:
        phi = tl.rand(seed, pid)
        keep = phi > dropout_p
        probs = tl.where(keep, probs / (1.0 - dropout_p), 0.0)

    # ---- 6. Split probs by pathway (no write, just masked multiply) ----
    intra_probs = tl.where(~mask_row, probs, 0.0)
    inter_probs = tl.where(mask_row, probs, 0.0)

    # ---- 7. Weighted sum, tiled over head dimension ----
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        # Load intra_values[b, h, :, d_block]: (T, BLOCK_D)
        iv_base = (
            intra_values_ptr
            + pid_b * stride_iv_b
            + pid_h * stride_iv_h
        )
        intra_v = tl.load(
            iv_base
            + offsets_k[:, None] * stride_iv_t
            + d_offsets[None, :] * stride_iv_d,
            mask=(mask_k[:, None] & d_mask[None, :]),
        )

        # Load inter_values[b, h, :, d_block]: (T, BLOCK_D)
        ev_base = (
            inter_values_ptr
            + pid_b * stride_ev_b
            + pid_h * stride_ev_h
        )
        inter_v = tl.load(
            ev_base
            + offsets_k[:, None] * stride_ev_t
            + d_offsets[None, :] * stride_ev_d,
            mask=(mask_k[:, None] & d_mask[None, :]),
        )

        # Dot product: sum over key dimension T
        # intra_probs: (T,), intra_v: (T, BLOCK_D) -> (BLOCK_D,)
        intra_out = tl.sum(intra_probs[:, None] * intra_v, axis=0)
        inter_out = tl.sum(inter_probs[:, None] * inter_v, axis=0)
        combined_out = intra_out + inter_out

        # Store output[b, h, tq, d_block]: (BLOCK_D,)
        o_base = (
            output_ptr
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + pid_tq * stride_o_t
        )
        tl.store(
            o_base + d_offsets * stride_o_d,
            combined_out.to(intra_v.dtype),
            mask=d_mask,
        )


def fused_multimer_combine(
    intra_logits: torch.Tensor,   # (B, H, T, T)
    inter_logits: torch.Tensor,   # (B, H, T, T)
    chain_mask: torch.Tensor,     # (B, T, T) bool
    intra_values: torch.Tensor,   # (B, H, T, D)
    inter_values: torch.Tensor,   # (B, H, T, D)
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Fused multimer combine -> single fused kernel.

    Args:
        intra_logits: (B, H, T, T) -- before-softmax logits from self_attn
        inter_logits: (B, H, T, T) -- before-softmax logits from multimer_attn
        chain_mask: (B, T, T) -- True where positions belong to different chains
        intra_values: (B, H, T, D) -- value vectors from self_attn
        inter_values: (B, H, T, D) -- value vectors from multimer_attn
        dropout_p: dropout probability (0.0 = disabled)

    Returns:
        (B, H, T, D) combined output, ready for out_proj.
    """
    B, H, T, D = intra_values.shape

    # Mask must be bool
    assert chain_mask.dtype == torch.bool, "chain_mask must be bool"

    output = torch.empty_like(intra_values)

    # Each program handles one (B, H, Tq) tuple
    grid = (B * H * T,)

    # Generate seed from current state if dropout is active
    seed = int(torch.rand(1).item() * 2**31) if dropout_p > 0 else 0

    # Triton compile-time constants
    BLOCK_T = triton.next_power_of_2(T)   # 512
    BLOCK_D = triton.next_power_of_2(D)   # 32

    # Autotune T-block to actual length for validation (where T may vary)
    # but for training T=512 is fixed
    BLOCK_T_actual = triton.next_power_of_2(T)

    fused_multimer_combine_kernel[grid](
        intra_logits, inter_logits, chain_mask,
        intra_values, inter_values,
        output,
        # Strides: (B, H, T, T) for logits
        *intra_logits.stride(),
        *inter_logits.stride(),
        # Strides: (B, T, T) for mask
        *chain_mask.stride(),
        # Strides: (B, H, T, D) for values
        *intra_values.stride(),
        *inter_values.stride(),
        # Strides: (B, H, T, D) for output
        *output.stride(),
        B=B, H=H, T=T, D=D,
        dropout_p=dropout_p,
        seed=seed,
        BLOCK_T=BLOCK_T_actual,
        BLOCK_D=BLOCK_D,
    )

    return output


# ---- Phase 2: Multi-pathway fused attention (fuses 2xbmm + combine + softmax) ----

def fused_multimer_combine_bwd(
    intra_logits: torch.Tensor, inter_logits: torch.Tensor,
    chain_mask: torch.Tensor,
    intra_values: torch.Tensor, inter_values: torch.Tensor,
    grad_output: torch.Tensor,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for fused_multimer_combine using a Triton kernel.

    Computes gradients for logits and values through the combined softmax.
    Since logits are already materialized (B, H, T, T), this kernel works
    on the full tensors -- one program per (batch, head, query).

    Returns:
        d_intra_logits, d_inter_logits, d_intra_values, d_inter_values
    """
    B, H, T, D = intra_values.shape
    d_intra_logits = torch.zeros_like(intra_logits)
    d_inter_logits = torch.zeros_like(inter_logits)
    d_intra_values = torch.zeros_like(intra_values, dtype=torch.float32)
    d_inter_values = torch.zeros_like(inter_values, dtype=torch.float32)

    grid = (B * H * T,)
    seed = 0

    _bwd_combine_kernel[grid](
        intra_logits, inter_logits, chain_mask,
        intra_values, inter_values, grad_output,
        d_intra_logits, d_inter_logits, d_intra_values, d_inter_values,
        intra_logits.stride(0), intra_logits.stride(1),
        intra_logits.stride(2), intra_logits.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        intra_values.stride(0), intra_values.stride(1),
        intra_values.stride(2), intra_values.stride(3),
        B=B, H=H, T=T, D=D,
        DROPOUT_P=dropout_p, SEED=seed,
    )
    return (d_intra_logits, d_inter_logits,
            d_intra_values.to(torch.bfloat16),
            d_inter_values.to(torch.bfloat16))


@triton.jit
def _bwd_combine_kernel(
    intra_l_ptr, inter_l_ptr, mask_ptr,
    intra_v_ptr, inter_v_ptr, dout_ptr,
    d_intra_l_ptr, d_inter_l_ptr,
    d_intra_v_ptr, d_inter_v_ptr,
    stride_lb, stride_lh, stride_ltq, stride_ltk,
    stride_mb, stride_mtq, stride_mtk,
    stride_vb, stride_vh, stride_vt, stride_vd,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
):
    """Backward: one program per (batch, head, query_position).

    Recomputes the forward (softmax, split), then computes gradients
    for logits (per-query, no conflict) and values (atomic_add).
    """
    pid = tl.program_id(0)
    tq = pid % T
    h = (pid // T) % H
    b = pid // (H * T)

    off_k = tl.arange(0, T)
    mask_k = off_k < T
    off_d = tl.arange(0, D)
    mask_d = off_d < D

    # ---- Load inputs for this query position ----
    intra_row = tl.load(
        intra_l_ptr + b * stride_lb + h * stride_lh
        + tq * stride_ltq + off_k * stride_ltk,
        mask=mask_k,
    )
    inter_row = tl.load(
        inter_l_ptr + b * stride_lb + h * stride_lh
        + tq * stride_ltq + off_k * stride_ltk,
        mask=mask_k,
    )
    mask_row = tl.load(
        mask_ptr + b * stride_mb + tq * stride_mtq + off_k * stride_mtk,
        mask=mask_k,
    ).to(tl.int1)

    # ---- Recompute forward: combine + softmax ----
    combined = tl.where(mask_row, inter_row, intra_row)
    m = tl.max(combined, axis=0)
    safe = combined - m
    exp = tl.exp(safe.to(tl.float32))
    sum_exp = tl.sum(exp, axis=0)
    probs = exp / sum_exp

    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        probs = tl.where(keep, probs / (1.0 - DROPOUT_P), 0.0)

    # Cast to bf16 for dot products
    probs_bf16 = probs.to(tl.bfloat16)

    # ---- Load grad_output for this query ----
    dout_row = tl.load(
        dout_ptr + b * stride_vb + h * stride_vh
        + tq * stride_vt + off_d * stride_vd,
        mask=mask_d,
    )

    # ---- dV: accumulate across queries via atomic_add ----
    # dV[k, d] += probs[k] * dout[d] (outer product, accumulated over queries)
    intra_probs = tl.where(~mask_row, probs, 0.0)
    inter_probs = tl.where(mask_row, probs, 0.0)
    intra_probs_bf16 = intra_probs.to(tl.bfloat16)
    inter_probs_bf16 = inter_probs.to(tl.bfloat16)

    # For each d, atomic-add dV[k, d] += probs[k] * dout[d]
    for d_start in range(0, D, 4):
        d_off = d_start + tl.arange(0, 4)
        d_mask_4 = d_off < D
        dval = tl.load(dout_ptr + b * stride_vb + h * stride_vh
                       + tq * stride_vt + d_off * stride_vd,
                       mask=d_mask_4)
        # intra: dV[k, d_start:d_start+4] += probs[k] * dval[d_start:d_start+4]
        dv_block = intra_probs_bf16[:, None] * dval[None, :]  # (T, 4)
        tl.atomic_add(
            d_intra_v_ptr + b * stride_vb + h * stride_vh
            + off_k[:, None] * stride_vt + d_off[None, :] * stride_vd,
            dv_block.to(tl.float32),
            mask=(mask_k[:, None] & d_mask_4[None, :]),
        )
        dv_block = inter_probs_bf16[:, None] * dval[None, :]  # (T, 4)
        tl.atomic_add(
            d_inter_v_ptr + b * stride_vb + h * stride_vh
            + off_k[:, None] * stride_vt + d_off[None, :] * stride_vd,
            dv_block.to(tl.float32),
            mask=(mask_k[:, None] & d_mask_4[None, :]),
        )

    # ---- dprobs = dout @ V^T (per-query, no conflict) ----
    # dprobs[k] = sum_d dout[d] * V[k, d]
    dprobs = tl.zeros((T,), dtype=tl.float32)
    for d_start in range(0, D, 4):
        d_off = d_start + tl.arange(0, 4)
        d_mask_4 = d_off < D
        v_intra = tl.load(
            intra_v_ptr + b * stride_vb + h * stride_vh
            + off_k[:, None] * stride_vt + d_off[None, :] * stride_vd,
            mask=(mask_k[:, None] & d_mask_4[None, :]),
        )
        v_inter = tl.load(
            inter_v_ptr + b * stride_vb + h * stride_vh
            + off_k[:, None] * stride_vt + d_off[None, :] * stride_vd,
            mask=(mask_k[:, None] & d_mask_4[None, :]),
        )
        dval = tl.load(dout_ptr + b * stride_vb + h * stride_vh
                       + tq * stride_vt + d_off * stride_vd,
                       mask=d_mask_4)
        # dprobs_intra[k] += dout[d] * V_intra[k, d]
        # dprobs_inter[k] += dout[d] * V_inter[k, d]
        dprobs += tl.sum(dval[None, :] * v_intra, axis=1)
        dprobs += tl.sum(dval[None, :] * v_inter, axis=1)

    # ---- Softmax backward ----
    # dlogits = probs * (dprobs - sum(probs * dprobs))
    p_f32 = probs
    dp_f32 = dprobs
    p_dp = p_f32 * dp_f32
    sum_p_dp = tl.sum(p_dp, axis=0)
    dlogits = p_f32 * (dp_f32 - sum_p_dp)
    dlogits = dlogits.to(intra_row.dtype)

    # ---- Split by mask for logit gradients ----
    d_intra = tl.where(~mask_row, dlogits, 0.0)
    d_inter = tl.where(mask_row, dlogits, 0.0)

    # ---- Store logit gradients (per-query, no conflict) ----
    tl.store(
        d_intra_l_ptr + b * stride_lb + h * stride_lh
        + tq * stride_ltq + off_k * stride_ltk,
        d_intra,
        mask=mask_k,
    )
    tl.store(
        d_inter_l_ptr + b * stride_lb + h * stride_lh
        + tq * stride_ltq + off_k * stride_ltk,
        d_inter,
        mask=mask_k,
    )

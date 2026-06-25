#!/usr/bin/env python3
"""
Benchmark each optimization suggestion individually vs the current baseline.

Tests at the training shape: B=32, T=1024, H=20, D=32 (150M frozen model).

Optimizations tested:
  A) Fused Q scaling (SM_SCALE passed as kernel param)
  B) FP32 accumulation via tl.dot(..., acc) for FMA
  C) Dynamic pathway skipping (tl.any checks to avoid unused K/V loads + matmuls)
  D) Dropout bug fix (per-element RNG offsets)
  E) All combined

Each variant is a JIT kernel with one thing changed.
"""

import os, sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mint_plus.models.attention import MultiHeadAttention, MultimerAttention
from mint_plus.models.kernels import fused_multimer_combine

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def time_kernel(fn, warmup=10, iters=50):
    """Time a kernel function in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def apply_rope_4d(q, k, rot_emb):
    B, H, T, D = q.shape
    q_3d = q.reshape(B * H, T, D)
    k_3d = k.reshape(B * H, T, D)
    q_r = rot_emb(q_3d)
    k_r = rot_emb(k_3d)
    return (q_r.view(B, H, T, D).contiguous(),
            k_r.view(B, H, T, D).contiguous())


# =====================================================================
# Variant 0: Baseline (current implementation)
# =====================================================================

@triton.jit
def _baseline_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr, output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
):
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

    # Load Q (scaling applied externally)
    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )

    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        # Load K, V for both pathways (always, no skipping)
        k_self = tl.load(
            k_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        k_multi = tl.load(
            k_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_self = tl.load(
            v_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_multi = tl.load(
            v_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )

        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        # Both pathways always computed
        intra_logits = tl.dot(q_self, tl.trans(k_self))
        inter_logits = tl.dot(q_multi, tl.trans(k_multi))

        combined = tl.where(mask_block, inter_logits, intra_logits)

        # Online softmax
        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max.to(tl.float32))
        old_scale = tl.exp(m - new_m)
        combined_f32 = combined.to(tl.float32)
        exp_corrected = tl.exp(combined_f32 - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        intra_probs = tl.where(~mask_block, exp_corrected, 0.0)
        inter_probs = tl.where(mask_block, exp_corrected, 0.0)

        intra_contrib = tl.dot(intra_probs.to(tl.bfloat16), v_self)
        inter_contrib = tl.dot(inter_probs.to(tl.bfloat16), v_multi)
        acc = acc + intra_contrib + inter_contrib
        m = new_m

    # Normalize
    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    # Dropout (block-uniform as in current code)
    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def baseline_attention(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                       chain_mask, dropout_p=0.0, training=True, qpp=None):
    B, H, T, D = q_self.shape
    output = torch.empty_like(q_self)

    BLOCK_TQ = 8 if T >= 1024 else 8
    BLOCK_TK = _next_power_of_2(min(64 if D > 32 else 128, T))
    QPP = qpp if qpp is not None else (8 if T >= 1024 else (4 if T >= 256 else 2))
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _baseline_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
    )
    return output


# =====================================================================
# Variant A: Fused Q scaling (SM_SCALE inside kernel)
# =====================================================================

@triton.jit
def _varA_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr, output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
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

    # Load Q and fuse scaling inside kernel
    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    if SM_SCALE != 1.0:
        q_self = q_self * SM_SCALE

    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    if SM_SCALE != 1.0:
        q_multi = q_multi * SM_SCALE

    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        k_self = tl.load(
            k_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        k_multi = tl.load(
            k_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_self = tl.load(
            v_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_multi = tl.load(
            v_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )

        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        intra_logits = tl.dot(q_self, tl.trans(k_self))
        inter_logits = tl.dot(q_multi, tl.trans(k_multi))
        combined = tl.where(mask_block, inter_logits, intra_logits)

        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max.to(tl.float32))
        old_scale = tl.exp(m - new_m)
        combined_f32 = combined.to(tl.float32)
        exp_corrected = tl.exp(combined_f32 - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        intra_probs = tl.where(~mask_block, exp_corrected, 0.0)
        inter_probs = tl.where(mask_block, exp_corrected, 0.0)

        intra_contrib = tl.dot(intra_probs.to(tl.bfloat16), v_self)
        inter_contrib = tl.dot(inter_probs.to(tl.bfloat16), v_multi)
        acc = acc + intra_contrib + inter_contrib
        m = new_m

    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def varA_attention(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                   chain_mask, dropout_p=0.0, training=True, qpp=None):
    B, H, T, D = q_self.shape
    output = torch.empty_like(q_self)

    BLOCK_TQ = 8 if T >= 1024 else 8
    BLOCK_TK = _next_power_of_2(min(64 if D > 32 else 128, T))
    QPP = qpp if qpp is not None else (8 if T >= 1024 else (4 if T >= 256 else 2))
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _varA_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
        SM_SCALE=1.0 / (D ** 0.5),
    )
    return output


# =====================================================================
# Variant B: FP32 accumulation with tl.dot(..., acc) for FMA
# =====================================================================

@triton.jit
def _varB_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr, output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
):
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

    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )

    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        k_self = tl.load(
            k_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        k_multi = tl.load(
            k_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_self = tl.load(
            v_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )
        v_multi = tl.load(
            v_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )

        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        # Logits computed directly into fp32 accumulator
        intra_logits = tl.dot(q_self, tl.trans(k_self),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))
        inter_logits = tl.dot(q_multi, tl.trans(k_multi),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))

        combined = tl.where(mask_block, inter_logits, intra_logits)

        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max)
        old_scale = tl.exp(m - new_m)

        exp_corrected = tl.exp(combined - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        intra_probs = tl.where(~mask_block, exp_corrected, 0.0).to(tl.bfloat16)
        inter_probs = tl.where(mask_block, exp_corrected, 0.0).to(tl.bfloat16)

        # Fused FMA: acc = tl.dot(a, b, acc)
        acc = tl.dot(intra_probs, v_self, acc)
        acc = tl.dot(inter_probs, v_multi, acc)

        m = new_m

    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def varB_attention(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                   chain_mask, dropout_p=0.0, training=True, qpp=None):
    B, H, T, D = q_self.shape
    output = torch.empty_like(q_self)

    BLOCK_TQ = 8 if T >= 1024 else 8
    BLOCK_TK = _next_power_of_2(min(64 if D > 32 else 128, T))
    QPP = qpp if qpp is not None else (8 if T >= 1024 else (4 if T >= 256 else 2))
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _varB_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
    )
    return output


# =====================================================================
# Variant C: Dynamic pathway skipping (tl.any checks)
# =====================================================================

@triton.jit
def _varC_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr, output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
):
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

    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )

    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        # Pre-initialize K/V to zeros in bf16 (matching the loaded dtype)
        # so Triton doesn't see a type mismatch across conditional branches
        zero_row = tl.full((BLOCK_TK, D), 0.0, dtype=tl.bfloat16)
        k_self = zero_row
        k_multi = zero_row
        v_self = zero_row
        v_multi = zero_row

        # Load mask FIRST to decide which pathways to compute
        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        # tl.any not available in Triton 3.3.1; use tl.max on int32
        any_intra = tl.max((~mask_block).to(tl.int32))
        any_inter = tl.max(mask_block.to(tl.int32))

        # Initialize logit accumulators in fp32
        intra_logits = tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32)
        inter_logits = tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32)

        # Conditionally load K/V and compute only needed pathways
        if any_intra:
            k_self = tl.load(
                k_self_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )
            intra_logits = tl.dot(q_self, tl.trans(k_self), intra_logits)

            v_self = tl.load(
                v_self_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )

        if any_inter:
            k_multi = tl.load(
                k_multi_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )
            inter_logits = tl.dot(q_multi, tl.trans(k_multi), inter_logits)

            v_multi = tl.load(
                v_multi_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )

        combined = tl.where(mask_block, inter_logits, intra_logits)

        # Online softmax (all in fp32 now)
        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max)
        old_scale = tl.exp(m - new_m)

        exp_corrected = tl.exp(combined - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        # Fused FMA into acc
        if any_intra:
            intra_probs = tl.where(~mask_block, exp_corrected, 0.0).to(tl.bfloat16)
            acc = tl.dot(intra_probs, v_self, acc)

        if any_inter:
            inter_probs = tl.where(mask_block, exp_corrected, 0.0).to(tl.bfloat16)
            acc = tl.dot(inter_probs, v_multi, acc)

        m = new_m

    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def varC_attention(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                   chain_mask, dropout_p=0.0, training=True, qpp=None):
    B, H, T, D = q_self.shape
    output = torch.empty_like(q_self)

    BLOCK_TQ = 8 if T >= 1024 else 8
    BLOCK_TK = _next_power_of_2(min(64 if D > 32 else 128, T))
    QPP = qpp if qpp is not None else (8 if T >= 1024 else (4 if T >= 256 else 2))
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _varC_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
    )
    return output


# =====================================================================
# Variant E: All combined (fused Q scale + FMA + pathway skipping + correct dropout)
# =====================================================================

@triton.jit
def _varE_kernel(
    q_self_ptr, k_self_ptr, v_self_ptr,
    q_multi_ptr, k_multi_ptr, v_multi_ptr,
    mask_ptr, output_ptr,
    stride_b, stride_h, stride_t, stride_d,
    stride_mb, stride_mtq, stride_mtk,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_TQ: tl.constexpr, BLOCK_TK: tl.constexpr,
    QPP: tl.constexpr,
    DROPOUT_P: tl.constexpr, SEED: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
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

    # Load Q and fuse scaling
    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    if SM_SCALE != 1.0:
        q_self = q_self * SM_SCALE

    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )
    if SM_SCALE != 1.0:
        q_multi = q_multi * SM_SCALE

    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        # Pre-initialize K/V to zeros in bf16 (matching the loaded dtype)
        # so Triton doesn't see a type mismatch across conditional branches
        zero_row = tl.full((BLOCK_TK, D), 0.0, dtype=tl.bfloat16)
        k_self = zero_row
        k_multi = zero_row
        v_self = zero_row
        v_multi = zero_row

        # Load mask first to decide pathway skipping
        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        # tl.any not available in Triton 3.3.1; use tl.max on int32
        any_intra = tl.max((~mask_block).to(tl.int32))
        any_inter = tl.max(mask_block.to(tl.int32))

        intra_logits = tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32)
        inter_logits = tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32)

        if any_intra:
            k_self = tl.load(
                k_self_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )
            intra_logits = tl.dot(q_self, tl.trans(k_self), intra_logits)
            v_self = tl.load(
                v_self_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )

        if any_inter:
            k_multi = tl.load(
                k_multi_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )
            inter_logits = tl.dot(q_multi, tl.trans(k_multi), inter_logits)
            v_multi = tl.load(
                v_multi_ptr + b_idx * stride_b + h_idx * stride_h
                + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
                mask=(mask_tk[:, None] & mask_d[None, :]),
            )

        combined = tl.where(mask_block, inter_logits, intra_logits)

        # Online softmax in fp32
        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max)
        old_scale = tl.exp(m - new_m)

        exp_corrected = tl.exp(combined - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        # Fused FMA into acc
        if any_intra:
            intra_probs = tl.where(~mask_block, exp_corrected, 0.0).to(tl.bfloat16)
            acc = tl.dot(intra_probs, v_self, acc)

        if any_inter:
            inter_probs = tl.where(mask_block, exp_corrected, 0.0).to(tl.bfloat16)
            acc = tl.dot(inter_probs, v_multi, acc)

        m = new_m

    # Normalize
    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    # Correct element-wise dropout
    if DROPOUT_P > 0:
        rng_offsets = offsets_tq[:, None] * D + offsets_d[None, :]
        phi = tl.rand(SEED, rng_offsets)
        keep = phi > DROPOUT_P
        output = tl.where(keep, output / (1.0 - DROPOUT_P), 0.0)

    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.bfloat16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def varE_attention(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                   chain_mask, dropout_p=0.0, training=True, qpp=None):
    B, H, T, D = q_self.shape
    output = torch.empty_like(q_self)

    BLOCK_TQ = 8 if T >= 1024 else 8
    BLOCK_TK = _next_power_of_2(min(64 if D > 32 else 128, T))
    QPP = qpp if qpp is not None else (8 if T >= 1024 else (4 if T >= 256 else 2))
    while BLOCK_TQ * QPP < 16:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _varE_kernel[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
        SM_SCALE=1.0 / (D ** 0.5),
    )
    return output


# =====================================================================
# Main: Benchmark all variants
# =====================================================================

def benchmark_variant(name, kernel_fn, B, T, H, D, sa, ma, chain_mask, x,
                      num_iters=50, dropout_p=0.0, training=False):
    E = H * D
    scaling = D ** -0.5

    # Prepare inputs exactly as the training pipeline does
    qs, ks, vss = sa.project_qkv_4d(x)
    qm, km, vmm = ma.project_qkv_4d(x)

    # For all variants: RoPE is applied externally (kernel can't do it since
    # it needs cos/sin tables). The SM_SCALE kernel variant only fuses the
    # scaling factor, not RoPE.
    qs_rope, ks_rope = apply_rope_4d(qs, ks, sa.rot_emb)

    # Apply scaling (for variants without SM_SCALE, do it externally)
    # Variants with SM_SCALE handle scaling inside the kernel
    has_sm_scale = 'varA' in name or 'varE' in name
    if has_sm_scale:
        qs_kernel = qs_rope
        qm_kernel = qm
    else:
        qs_kernel = qs_rope * scaling
        qm_kernel = qm * scaling

    # Correctness: compare to reference pipeline
    ls, vs = sa.forward_before_softmax(x)
    lm, vm = ma.forward(x)
    ref = fused_multimer_combine(ls, lm, chain_mask, vs, vm, dropout_p=dropout_p)

    new = kernel_fn(qs_kernel, ks_rope, vss, qm_kernel, km, vmm, chain_mask,
                    dropout_p=dropout_p, training=training)

    diff = (ref.float() - new.float()).abs()
    max_diff = diff.max().item()
    has_nan = torch.isnan(new).any().item()
    # Correctness threshold: 0.01 for same-computation-path variants,
    # 0.1 for variants where scaling is done inside vs outside the kernel (bf16 rounding)
    tol = 0.1 if 'varA' in name or 'varE' in name else 0.01
    passed = max_diff < tol and not has_nan

    # Time just the kernel call
    def kernel_only():
        kernel_fn(qs_kernel, ks_rope, vss, qm_kernel, km, vmm, chain_mask,
                  dropout_p=dropout_p, training=training)

    t_ms = time_kernel(kernel_only, warmup=10, iters=num_iters)

    return {'name': name, 'ms': t_ms, 'max_diff': max_diff, 'passed': passed,
            'has_nan': has_nan}


def run_benchmarks(num_iters=50):
    B, T, H, D = 32, 1024, 20, 32
    E = H * D

    print(f"Benchmarking at B={B}, T={T}, H={H}, D={D} (E={E})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    sa = MultiHeadAttention(E, H, use_rotary_embeddings=True).to(DEVICE).to(DTYPE)
    ma = MultimerAttention(E, H).to(DEVICE).to(DTYPE)
    sa.eval()
    ma.eval()

    x = torch.randn(T, B, E, device=DEVICE, dtype=DTYPE)
    chain_mask = torch.rand(B, T, T, device=DEVICE) < 0.4
    chain_mask = chain_mask & ~torch.eye(T, device=DEVICE, dtype=torch.bool).unsqueeze(0)

    variants = [
        ("0: Baseline (current)", baseline_attention),
        ("B: FP32 acc + tl.dot(...,acc) FMA", varB_attention),
        ("C: Dynamic pathway skipping", varC_attention),
        ("E: B + C + correct dropout + fused Q scale", varE_attention),
    ]

    results = []
    for name, fn in variants:
        r = benchmark_variant(name, fn, B, T, H, D, sa, ma, chain_mask, x,
                              num_iters=num_iters)
        results.append(r)

        status = "PASS" if r['passed'] else "FAIL"
        nan_str = " (NaN!)" if r['has_nan'] else ""
        print(f"  {name:<50s}  {r['ms']:>7.3f} ms  diff={r['max_diff']:.6f}  {status}{nan_str}")

    base_ms = results[0]['ms']
    print(f"\n  Relative to baseline ({base_ms:.3f} ms):")
    for r in results[1:]:
        if r['passed']:
            change = (base_ms - r['ms']) / base_ms * 100
            print(f"    {r['name']:<50s}  {r['ms']:>7.3f} ms  ({change:+.1f}%)")
        else:
            print(f"    {r['name']:<50s}  FAILED (max_diff={r['max_diff']:.6f})")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print()
    run_benchmarks(num_iters=50)

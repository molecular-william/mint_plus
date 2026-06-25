"""FP16 multi-pathway fused attention for Turing GPUs (CC 7.5).

Strategy: Load Q/K/V as fp16 (native pointer type), use tl.dot with fp16
operands (fp32 accumulator implicitly), keep online softmax in fp32,
store output as fp16. This keeps shared memory tiles at 2 bytes/element
so they fit in Turing's 64 KB shared memory limit.
"""
import torch
import triton
import triton.language as tl


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _fused_multi_pathway_kernel_fp16(
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

    # ---- 1. Load Q in native fp16 (Tensor Core operands) ----
    q_self = tl.load(
        q_self_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )  # (BLOCK_TQ*QPP, D) fp16

    q_multi = tl.load(
        q_multi_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )  # fp16

    # ---- 2. Online-softmax state (FP32) ----
    m = tl.full((BLOCK_TQ * QPP,), float("-inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_TQ * QPP,), dtype=tl.float32)

    # fp32 accumulator for weighted sum
    acc = tl.zeros((BLOCK_TQ * QPP, D), dtype=tl.float32)

    # ---- 3. Tile over key dimension ----
    for tk_start in range(0, T, BLOCK_TK):
        offsets_tk = tk_start + offsets_tk_base
        mask_tk = offsets_tk < T

        # Load K, V in native fp16
        k_self = tl.load(
            k_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D) fp16

        k_multi = tl.load(
            k_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # fp16

        v_self = tl.load(
            v_self_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # (BLOCK_TK, D) fp16

        v_multi = tl.load(
            v_multi_ptr + b_idx * stride_b + h_idx * stride_h
            + offsets_tk[:, None] * stride_t + offsets_d[None, :] * stride_d,
            mask=(mask_tk[:, None] & mask_d[None, :]),
        )  # fp16

        # Load mask
        mask_block = tl.load(
            mask_ptr + b_idx * stride_mb
            + offsets_tq[:, None] * stride_mtq
            + offsets_tk[None, :] * stride_mtk,
            mask=(mask_tq[:, None] & mask_tk[None, :]),
        ).to(tl.int1)

        # Dot products: fp16 inputs (native Tensor Cores), fp32 accumulate
        intra_logits = tl.dot(q_self, tl.trans(k_self),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))
        inter_logits = tl.dot(q_multi, tl.trans(k_multi),
                              acc=tl.zeros((BLOCK_TQ * QPP, BLOCK_TK), dtype=tl.float32))

        # Combine
        combined = tl.where(mask_block, inter_logits, intra_logits)  # fp32

        # Online softmax (fp32 for numerical stability)
        block_max = tl.max(combined, axis=1)
        new_m = tl.maximum(m, block_max)
        old_scale = tl.exp(m - new_m)

        exp_corrected = tl.exp(combined - new_m[:, None])
        sum_exp = tl.sum(exp_corrected, axis=1)

        acc = acc * old_scale[:, None]
        d = d * old_scale + sum_exp

        # Split probs, convert to fp16 for Tensor Core weighted sum
        intra_probs = tl.where(~mask_block, exp_corrected, 0.0).to(tl.float16)
        inter_probs = tl.where(mask_block, exp_corrected, 0.0).to(tl.float16)

        # Fused FMA with fp16 operands, fp32 accumulate
        acc = tl.dot(intra_probs, v_self, acc)
        acc = tl.dot(inter_probs, v_multi, acc)

        m = new_m

    # ---- 4. Normalize ----
    d_safe = tl.where(d > 0, d, 1.0)
    output = acc / d_safe[:, None]

    # ---- 5. Dropout ----
    if DROPOUT_P > 0:
        phi = tl.rand(SEED, pid)
        keep = phi > DROPOUT_P
        output = tl.where(keep[:, None], output / (1.0 - DROPOUT_P), 0.0)

    # ---- 6. Store as fp16 ----
    tl.store(
        output_ptr + b_idx * stride_b + h_idx * stride_h
        + offsets_tq[:, None] * stride_t + offsets_d[None, :] * stride_d,
        output.to(tl.float16),
        mask=(mask_tq[:, None] & mask_d[None, :]),
    )


def fused_multi_pathway_attention_fp16(
    q_self: torch.Tensor, k_self: torch.Tensor, v_self: torch.Tensor,
    q_multi: torch.Tensor, k_multi: torch.Tensor, v_multi: torch.Tensor,
    chain_mask: torch.Tensor, dropout_p: float = 0.0,
    training: bool = True, qpp: int = None,
) -> torch.Tensor:
    """Multi-pathway fused attention for fp16 (Turing-compatible, CC 7.5).

    Uses fp16 for Tensor Core matmuls (native Turing support) and fp32 for
    online softmax + accumulator. Compatible with 64 KB shared memory limit.
    """
    B, H, T, D = q_self.shape
    assert all(t.shape == q_self.shape for t in [k_self, v_self, q_multi, k_multi, v_multi])
    assert chain_mask.shape == (B, T, T)
    assert q_self.is_contiguous() and q_self.dtype == torch.float16

    output = torch.empty_like(q_self)

    # Block sizes — fp16 tensor cores work with M >= 8
    BLOCK_TQ = 8
    if D > 32:
        BLOCK_TK = _next_power_of_2(min(64, T))
    else:
        BLOCK_TK = _next_power_of_2(min(128, T))

    if qpp is not None:
        QPP = qpp
    elif T >= 1024:
        QPP = 8
    elif T >= 256:
        QPP = 4
    else:
        QPP = 2

    while BLOCK_TQ * QPP < 8:
        BLOCK_TQ *= 2

    num_q_groups = triton.cdiv(T, BLOCK_TQ * QPP)
    grid = (B * H * num_q_groups,)
    seed = int(torch.rand(1).item() * 2**31) if (dropout_p > 0 and training) else 0

    _fused_multi_pathway_kernel_fp16[grid](
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, output,
        q_self.stride(0), q_self.stride(1), q_self.stride(2), q_self.stride(3),
        chain_mask.stride(0), chain_mask.stride(1), chain_mask.stride(2),
        B=B, H=H, T=T, D=D,
        BLOCK_TQ=BLOCK_TQ, BLOCK_TK=BLOCK_TK, QPP=QPP,
        DROPOUT_P=dropout_p if training else 0.0, SEED=seed,
    )
    return output

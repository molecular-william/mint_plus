"""Autograd-compatible wrapper for the fused multi-pathway attention kernel.

The raw Triton kernel (@triton.jit) is not differentiable -- PyTorch autograd
cannot trace through it. This wrapper provides a backward function using native
PyTorch ops so gradients flow through the attention combine step.

During forward: uses the fast Triton kernel.
During backward: re-computes using native PyTorch ops (differentiable).
"""
import torch
from torch.autograd import Function


class _MultiPathwayAttentionFn(Function):
    """Differentiable wrapper around fused_multi_pathway_attention.

    Forward: delegates to the fast Triton kernel.
    Backward: re-computes using native PyTorch matmuls and softmax so
    gradients flow to Q/K/V/O projections.
    """

    @staticmethod
    def forward(ctx, q_self, k_self, v_self, q_multi, k_multi, v_multi,
                chain_mask, dropout_p, training, qpp):
        # Save for backward
        ctx.save_for_backward(q_self, k_self, v_self, q_multi, k_multi, v_multi,
                              chain_mask)
        ctx.dropout_p = dropout_p
        ctx.training = training

        # Forward via Triton kernel
        from mint_plus.models.kernels.multi_pathway_attention import \
            fused_multi_pathway_attention
        output = fused_multi_pathway_attention(
            q_self, k_self, v_self, q_multi, k_multi, v_multi,
            chain_mask, dropout_p=dropout_p, training=training, qpp=qpp,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q_self, k_self, v_self, q_multi, k_multi, v_multi, chain_mask = \
            ctx.saved_tensors

        # Use the Triton backward kernel (tiled, no (B,H,T,T) materialization)
        from mint_plus.models.kernels.multi_pathway_attention_bwd import (
            fused_multi_pathway_attention_bwd)
        dq_self, dk_self, dv_self, dq_multi, dk_multi, dv_multi = \
            fused_multi_pathway_attention_bwd(
                q_self, k_self, v_self, q_multi, k_multi, v_multi,
                chain_mask, grad_output,
            )

        return (dq_self, dk_self, dv_self, dq_multi, dk_multi, dv_multi,
                None, None, None, None)


# Public API -- drop-in replacement for fused_multi_pathway_attention
def differentiable_multi_pathway_attention(
    q_self, k_self, v_self, q_multi, k_multi, v_multi,
    chain_mask, dropout_p=0.0, training=True, qpp=8,
):
    """Same interface as fused_multi_pathway_attention but with autograd support.

    During forward: uses the fast Triton kernel.
    During backward: uses the Triton backward kernel (tiled, atomic_add).
    """
    return _MultiPathwayAttentionFn.apply(
        q_self, k_self, v_self, q_multi, k_multi, v_multi,
        chain_mask, dropout_p, training, qpp,
    )


# ---- Autograd wrapper for fused_multimer_combine ----

class _MultimerCombineFn(Function):
    """Differentiable wrapper around fused_multimer_combine.

    Forward: delegates to the fast Triton kernel.
    Backward: uses the new Triton backward kernel with atomic_add.
    """

    @staticmethod
    def forward(ctx, intra_logits, inter_logits, chain_mask,
                intra_values, inter_values, dropout_p):
        ctx.save_for_backward(intra_logits, inter_logits, chain_mask,
                              intra_values, inter_values)
        ctx.dropout_p = dropout_p

        from mint_plus.models.kernels import fused_multimer_combine
        output = fused_multimer_combine(
            intra_logits, inter_logits, chain_mask,
            intra_values, inter_values,
            dropout_p=dropout_p,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        intra_logits, inter_logits, chain_mask, intra_values, inter_values = \
            ctx.saved_tensors
        dropout_p = ctx.dropout_p

        from mint_plus.models.kernels import fused_multimer_combine_bwd
        d_il, d_el, d_iv, d_ev = fused_multimer_combine_bwd(
            intra_logits, inter_logits, chain_mask,
            intra_values, inter_values, grad_output,
            dropout_p=dropout_p,
        )
        return d_il, d_el, None, d_iv, d_ev, None


def differentiable_multimer_combine(
    intra_logits, inter_logits, chain_mask,
    intra_values, inter_values,
    dropout_p=0.0,
):
    """Same interface as fused_multimer_combine but with autograd support.

    Forward: uses the fast Triton kernel.
    Backward: uses the Triton backward kernel with atomic_add.
    """
    return _MultimerCombineFn.apply(
        intra_logits, inter_logits, chain_mask,
        intra_values, inter_values, dropout_p,
    )

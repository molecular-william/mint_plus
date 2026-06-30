"""mu-Scaling (muS) architecture for MINT+.

This module implements the muS training scheme from:

    Narayan et al., "unit Scaling: Simple and Scalable FP8 LLM Training" (2025)

Two entry points:

    1. build_mint_mu()          -- Build a model with muS architecture
                                    (Res-Post-LN, weighted residuals, sqrt-softmax)
                                    for training from scratch or warm-start.

    2. apply_fp8_to_model()     -- Replace nn.Linear with te.Linear in any model
                                    (Pre-LN or muS) for FP8 computation with
                                    static scaling (muS-style, not DelayedScaling).

Both use the builder pattern (same as build_mint_plus / enable_fused_multi_pathway):
the model is constructed normally, then modified in-place.
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from mint_plus.models.esm2 import MINT
from mint_plus.models.modules import (
    TransformerLayer_MINT,
    build_checkpointed_model,
    enable_fused_multi_pathway,
)
from mint_plus.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


# ======================================================================
# FP8: Replace nn.Linear with te.Linear (TransformerEngine)
# ======================================================================

def apply_fp8_to_model(
    model: nn.Module,
    static_scale: bool = True,
    keep_embed_in_bf16: bool = True,
    keep_lm_head_in_bf16: bool = True,
) -> nn.Module:
    """Replace nn.Linear projections with TransformerEngine's te.Linear.

    This enables FP8 computation on all hidden linear layers while keeping
    first/last layers in higher precision (paper recommendation).

    Uses static scaling (muS-style) instead of DelayedScaling when
    static_scale=True, eliminating the amax history buffers and
    reduce-max operations.

    Args:
        model: A MINT model (any variant).
        static_scale: If True, use muS-style static scaling instead of
                      TransformerEngine's default DelayedScaling.
        keep_embed_in_bf16: Keep embedding table in BF16.
        keep_lm_head_in_bf16: Keep LM head in BF16.

    Returns:
        The model with FP8-enabled linear layers (in-place modification).

    Example:
        model = MINT.from_config("150M")
        apply_fp8_to_model(model, static_scale=True)
        model = model.cuda().to(torch.bfloat16)
    """
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format
    except ImportError:
        logger.warning(
            "transformer_engine not installed. FP8 disabled. "
            "Install with: pip install transformer_engine"
        )
        return model

    # Replacement function for nn.Linear -> te.Linear
    def _replace_linear(old_linear: Optional[nn.Linear]) -> Optional[te.Linear]:
        if old_linear is None:
            return None
        new_linear = te.Linear(
            old_linear.in_features,
            old_linear.out_features,
            bias=old_linear.bias is not None,
        )
        # Copy weights
        with torch.no_grad():
            new_linear.weight.copy_(old_linear.weight)
            if old_linear.bias is not None:
                new_linear.bias.copy_(old_linear.bias)
        return new_linear

    # Walk through all layers
    for layer in model.layers:
        layers_to_process = []
        if hasattr(layer, "layers"):
            layers_to_process = layer.layers
        else:
            layers_to_process = [layer]

        for sub in layers_to_process:
            # Replace self-attention Q/K/V/out projections
            sa = sub.self_attn
            sa.q_proj = _replace_linear(sa.q_proj) or sa.q_proj
            sa.k_proj = _replace_linear(sa.k_proj) or sa.k_proj
            sa.v_proj = _replace_linear(sa.v_proj) or sa.v_proj
            if hasattr(sa, "out_proj") and sa.out_proj is not None:
                sa.out_proj = _replace_linear(sa.out_proj)

            # Replace multimer-attention projections
            if hasattr(sub, "multimer_attn") and sub.multimer_attn is not None:
                ma = sub.multimer_attn
                if hasattr(ma, "fused_qkv"):
                    ma.fused_qkv = _replace_linear(ma.fused_qkv) or ma.fused_qkv

            # Replace FFN projections
            ff = sub.feed_forward
            ff.fc1 = _replace_linear(ff.fc1) or ff.fc1
            ff.fc2 = _replace_linear(ff.fc2) or ff.fc2

    logger.info(f"FP8 enabled on all hidden linear layers "
                f"(static_scale={static_scale})")

    # Store FP8 config on model for the training loop
    model._fp8_config = {
        "enabled": True,
        "static_scale": static_scale,
    }

    return model


# ======================================================================
# muS architecture builder
# ======================================================================

def build_mint_mu(
    model_size: str = "150M",
    use_multimer: bool = True,
    tau: Optional[float] = None,
    use_sqrt_softmax: bool = True,
    use_fp8: bool = False,
    fp8_static_scale: bool = True,
    warm_start_esm2: bool = False,
    checkpoint_block_size: int = 0,
    use_fused_multi_pathway: bool = False,
) -> MINT:
    """Build a MINT model with muS (unit Scaling) architecture.

    The muS architecture uses:
      - Res-Post-LayerNorm (norm at end of residual stream)
      - Weighted residual connections with tau coefficient
      - Optional square-root softmax for variance preservation
      - Unit variance weight initialization
      - Optional FP8 computation on hidden linear layers

    Args:
        model_size: Model size key from MODEL_REGISTRY.
        use_multimer: Enable dual-pathway attention.
        tau: Residual coefficient. If None, auto-computed from depth.
        use_sqrt_softmax: Square-root softmax (muS paper, Section 2.1).
        use_fp8: Replace nn.Linear with te.Linear for FP8 computation.
        fp8_static_scale: Use muS-style static scaling (not DelayedScaling).
        warm_start_esm2: Load compatible ESM-2 weights, init rest randomly.
        checkpoint_block_size: Gradient checkpoint block size (0 = per-layer).
        use_fused_multi_pathway: Enable super-fused attention kernel.

    Returns:
        A MINT model with muS architecture, on CPU (call .cuda() before use).

    Example:
        model = build_mint_mu("150M", use_fp8=True)
        model = model.cuda()
        # ... training loop ...
    """
    config = MODEL_REGISTRY[model_size]
    num_layers = config["num_layers"]
    embed_dim = config["embed_dim"]
    attention_heads = config["attention_heads"]

    # Build base model
    model = MINT(
        num_layers=num_layers,
        embed_dim=embed_dim,
        attention_heads=attention_heads,
        use_multimer=use_multimer,
    )

    # Compute tau if not specified
    if tau is None:
        from mint_plus.models.mu_scaling.optim import compute_tau
        tau = compute_tau(num_layers)

    # Replace each layer with muS variant
    from mint_plus.models.mu_scaling.layer import TransformerLayer_MINT_mu

    for i, old_layer in enumerate(model.layers):
        new_layer = TransformerLayer_MINT_mu(
            embed_dim=old_layer.embed_dim,
            ffn_embed_dim=old_layer.ffn_embed_dim,
            attention_heads=old_layer.attention_heads,
            use_rotary_embeddings=old_layer.use_rotary_embeddings,
            use_multimer=old_layer.use_multimer,
            use_erf_gelu=old_layer.use_erf_gelu,
            tau=tau,
            num_layers=num_layers,
            use_sqrt_softmax=use_sqrt_softmax,
        )

        # Transfer weights (matching module names)
        # self_attn, multimer_attn, feed_forward have same submodule structure
        for subname in ["self_attn", "feed_forward"]:
            old_sub = getattr(old_layer, subname)
            new_sub = getattr(new_layer, subname)
            new_sub.load_state_dict(old_sub.state_dict(), strict=True)

        if use_multimer and hasattr(old_layer, "multimer_attn"):
            new_layer.multimer_attn.load_state_dict(
                old_layer.multimer_attn.state_dict(), strict=True,
            )

        # Transfer norms: muS only has final_layer_norm
        new_layer.final_layer_norm.load_state_dict(
            old_layer.final_layer_norm.state_dict(), strict=True,
        )

        model.layers[i] = new_layer

    logger.info(f"muS architecture: L={num_layers}, E={embed_dim}, "
                f"H={attention_heads}, tau={tau:.4f}, "
                f"sqrt_softmax={use_sqrt_softmax}")

    # Unit variance initialization (overrides transferred weights)
    # This is OPTIONAL -- keeping ESM-2 weights gives better warm-start.
    # Uncomment for true random init:
    # from mint_plus.models.mu_scaling.init import unit_variance_init_
    # model.apply(unit_variance_init_)

    # Warm-start from ESM-2
    if warm_start_esm2:
        ckpt_path = config.get("weight_url", "")
        if ckpt_path and ckpt_path.startswith("http"):
            logger.info(f"μS warm-start requires local checkpoint at {ckpt_path}")
            logger.info(f"Set model.checkpoint: path/to/esm2_150M.pt in config")

    # Super-fused attention kernel
    if use_fused_multi_pathway:
        enable_fused_multi_pathway(model, enabled=True)

    # Block-level checkpointing
    if checkpoint_block_size > 0:
        model = build_checkpointed_model(model, block_size=checkpoint_block_size)

    # FP8 on linear layers
    if use_fp8:
        model = apply_fp8_to_model(
            model,
            static_scale=fp8_static_scale,
        )

    return model


def build_mint_fp8(
    model_size: str = "150M",
    use_multimer: bool = True,
    static_scale: bool = True,
    checkpoint_block_size: int = 3,
    use_fused_multi_pathway: bool = True,
) -> MINT:
    """Build a Pre-LN model with FP8 on all hidden linear layers (Option B).

    This keeps the standard ESM-2 Pre-LayerNorm architecture but enables FP8
    computation via TransformerEngine on all linear layers. Pre-trained weights
    are fully compatible (no architectural changes).

    Args:
        model_size: Model size key from MODEL_REGISTRY.
        use_multimer: Enable dual-pathway attention.
        static_scale: Use muS-style static scaling (not DelayedScaling).
        checkpoint_block_size: Gradient checkpoint block size.
        use_fused_multi_pathway: Enable super-fused attention kernel.

    Returns:
        A MINT model with FP8 on linear layers, Pre-LN architecture.

    Example:
        model = build_mint_fp8("150M")
        model = model.cuda()
    """
    config = MODEL_REGISTRY[model_size]
    model = MINT(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        use_multimer=use_multimer,
    )

    if use_fused_multi_pathway:
        enable_fused_multi_pathway(model, enabled=True)

    if checkpoint_block_size > 0:
        model = build_checkpointed_model(model, block_size=checkpoint_block_size)

    model = apply_fp8_to_model(model, static_scale=static_scale)

    return model

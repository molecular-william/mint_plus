"""Optimization helpers for mu-Scaling (muS).

Implements:
  1. tau computation from network depth (paper Appendix A.3, Fig. 9)
  2. Learning rate scaling rules (paper Table 2)
  3. Parameter group builder for muS LR groups
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Tau (residual coefficient) from depth
# ---------------------------------------------------------------------------

def compute_tau(num_layers: int) -> float:
    """Return the recommended residual coefficient tau for a given depth.

    Paper Fig. 9 shows tau* decreases with depth. Empirical fit from the
    4-layer and 100-layer experiments:

        L=4   -> tau ~ 0.4
        L=20  -> tau ~ 0.2
        L=30  -> tau ~ 0.08  (150M model: 30 layers)
        L=33  -> tau ~ 0.06  (650M model: 33 layers)
        L=100 -> tau ~ 0.02  (extrapolated)

    The relationship is approximately: tau ~ 0.5 / sqrt(L)

    Args:
        num_layers: Number of transformer layers.

    Returns:
        Recommended tau value (0 < tau < 1).
    """
    idx = max(1, num_layers)
    return 0.5 / math.sqrt(idx)


# ---------------------------------------------------------------------------
# 2. Learning rate scaling rules (paper Table 2)
# ---------------------------------------------------------------------------

def compute_lr_scales(
    hidden_dim: int,
    base_width: int = 320,
) -> Dict[str, float]:
    """Return LR multipliers for each weight type under muS.

    muS scales hidden-layer LRs by sqrt(d_base / d_new), while
    embedding and LM head LRs stay at the base LR.

    Args:
        hidden_dim: Model's hidden dimension (embed_dim).
        base_width: Base model width used for hyperparameter transfer.
                    Default: 320 (ESM-2 8M embed_dim).

    Returns:
        Dictionary with keys 'hidden', 'embed', 'lm_head' mapping to
        LR multipliers.

    Example:
        scales = compute_lr_scales(640)  # 150M model
        # => {'hidden': 0.707, 'embed': 1.0, 'lm_head': 1.0}
    """
    scale = math.sqrt(base_width / max(hidden_dim, 1))
    return {
        "hidden": scale,
        "embed": 1.0,
        "lm_head": 1.0,
    }


# ---------------------------------------------------------------------------
# 3. Parameter group builder
# ---------------------------------------------------------------------------

def build_mu_param_groups(
    model: nn.Module,
    base_lr: float,
    hidden_dim: int,
    weight_decay: float = 0.01,
    base_width: int = 320,
    requires_grad_filter: bool = True,
) -> List[Dict]:
    """Build optimizer parameter groups with muS LR scaling.

    Groups parameters into:
      - hidden: all transformer linear weights, scaled by sqrt(d_base/d)
      - embed:  embedding table, uses base_lr
      - lm_head: LM head weights, uses base_lr

    Args:
        model: The model (may have frozen params).
        base_lr: Base learning rate (tuned on base_width model).
        hidden_dim: Model hidden dimension (embed_dim).
        weight_decay: Weight decay coefficient.
        base_width: Base model width for transfer.
        requires_grad_filter: If True, only include params with requires_grad=True.

    Returns:
        List of dicts suitable for torch.optim.Optimizer(param_groups=...).
    """
    scales = compute_lr_scales(hidden_dim, base_width)

    groups = {
        "hidden": {"params": [], "lr": base_lr * scales["hidden"], "weight_decay": weight_decay},
        "embed":  {"params": [], "lr": base_lr * scales["embed"],  "weight_decay": weight_decay},
        "lm_head": {"params": [], "lr": base_lr * scales["lm_head"], "weight_decay": weight_decay},
    }

    for name, param in model.named_parameters():
        if requires_grad_filter and not param.requires_grad:
            continue
        if "embed_tokens" in name:
            groups["embed"]["params"].append(param)
        elif "lm_head" in name:
            groups["lm_head"]["params"].append(param)
        else:
            groups["hidden"]["params"].append(param)

    # Remove empty groups
    return [g for g in groups.values() if g["params"]]

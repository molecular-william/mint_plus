"""Unit variance initialization for mu-Scaling (muS).

In muS, all weights are initialized with unit variance (std = 1.0)
instead of Xavier/Glorot. The output scaling factor is applied at
runtime as part of the linear layer forward pass (see output_multiplier
in mu_scaling/__init__.py).

This differs from the standard ESM-2 Xavier init:
  Standard:  W ~ XavierUniform(gain=1/sqrt(2))  => std = 1/sqrt(2*fan_in)
  muS:       W ~ N(0, 1)                         => std = 1.0

The muS scheme satisfies:
  1. Weight variance = 1 at initialization (requirement 2 in paper Sec. A.1.1)
  2. Linear output variance = 1 at init, assuming iid N(0,1) inputs,
     because the output multiplier 1/sqrt(fan_in) restores unit variance
"""

import math
import torch
import torch.nn as nn


def unit_variance_init_(module: nn.Module) -> nn.Module:
    """Apply unit-variance initialization to all linear and embedding layers.

    Weights are drawn from N(0, 1). Biases are zero-initialized.
    LayerNorm gains are initialized to 1 (default).

    Args:
        module: A nn.Module (in-place modification).

    Returns:
        The same module (for chaining).

    Example:
        model = MINT(...)
        model.apply(unit_variance_init_)
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm gains stay at 1 (default), biases at 0 (default)
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.RMSNorm):
            nn.init.ones_(m.weight)

    return module


def unit_variance_output_multiplier(fan_in: int) -> float:
    """Return the output multiplier for a linear layer under muS.

    muS output:  Y = (1 / sqrt(fan_in)) * X @ W

    With W ~ N(0,1) and X[i] ~ N(0,1) iid, the output Y has variance ~ 1.

    Args:
        fan_in: Input dimension of the linear layer.

    Returns:
        Scale factor to multiply the output by.
    """
    return 1.0 / math.sqrt(max(fan_in, 1))

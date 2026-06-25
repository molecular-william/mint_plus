# from μnit Scaling
# ensure all weights, activations, gradients have unit variance at initialization and throughout forward/backward passes
# hidden linear layer matmuls in torch.float8_e4m3fn (weights/activations) and torch.float8_e5m2 (gradients)
# supposedly no precision loss or overflow/underflow

# static scaling factor to outputs of all hidden linear layers
# initialize all hidden linear weights with variance of 1.0
# res-post-layer norm, this fixes variance vanishing or exploding along sequence lines
# fixed residual multiplication with variance preserving formula
# clamp values to maximum representable range of FP8 before casting, but keep initial embeddings and final LM output head in bfloat16

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnitScaledLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, fan_in):
        # save tensors for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.fan_in = fan_in
        # clamp to fp8 e4m3 max range to prevent overflows before casting
        f8_max_e4m3 = 448.0
        x_clipped = torch.clamp(x, -f8_max_e4m3, f8_max_e4m3)
        w_clipped = torch.clamp(weight, -f8_max_e4m3, f8_max_e4m3)
        # cast to fp8 for matmul
        x_f8 = x_clipped.to(torch.float8_e4m3fn)
        w_f8 = w_clipped.to(torch.float8_e4m3fn)
        # compute linear output in fp8, should be native on Hopper
        output = torch.matmul(x_f8, w_f8.t())
        if bias is not None:
            output = output + bias.to(output.dtype)
        # Apply output static scaling factor: 1 / sqrt(fan_in)
        return output * (1.0 / (fan_in ** 0.5))

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        fan_in = ctx.fan_in
        # Scale incoming gradient as per backward rules
        grad_output = grad_output * (1.0 / (fan_in ** 0.5))
        # Clamp and cast gradients to FP8 E5M2 for backward efficiency
        f8_max_e5m2 = 57344.0 # Max finite value for float8_e5m2
        f8_max_e4m3 = 448.0
        grad_clipped = torch.clamp(grad_output, -f8_max_e5m2, f8_max_e5m2)
        grad_f8 = grad_clipped.to(torch.float8_e5m2)
        
        x_clipped = torch.clamp(x, -f8_max_e4m3, f8_max_e4m3).to(torch.float8_e4m3fn)
        w_clipped = torch.clamp(weight, -f8_max_e4m3, f8_max_e4m3).to(torch.float8_e4m3fn)
        
        # Gradients calculations
        grad_x = torch.matmul(grad_f8, w_clipped)
        # Handle flattening for 3D activation matrices [B, T, E]
        grad_flat = grad_f8.reshape(-1, grad_f8.shape[-1])
        x_flat = x_clipped.reshape(-1, x_clipped.shape[-1])
        grad_w = torch.matmul(grad_flat.t(), x_flat)
        
        grad_bias = grad_flat.sum(dim=0) if bias is not None else None
        
        return grad_x, grad_w, grad_bias, None


class UnitScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Unit variance initialization: N(0, 1)
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return UnitScaledLinearFunction.apply(x, self.weight, self.bias, self.in_features)
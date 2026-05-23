"""
Liger Kernel — SwiGLU MLP Module Wrapper

Source: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/swiglu.py
License: BSD-2-Clause
Retrieved: 2026-05-23

This is the nn.Module wrapper that replaces HuggingFace's LlamaMLP.
It uses standard nn.Linear for the three projections (gate, up, down)
and fuses only the SiLU(gate) * up activation via LigerSiLUMulFunction.
"""

import torch
import torch.nn as nn

from liger_kernel.ops import LigerSiLUMulFunction


class LigerSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

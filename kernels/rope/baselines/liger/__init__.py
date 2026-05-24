"""Liger-Kernel RoPE baseline (vendored).

Source: https://github.com/linkedin/Liger-Kernel
License: BSD-2-Clause
Vendored: 2026-05-23

`apply_rope(q, k, cos, sin)` is a thin wrapper around `LigerRopeFunction.apply`
that matches HF's call signature (no position_ids, no unsqueeze_dim).
"""

import torch
from .rope import LigerRopeFunction


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to Q and K using Liger's fused kernel.

    Args:
        q: (batch, n_q_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        cos, sin: (1, seq_len, head_dim) or (batch, seq_len, head_dim)

    Returns:
        (q_out, k_out) with the same shapes.
    """
    return LigerRopeFunction.apply(q, k, cos, sin, None, 1)


__all__ = ["apply_rope", "LigerRopeFunction"]

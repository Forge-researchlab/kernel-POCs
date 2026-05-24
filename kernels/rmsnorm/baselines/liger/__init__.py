"""Liger-Kernel RMSNorm baseline (vendored).

Source: https://github.com/linkedin/Liger-Kernel
        src/liger_kernel/ops/rms_norm.py
License: BSD-2-Clause
Vendored: 2026-05-24

Provides `apply_rmsnorm(x, weight, eps, offset=0.0, casting_mode="llama")`
matching Forge's own kernel surface. The underlying autograd Function is
`LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place, row_mode)`
— we adapt it to a Forge-style call signature.
"""
import torch

from .rms_norm import LigerRMSNormFunction


def apply_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
):
    """Apply RMSNorm using Liger's fused kernel.

    Args:
        x: (..., H) — input tensor. Last dim is the normalized axis.
        weight: (H,) — the affine scale (Gemma uses near-zero init; offset=1.0).
        eps: small constant for numerical stability of rsqrt.
        offset: added to weight inside the kernel (Llama: 0.0, Gemma: 1.0).
        casting_mode: "llama" (rstd-only fp32), "gemma" (all fp32), or "none".

    Returns:
        Tensor with the same shape and dtype as x.
    """
    # Liger's Function.apply signature:
    #   (X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None)
    # in_place=False keeps backward bit-equivalent to Forge's convention.
    return LigerRMSNormFunction.apply(x, weight, eps, offset, casting_mode, False, None)


__all__ = ["apply_rmsnorm", "LigerRMSNormFunction"]

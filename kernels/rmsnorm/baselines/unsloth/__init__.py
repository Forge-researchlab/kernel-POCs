"""Unsloth RMSNorm baseline (vendored).

Source: https://github.com/unslothai/unsloth
        unsloth/kernels/rms_layernorm.py
License: Apache-2.0 (the rms_layernorm file specifically — NOT LGPL like
         Unsloth's rope_embedding.py). Confirmed via the file's own header.
         Compatible with Forge OSS launch.
Vendored: 2026-05-24

Unsloth exposes a single autograd Function `Fast_RMS_Layernorm.apply(X, W, eps,
gemma: bool)` plus a high-level helper `fast_rms_layernorm(layernorm_module, X,
gemma=False)` that pulls weight + eps from an nn.Module.

We expose a Forge-style `apply_rmsnorm(x, weight, eps, offset, casting_mode)`.
`offset==1.0` flips Unsloth's `gemma=True` branch, which routes through
`_gemma_rms_layernorm_forward` (the fp32-throughout kernel) and applies
`(W + 1.0)` in backward. casting_mode is accepted for API parity but Unsloth
has no explicit casting-mode surface — its split is binary (gemma=True/False),
so we map "gemma" → gemma=True and everything else → gemma=False.
"""
import torch

from .rms_layernorm import Fast_RMS_Layernorm


def apply_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
):
    """Apply RMSNorm using Unsloth's fused kernel.

    Args:
        x: (..., H) — input tensor. Last dim is the normalized axis.
        weight: (H,) — affine scale.
        eps: rsqrt epsilon.
        offset: 0.0 -> Llama path; 1.0 -> Gemma path (kernel adds +1 to weight).
        casting_mode: accepted for API parity; "gemma" forces Gemma path.

    Returns:
        Tensor with the same shape and dtype as x.
    """
    gemma = (offset == 1.0) or (casting_mode == "gemma")
    return Fast_RMS_Layernorm.apply(x, weight, eps, gemma)


__all__ = ["apply_rmsnorm", "Fast_RMS_Layernorm"]

"""Vendored Liger and Unsloth RMSNorm baselines for benchmarking against ForgeRMSNorm.

Each baseline exposes the same minimal interface:

    apply_rmsnorm(x, weight, eps=1e-6, offset=0.0, casting_mode="llama") -> Tensor

Shape contract:
    x:       (..., H)  bf16 / fp16 / fp32
    weight:  (H,)      same dtype family as x
    offset:  0.0 (Llama/Qwen) or 1.0 (Gemma)
    casting_mode: "llama" | "gemma" | "none"

Returns a tensor with the same shape and dtype as x.

License posture:
    - Liger (BSD-2-Clause) — compatible with Forge OSS launch.
    - Unsloth rms_layernorm.py (Apache-2.0) — compatible. Note this differs
      from Unsloth's rope_embedding.py which is LGPL; RMSNorm is the safer
      file in their tree.
"""
from . import liger
from . import unsloth

__all__ = ["liger", "unsloth"]

"""Vendored Liger and Unsloth RoPE baselines for benchmarking against ForgeRoPE.

Each baseline exposes the same minimal interface:

    apply_rope(q, k, cos, sin) -> (q_out, k_out)

Shape contract (matches HF Qwen3):
    q:   (batch, n_q_heads,  seq_len, head_dim)
    k:   (batch, n_kv_heads, seq_len, head_dim)
    cos: (1, seq_len, head_dim) or (batch, seq_len, head_dim)
    sin: same as cos

Returns:
    q_out, k_out with the same shapes as inputs.
"""

from . import liger
from . import unsloth

__all__ = ["liger", "unsloth"]

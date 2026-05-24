"""Unsloth RoPE baseline (vendored).

Source: https://github.com/unslothai/unsloth (`unsloth/kernels/rope_embedding.py`)
License: ⚠️ File header is LGPL-3.0-or-later, repo top-level LICENSE is Apache-2.0.
         Reference reading is fine; do NOT copy patterns into Forge's OSS release.
Vendored: 2026-05-23

Two paths exposed via `fast_rope_embedding`:
- Default (no rope_indices): runs `Fast_RoPE_Embedding` separately for Q and K (two launches).
- With rope_indices: runs `Fast_RoPE_Embedding_QK` (single fused launch).

`apply_rope(q, k, cos, sin)` calls the default (no indices) entry point, which is
what our `forge.patch` will see when patching HF's `apply_rotary_pos_emb`.
`apply_rope_qk_fused(q, k, cos, sin)` calls the genuinely-fused path with a
dummy contiguous range for rope_indices.
"""

import torch
from .rope_embedding import (
    fast_rope_embedding,
    Fast_RoPE_Embedding,
    Fast_RoPE_Embedding_QK,
)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """Default Unsloth entry point — Q and K applied in separate kernel launches.

    Args:
        q: (batch, n_q_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        cos, sin: (1, seq_len, head_dim) or (batch, seq_len, head_dim)

    Returns:
        (q_out, k_out)
    """
    return fast_rope_embedding(q, k, cos, sin, rope_embedding_indices=None)


def apply_rope_qk_fused(q: torch.Tensor, k: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor):
    """Fused-QK entry point — single launch for both Q and K.

    Forces use of `Fast_RoPE_Embedding_QK` by passing a contiguous-range
    rope_embedding_indices. This is the kernel the hackathon plan refers to.
    """
    batch, _, seq_len, _ = q.shape
    rope_indices = torch.arange(seq_len, device=q.device, dtype=torch.int32)
    rope_indices = rope_indices.unsqueeze(0).expand(batch, -1)
    return fast_rope_embedding(q, k, cos, sin, rope_embedding_indices=rope_indices)


__all__ = [
    "apply_rope",
    "apply_rope_qk_fused",
    "fast_rope_embedding",
    "Fast_RoPE_Embedding",
    "Fast_RoPE_Embedding_QK",
]

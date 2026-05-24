"""ForgeRoPE V3 — fused Q+K rotary embedding with GQA grouping + autotune.

Measured 2026-05-23 on A100 80GB at Qwen3-8B train shape (b=2, s=2048, n_q=32, n_kv=8):
    forward 66 us  = 7.1x PyTorch, 2.8x Unsloth-fused-QK, 63% of HBM peak.

The patching layer (forge.patching.core._forge_apply_rotary_pos_emb) calls
`apply_rope` directly with the q/k/cos/sin that transformers passes through
its module-level `apply_rotary_pos_emb` function. There is no module-instance
forward to replace for RoPE — it lives inline inside attention.forward, so we
monkey-patch the module-level function instead.
"""
from kernels.rope.forge_rope_v3 import (
    apply_rope,
    ForgeRoPEv3,
    ForgeRoPEv3Function,
)


def forge_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Drop-in replacement for transformers.models.{qwen2,gemma2}.modeling.apply_rotary_pos_emb.

    Assumes the new HF (>=4.41) contract: cos/sin already indexed to the current
    sequence, shape (batch, seq_len, head_dim). Older transformers that pass the
    full cos/sin table plus position_ids are NOT handled by this shim — patch the
    Qwen2Attention.forward directly if you need that path.

    HF sometimes pre-unsqueezes cos/sin to shape (batch, 1, seq_len, dim) before
    calling so broadcasting over the head dim is implicit. We undo that — our
    apply_rope handles the per-head broadcast internally via its grid layout.

    Returns: (q_rot, k_rot), same shapes as q, k.
    """
    if cos.dim() == 4:
        cos = cos.squeeze(unsqueeze_dim)
        sin = sin.squeeze(unsqueeze_dim)
    return apply_rope(q, k, cos, sin)


__all__ = [
    "apply_rope",
    "ForgeRoPEv3",
    "ForgeRoPEv3Function",
    "forge_apply_rotary_pos_emb",
]

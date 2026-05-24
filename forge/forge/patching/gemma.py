"""Gemma-2 patch mapping.

Same shape as QWEN3_MAPPING, three architectural differences captured as config:

  - RMSNorm has an additive offset of 1.0 on the affine weight
    (Gemma computes  (1 + w) * x / rms  vs Qwen's  w * x / rms).
  - MLP activation is GELU, not SiLU (GeGLU vs SwiGLU).
  - RoPE base differs from Qwen's 10000.0 — set in model.config.rope_theta and
    consumed by HF's rotary embedding module. Our kernel doesn't need to know;
    it operates on whatever cos/sin tensors HF computes from that base.

Status: Embedding, RMSNorm(offset=1.0), RoPE, and GeGLU wire to real kernels.
GeGLU selection stays in the kernel-adapter registry, so core.py remains only
responsible for traversal, mutation, and restoration.
"""
from forge.kernels.rope import forge_apply_rotary_pos_emb


GEMMA_MAPPING = {
    # Normalization — Gemma applies (1 + w), captured via offset=1.0
    "GemmaRMSNorm":     ("rmsnorm",   {"offset": 1.0}),
    "Gemma2RMSNorm":    ("rmsnorm",   {"offset": 1.0}),

    # MLP — Gemma uses GeGLU (GELU), not SwiGLU. The adapter infers the HF
    # GELU approximation mode from the module config and calls projections as
    # modules so PEFT wrappers and live weights remain visible.
    "GemmaMLP":         ("geglu",     {"activation": "gelu"}),
    "Gemma2MLP":        ("geglu",     {"activation": "gelu"}),

    # Embedding — same as Qwen; nn.Embedding at model.embed_tokens
    "Embedding":        ("embedding", {}),
}


GEMMA_MODULE_LEVEL_PATCHES = {
    "rope": (
        "transformers.models.gemma2.modeling_gemma2",
        "apply_rotary_pos_emb",
        forge_apply_rotary_pos_emb,
    ),
}

"""Qwen2 / Qwen3 patch mapping.

Two kinds of patches:

QWEN3_MAPPING:   keyed by HF module class name; value = (kernel_name, config).
                 The patching loop walks model.named_modules(), looks each class
                 up here, and rebinds module.forward via the closure factory in
                 patching/core.py.

QWEN3_MODULE_LEVEL_PATCHES: keyed by kernel name; value = (module_path, attr,
                 replacement). Used for things like apply_rotary_pos_emb that
                 are module-level functions, not nn.Module forwards.

Class names verified at scaffold time against transformers ~4.40+. If a future
transformers release renames a class, the mapping silently skips that module —
detect this by running patch(model) and confirming forge._forge_originals
contains the expected number of entries.

Must-ship for the hackathon demo (priority items 3-5 of the scope ladder):
    Qwen3RMSNorm  -> rmsnorm  ✓ wired
    Qwen3MLP      -> swiglu   ✓ wired
    Embedding     -> embedding ✓ wired
Stretch (originally deferred — but our RoPE V3 is the strongest kernel, so
we wire it via the module-level patch table):
    apply_rotary_pos_emb -> forge_apply_rotary_pos_emb ✓ wired
"""
from forge.kernels.rope import forge_apply_rotary_pos_emb


# class_name -> (kernel_name, config)
QWEN3_MAPPING = {
    # Normalization (covers Qwen2.5 and Qwen3 — class name varies by transformers version)
    "Qwen2RMSNorm":   ("rmsnorm",   {"offset": 0.0}),
    "Qwen3RMSNorm":   ("rmsnorm",   {"offset": 0.0}),

    # MLP — replaces the whole gate/up/down block so the activation lives inside Forge
    "Qwen2MLP":       ("swiglu",    {"activation": "silu"}),
    "Qwen3MLP":       ("swiglu",    {"activation": "silu"}),

    # Token embedding — Qwen exposes it as a vanilla nn.Embedding at model.embed_tokens
    "Embedding":      ("embedding", {}),
}


# kernel_name -> (module_path, attr_name, replacement_callable)
# These are restored on unpatch().
QWEN3_MODULE_LEVEL_PATCHES = {
    "rope": [
        (
            "transformers.models.qwen2.modeling_qwen2",
            "apply_rotary_pos_emb",
            forge_apply_rotary_pos_emb,
        ),
        (
            "transformers.models.qwen3.modeling_qwen3",
            "apply_rotary_pos_emb",
            forge_apply_rotary_pos_emb,
        ),
    ],
}

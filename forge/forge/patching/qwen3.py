"""Qwen2 / Qwen3 patch mapping.

Two kinds of patches:

QWEN3_MAPPING:   keyed by HF module class name; value = (kernel_name, config)
                 or a priority list of those tuples. The patching loop walks
                 model.named_modules(), looks each class up here, and rebinds
                 module.forward via the closure factory in patching/core.py.

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
Additional wired paths:
    apply_rotary_pos_emb -> forge_apply_rotary_pos_emb ✓ wired
    Qwen*ForCausalLM    -> fused_linear_ce             ✓ wired for training labels
    torch F.cross_entropy -> forge_cross_entropy       ✓ wired
    Qwen*Attention      -> lora_qkv_v3                 ✓ PEFT LoRA only
    Qwen*MLP            -> lora_mlp_v3                 ✓ PEFT LoRA only, else swiglu
"""
from forge.kernels.rope import forge_apply_rotary_pos_emb
from forge.patching.kernels.cross_entropy import forge_cross_entropy_replacement


# class_name -> (kernel_name, config)
QWEN3_MAPPING = {
    # Normalization (covers Qwen2.5 and Qwen3 — class name varies by transformers version)
    "Qwen2RMSNorm":   ("rmsnorm",   {"offset": 0.0}),
    "Qwen3RMSNorm":   ("rmsnorm",   {"offset": 0.0}),

    # Causal LM head — enables fused linear + CE in training without materializing logits.
    "Qwen2ForCausalLM": ("fused_linear_ce", {}),
    "Qwen3ForCausalLM": ("fused_linear_ce", {}),

    # Attention — LoRA-QKV v3 is selected only when PEFT LoRA adapters are present.
    "Qwen2Attention": [("lora_qkv", {}),],
    "Qwen3Attention": [("lora_qkv", {}),],

    # MLP — prefer LoRA-MLP when adapters are present, otherwise use base SwiGLU.
    "Qwen2MLP":       [("lora_mlp", {}), ("swiglu",    {"activation": "silu"})],
    "Qwen3MLP":       [("lora_mlp", {}), ("swiglu",    {"activation": "silu"})],

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
    "cross_entropy": (
        "torch.nn.functional",
        "cross_entropy",
        forge_cross_entropy_replacement,
    ),
}

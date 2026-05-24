"""Forge kernels — re-exports of the POC kernels at /workspace/kernel-POCs/kernels/.

Available (real, wired into the patching layer):
    forge.kernels.rope         — ForgeRoPEv3, apply_rope    (kernels/rope/forge_rope_v3.py)
    forge.kernels.layernorm    — ForgeLayerNormLiger        (kernels/layernorm/)
    forge.kernels.embedding    — ForgeEmbeddingFunction     (kernels/embedding/experiments/v1/)
    forge.kernels.lora_mlp     — lora_mlp_v3                (kernels/lora_mlp/experiments/v3/)
                                  [not wired into patching yet — PEFT integration deferred]

Declared in QWEN3/GEMMA_MAPPING but NOT yet built by the team — calling these
through forge.patch raises NotImplementedError:
    rmsnorm           — H7 port (port from RMSNorm POC; POC not in repo)
    swiglu (SiLU)     — H1
    geglu (GELU)      — H6
    cross_entropy     — H3
    lora_qkv          — H5
    fused_linear_ce   — H7 port
"""

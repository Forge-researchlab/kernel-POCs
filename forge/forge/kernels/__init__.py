"""Forge kernels — re-exports of the POC kernels at /workspace/kernel-POCs/kernels/.

Available (real, wired into the patching layer):
    forge.kernels.rope         — ForgeRoPEv3, apply_rope    (kernels/rope/forge_rope_v3.py)
    forge.kernels.rmsnorm      — ForgeRMSNormFunction       (kernels/rmsnorm/rmsnorm.py)
    forge.kernels.swiglu       — ForgeSwiGLUFunction        (kernels/swiglu/swiglu.py)
    forge.kernels.cross_entropy — ForgeCrossEntropyFunction,
                                  ForgeFusedLinearCrossEntropyFunction
                                  (kernels/cross_entropy/experiments/v2/)
    forge.kernels.layernorm    — ForgeLayerNormLiger        (kernels/layernorm/)
    forge.kernels.embedding    — ForgeEmbeddingFunction     (kernels/embedding/experiments/v1/)
    forge.kernels.lora_mlp     — lora_mlp_v3                (kernels/lora_mlp/experiments/v3/)
    forge.kernels.lora_qkv     — lora_qkv_v3                (kernels/lora_qkv/experiments/v3/)

Declared in QWEN3/GEMMA_MAPPING but NOT yet built/wired by the team — calling
these through forge.patch raises NotImplementedError when explicitly requested:
    geglu (GELU)      — H6
"""

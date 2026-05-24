"""Forge LayerNorm — Liger and Unsloth variants.

The patching layer uses ForgeLayerNormLigerFunction (full dX + dW + dB backward).
The Unsloth variant skips dW/dB to save memory and is unsafe when the affine
parameters are trainable (e.g., under LoRA), so it is not wired into patch.py.

Note: Qwen3 and Gemma both use RMSNorm, not LayerNorm. This kernel is here for
GPT-2 / BERT / models that still ship with affine LayerNorm. It is exported so
external callers can use it directly; it does not appear in QWEN3_MAPPING or
GEMMA_MAPPING.
"""
from kernels.layernorm import (
    ForgeLayerNormLiger,
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnsloth,
    ForgeLayerNormUnslothFunction,
)

__all__ = [
    "ForgeLayerNormLiger",
    "ForgeLayerNormLigerFunction",
    "ForgeLayerNormUnsloth",
    "ForgeLayerNormUnslothFunction",
]

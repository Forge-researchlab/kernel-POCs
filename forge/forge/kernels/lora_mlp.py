"""Forge LoRA-MLP V3 — cuBLAS for matmuls + Triton epilogue that fuses LoRA add and SwiGLU.

Measured 2026-05-23 at LLaMA-8B scale (batch=4, seq=2048, rank=16):
    V3:        12.38 ms  = 1.14x Unsloth's LoRA_MLP, rank-independent up to r=64.

NOT yet wired into forge.patch — PEFT integration deferred per the Day 2 scope
ladder (item 9). The mapping in QWEN3_MAPPING / GEMMA_MAPPING declares "lora_mlp"
as a kernel name so the future PEFT-aware closure factory can be a one-line
swap-in, but today it raises NotImplementedError.

Re-exported here for direct callers (benchmarks, tests, the LoRA-MLP test suite
under kernels/lora_mlp/tests/).
"""
from kernels.lora_mlp.experiments.v3.lora_mlp_kernel_v3 import (
    lora_mlp_v3,
    fused_lora_swiglu,
)

__all__ = ["lora_mlp_v3", "fused_lora_swiglu"]

"""Forge LoRA-MLP V3 — cuBLAS for matmuls + Triton epilogue that fuses LoRA add and SwiGLU.

Measured 2026-05-23 at LLaMA-8B scale (batch=4, seq=2048, rank=16):
    V3:        12.38 ms  = 1.14x Unsloth's LoRA_MLP, rank-independent up to r=64.

Wired into forge.patch for Qwen PEFT models when gate/up/down projections all
have one active, bias-free, dropout-free LoRA adapter. Re-exported here for
direct callers (benchmarks, tests, the LoRA-MLP test suite under
kernels/lora_mlp/tests/).
"""
import sys
from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parents[3] / "kernels" / "lora_mlp"
if str(_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_DIR))

from kernels.lora_mlp.experiments.v3.lora_mlp_kernel_v3 import (  # noqa: E402
    LoRAMLPv3,
    fused_lora_swiglu,
    lora_mlp_v3,
)

__all__ = ["LoRAMLPv3", "lora_mlp_v3", "fused_lora_swiglu"]

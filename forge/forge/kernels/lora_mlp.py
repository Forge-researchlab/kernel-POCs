"""Forge LoRA-MLP kernel re-exports.

Two versions live here:

  * V3 — cuBLAS matmuls + Triton epilogue fusing LoRA-add and SwiGLU.
    Used by `forge.patching.kernels.lora.make_lora_mlp_forward` (the wired patch
    factory). Measured 2026-05-23 at LLaMA-8B scale (batch=4, seq=2048, rank=16):
        V3:  12.38 ms = 1.14x Unsloth's LoRA_MLP, rank-independent up to r=64.

  * V6 — cuBLAS + Triton stacked LoRA-A + optional CUDA-stream parallelism.
    Latest experimental kernel; ~449 MB peak forward memory win vs v5 via the
    in-place fused LoRA-SwiGLU step. NOT yet wired into forge.patch; exported
    here so tests and benchmarks can target it.

Direct callers (kernels/lora_mlp/tests/, forge/tests/test_lora_mlp.py) import
from this shim rather than the experiments/vN/ tree to keep the import surface
stable across kernel revisions.
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
from kernels.lora_mlp.experiments.v6.lora_mlp_kernel_v6 import (  # noqa: E402
    LoRAMLPv6,
    LoRAMLPv6Module,
    lora_mlp_v6,
    stack_lora_a,
)

__all__ = [
    # v3 (wired)
    "LoRAMLPv3", "lora_mlp_v3", "fused_lora_swiglu",
    # v6 (latest — for tests / benchmarks)
    "LoRAMLPv6", "lora_mlp_v6", "LoRAMLPv6Module", "stack_lora_a",
]

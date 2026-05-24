"""Forge LoRA-QKV kernel re-exports.

Two versions live here:

  * V3 — `LoRAQKVFunction` + `lora_qkv_v3` entry helper. Used by
    `forge.patching.kernels.lora.make_lora_qkv_forward` (the wired patch
    factory). Training-compatible, supports GQA + 2D/3D inputs.

  * V4 — `LoRAQKVv4Function` + `lora_qkv_v4` entry helper. Adds a packed-weights
    fast path for the backward (9 cuBLAS + 1 Triton epilogue) plus
    `pack_weights_backward` / `pack_lora_a` helpers. NOT yet wired into
    forge.patch; exported here so tests and benchmarks can target it.
"""

from kernels.lora_qkv.experiments.v2.lora_qkv_kernel_v2_3 import pack_weights_all
from kernels.lora_qkv.experiments.v3.lora_qkv_kernel_v3 import (
    LoRAQKVFunction,
    lora_qkv_v3,
)
from kernels.lora_qkv.experiments.v4.lora_qkv_kernel_v4 import (
    LoRAQKVv4Function,
    lora_qkv_v4,
    pack_lora_a,
    pack_weights_backward,
)

__all__ = [
    # v3 (wired into forge.patch)
    "LoRAQKVFunction", "lora_qkv_v3", "pack_weights_all",
    # v4 (latest — for tests / benchmarks)
    "LoRAQKVv4Function", "lora_qkv_v4",
    "pack_weights_backward", "pack_lora_a",
]

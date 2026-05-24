"""Stable Forge package export for the LoRA-QKV v3 kernel.

Use the training-compatible v3 implementation, not any later experimental
variant, for patching Qwen attention projections.
"""

from kernels.lora_qkv.experiments.v3.lora_qkv_kernel_v3 import lora_qkv_v3
from kernels.lora_qkv.experiments.v2.lora_qkv_kernel_v2_3 import pack_weights_all

__all__ = ["lora_qkv_v3", "pack_weights_all"]

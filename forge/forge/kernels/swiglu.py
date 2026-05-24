"""Forge SwiGLU re-export for package-level patching.

The implementation lives in the POC tree at ``kernels/swiglu/swiglu.py``.
The Qwen patch replaces the whole HF MLP forward with:
``down_proj(swiglu(gate_proj(x), up_proj(x)))``.
"""
from kernels.swiglu import (
    ForgePackedSwiGLUFunction,
    ForgeSwiGLUFunction,
    swiglu,
    swiglu_backward,
    swiglu_forward,
    swiglu_packed,
    swiglu_packed_backward,
    swiglu_packed_forward,
    torch_swiglu_packed_reference,
    torch_swiglu_reference,
)

__all__ = [
    "ForgePackedSwiGLUFunction",
    "ForgeSwiGLUFunction",
    "swiglu",
    "swiglu_backward",
    "swiglu_forward",
    "swiglu_packed",
    "swiglu_packed_backward",
    "swiglu_packed_forward",
    "torch_swiglu_packed_reference",
    "torch_swiglu_reference",
]

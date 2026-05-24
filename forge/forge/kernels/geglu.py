"""Forge GeGLU re-export for package-level patching.

The implementation lives in the POC tree at ``kernels/geglu/geglu.py``.
The Gemma patch replaces the whole HF MLP forward with:
``down_proj(geglu(gate_proj(x), up_proj(x), approximate=...))``.
"""
from kernels.geglu import (
    ForgeGEGLUFunction,
    ForgePackedGEGLUFunction,
    geglu,
    geglu_backward,
    geglu_forward,
    geglu_packed,
    geglu_packed_backward,
    geglu_packed_forward,
    torch_geglu_packed_reference,
    torch_geglu_reference,
)

__all__ = [
    "ForgeGEGLUFunction",
    "ForgePackedGEGLUFunction",
    "geglu",
    "geglu_backward",
    "geglu_forward",
    "geglu_packed",
    "geglu_packed_backward",
    "geglu_packed_forward",
    "torch_geglu_packed_reference",
    "torch_geglu_reference",
]

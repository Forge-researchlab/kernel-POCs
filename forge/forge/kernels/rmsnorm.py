"""Forge RMSNorm re-export for package-level patching.

The implementation lives in the POC tree at ``kernels/rmsnorm/rmsnorm.py``.
This module gives the patching layer a stable import path:
``from forge.kernels.rmsnorm import ForgeRMSNormFunction``.
"""
from kernels.rmsnorm import (
    ForgeRMSNormFunction,
    apply_rmsnorm,
    rmsnorm,
    rmsnorm_backward,
    rmsnorm_forward,
    torch_rmsnorm_reference,
)

__all__ = [
    "ForgeRMSNormFunction",
    "apply_rmsnorm",
    "rmsnorm",
    "rmsnorm_backward",
    "rmsnorm_forward",
    "torch_rmsnorm_reference",
]

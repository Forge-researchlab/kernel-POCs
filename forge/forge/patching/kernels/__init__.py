"""Registry of kernel-specific patch adapters.

``core.py`` imports this registry and stays responsible only for traversal,
mutation, and restoration. Architecture- or kernel-specific extraction logic
lives in the adapter modules next to this file.
"""
from __future__ import annotations

from typing import Callable, Dict

from .basic import (
    make_embedding_forward,
    make_geglu_forward,
    make_rmsnorm_forward,
    make_swiglu_forward,
)
from .common import ForgeSkipPatch
from .fused_linear_ce import make_fused_linear_ce_forward
from .lora import make_lora_mlp_forward, make_lora_qkv_forward


FORWARD_MAKERS: Dict[str, Callable] = {
    "embedding": make_embedding_forward,
    "rmsnorm": make_rmsnorm_forward,
    "swiglu": make_swiglu_forward,
    "fused_linear_ce": make_fused_linear_ce_forward,
    "lora_mlp": make_lora_mlp_forward,
    "lora_qkv": make_lora_qkv_forward,
    "geglu": make_geglu_forward,
}


__all__ = ["FORWARD_MAKERS", "ForgeSkipPatch"]

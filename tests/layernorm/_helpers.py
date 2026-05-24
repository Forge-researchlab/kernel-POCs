"""Shared helpers for LayerNorm tests.

Centralizes shape lists, tolerance constants, input factories, and the
analytical byte-count formulas used by the bandwidth test. Byte formulas are
ported verbatim from layernorm/layernorm_profiling.py:643,652,663.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


# --- A100 constants ---------------------------------------------------------

A100_PEAK_BW = 1555e9  # bytes/sec — A100 40GB HBM peak


# --- Shapes -----------------------------------------------------------------

SHAPES_SMALL = [
    (2, 8, 1024),
    (1, 1, 128),
    (4, 512, 4096),
]

SHAPES_DESIGN = [
    (4, 2048, 4096),
    (8, 2048, 4096),
]

SHAPES_SWEEP = SHAPES_SMALL + SHAPES_DESIGN + [(8, 4096, 4096)]


# --- Tolerances -------------------------------------------------------------
# bf16 LN mean/var reductions cast back to bf16 can drift; empirically rtol=1e-2
# is the floor that holds for (8, 2048, 4096). Gradcheck stays strict at fp64.

TOL_BF16 = dict(rtol=1e-2, atol=1e-2)
TOL_FP16 = dict(rtol=1e-3, atol=1e-3)
TOL_FP32 = dict(rtol=1e-5, atol=1e-5)
TOL_FP64 = dict(rtol=1e-7, atol=1e-7)


def tol_for(dtype: torch.dtype) -> dict:
    if dtype == torch.bfloat16:
        return TOL_BF16
    if dtype == torch.float16:
        return TOL_FP16
    if dtype == torch.float32:
        return TOL_FP32
    if dtype == torch.float64:
        return TOL_FP64
    raise ValueError(f"unhandled dtype {dtype}")


# --- Input factories --------------------------------------------------------

def make_inputs(
    shape: Tuple[int, int, int],
    dtype: torch.dtype,
    device: str = "cuda",
    requires_grad: bool = True,
    nontrivial_affine: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X, W, B, dY).

    If nontrivial_affine=True, W and B are randomized rather than ones/zeros so
    that affine gradients have signal in them.
    """
    B, S, H = shape
    X = torch.randn(B, S, H, dtype=dtype, device=device, requires_grad=requires_grad)
    if nontrivial_affine:
        W = torch.randn(H, dtype=dtype, device=device, requires_grad=requires_grad) * 0.5 + 1.0
        Bp = torch.randn(H, dtype=dtype, device=device, requires_grad=requires_grad) * 0.1
        W = W.detach().requires_grad_(requires_grad)
        Bp = Bp.detach().requires_grad_(requires_grad)
    else:
        W = torch.ones(H, dtype=dtype, device=device, requires_grad=requires_grad)
        Bp = torch.zeros(H, dtype=dtype, device=device, requires_grad=requires_grad)
    dY = torch.randn(B, S, H, dtype=dtype, device=device)
    return X, W, Bp, dY


def clone_leaf(t: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    """Return a fresh leaf tensor with the same data and given requires_grad."""
    return t.detach().clone().requires_grad_(requires_grad)


# --- Byte-count formulas (ported from layernorm_profiling.py) ---------------

def _elem_size(dtype: torch.dtype) -> int:
    return 2 if dtype in (torch.bfloat16, torch.float16) else 4


def n_rows(shape: Tuple[int, int, int]) -> int:
    B, S, _ = shape
    return B * S


def compute_fwd_bytes(shape: Tuple[int, int, int], dtype: torch.dtype) -> int:
    elem = _elem_size(dtype)
    n = n_rows(shape)
    d = shape[-1]
    read = n * d * elem + 2 * d * elem            # X + W + B
    write = n * d * elem + 2 * n * 4              # Y + mean(fp32) + rstd(fp32)
    return read + write


def compute_bwd_bytes_liger(shape: Tuple[int, int, int], dtype: torch.dtype,
                            sm_count: int) -> int:
    elem = _elem_size(dtype)
    n = n_rows(shape)
    d = shape[-1]
    num_progs = min(n, sm_count)
    read = 2 * n * d * elem + d * elem + 2 * n * 4         # dY + X + W + mean + rstd
    write = n * d * elem + 2 * num_progs * d * 4           # dX + partial dW/dB (fp32)
    return read + write


def compute_bwd_bytes_unsloth(shape: Tuple[int, int, int], dtype: torch.dtype) -> int:
    elem = _elem_size(dtype)
    n = n_rows(shape)
    d = shape[-1]
    read = 2 * n * d * elem + d * elem + 2 * n * 4         # dY + X + W + r + mu
    write = n * d * elem                                    # dX in-place into dY
    return read + write


def sm_count(device: str = "cuda") -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# --- Misc -------------------------------------------------------------------

def to_dtype_for_ref(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().clone().to(dtype)

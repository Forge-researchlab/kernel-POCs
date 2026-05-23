"""
Unsloth — SwiGLU Triton Kernels (Forward & Backward)

Source: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py
License: Apache-2.0
Retrieved: 2026-05-23

These are the Triton kernels used for the SwiGLU activation inside
Unsloth's LoRA MLP pipeline. They are pointwise kernels (no matmul fusion)
that compute:
  Forward:  h = SiLU(e) * g
  Backward: computes h, df, de in-place, reusing the e and g buffers

Key feature: the backward kernel writes results IN-PLACE into the input
buffers (DW, e, g) to save memory allocations.
"""

import triton
import triton.language as tl
import torch

BLOCK_SIZE = 1024
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    Forward SwiGLU kernel: h = SiLU(e) * g

    Each program processes BLOCK_SIZE elements.
    SiLU computed in fp32 for numerical stability.
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = e * sigmoid(e) = SiLU(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    # h = f * g
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


def swiglu_fg_kernel(e, g):
    """Python wrapper for forward SwiGLU kernel."""
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fg_kernel[grid](
        e, g, h, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
    )
    return h


@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    Backward SwiGLU kernel — computes and stores IN-PLACE:
      DW buffer <- h   (f * g, needed for down-projection LoRA grads)
      e  buffer <- df  (DW * f, the up-projection gradient signal)
      g  buffer <- de  (gate-projection gradient signal)

    This avoids allocating 3 new tensors for the backward intermediates.
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Recompute forward values from saved e
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)

    # SwiGLU output (recomputed)
    h_row = f_row * g_row

    # Gradient through the up-projection path: df = DW * f
    df_row = DW_row * f_row

    # Gradient through the gate path:
    # de = DW * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
    dg_row = DW_row * g_row
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Write results in-place
    tl.store(DW + offsets, h_row, mask=mask)  # h = f * g
    tl.store(e + offsets, df_row, mask=mask)   # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)   # de


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    """Python wrapper for backward SwiGLU kernel."""
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _DWf_DW_dfg_kernel[grid](
        DW, e, g, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
    )
    return DW, e, g

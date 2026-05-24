"""
Unsloth's SwiGLU Triton kernels — extracted from GitHub.

Source: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py
License: Apache-2.0
Retrieved: 2026-05-24

These kernels implement the SwiGLU activation function used in LLaMA's MLP:
    SwiGLU(e, g) = silu(e) * g = (e * sigmoid(e)) * g

Two kernels are provided:
    1. _fg_kernel: forward pass — computes h = silu(e) * g
    2. _DWf_DW_dfg_kernel: backward pass — computes derivatives in-place

RELEVANCE TO OUR PROJECT:
    These are NOT directly used in QKV projections (no activation in attention projections).
    However, they are valuable reference for:
    - Triton kernel structure (grid, block, mask patterns)
    - In-place computation patterns (backward overwrites input buffers)
    - fp32 accumulation for sigmoid (numerical stability)
    - Long indexing for large tensors (int64 offsets when > 2^31 elements)
    - How Unsloth structures their Triton kernels generally

PATTERNS TO LEARN FROM:
    1. Elementwise kernel grid: one program per BLOCK_SIZE elements (flat 1D grid)
    2. fp32 cast for sigmoid: e_row = tl.load(...).to(tl.float32) before sigmoid
    3. Cast back to input dtype after fp32 compute: f_row.to(g_row.dtype)
    4. In-place backward: stores derivatives into the input buffers (DW, e, g)
       to avoid allocating new tensors — saves memory during training
    5. Long indexing guard: uses int64 offsets when n_elements > 2^31 - buffer
"""

import triton
import triton.language as tl
import torch

# Unsloth uses a safety buffer to avoid int32 overflow in offset calculations.
# signed int32 max is 2^31 - 1, so with BLOCK_SIZE=1024 and a 4x buffer,
# they switch to int64 indexing when elements > 2^31 - 4096.
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


# ============================================================================
# Forward kernel: h = silu(e) * g
# ============================================================================
#
# This computes the SwiGLU activation: h = (e * sigmoid(e)) * g
# It's a simple elementwise kernel with one optimization:
#   - sigmoid is computed in fp32 for numerical stability
#   - result is cast back to the input dtype before multiplying with g
#
# Grid: 1D, one program per BLOCK_SIZE contiguous elements
# Each thread block processes BLOCK_SIZE elements of the flat tensor

@triton.jit
def _fg_kernel(
    e,            # pointer to gate activation tensor (input)
    g,            # pointer to up-projection tensor (input)
    h,            # pointer to output tensor
    n_elements,   # total number of elements
    BLOCK_SIZE: tl.constexpr,     # elements per thread block
    LONG_INDEXING: tl.constexpr,  # use int64 offsets if tensor > 2^31 elements
):
    block_idx = tl.program_id(0)

    # Compute element offsets for this block.
    # Uses int64 indexing for large tensors to avoid int32 overflow.
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs. Cast e to fp32 for sigmoid numerical stability.
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # SwiGLU: f = silu(e) = e * sigmoid(e), then h = f * g
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)  # cast back to input dtype (e.g., bf16)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


def swiglu_fg_kernel(e, g):
    """Python wrapper for the forward SwiGLU kernel."""
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fg_kernel[grid](e, g, h, n_elements,
                     BLOCK_SIZE=BLOCK_SIZE,
                     LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1)
    return h


# ============================================================================
# Backward kernel: computes derivatives IN-PLACE
# ============================================================================
#
# This kernel computes the SwiGLU backward pass and stores results IN-PLACE
# in the input buffers to save memory. The mapping is:
#   DW buffer → overwritten with h = f * g  (forward recomputation)
#   e buffer  → overwritten with df = DW * f (gate derivative)
#   g buffer  → overwritten with de = DW * g * sigmoid(e) * (1 + e*(1-sigmoid(e)))
#
# The math:
#   se = sigmoid(e)
#   f = se * e                              (silu)
#   h = f * g                               (swiglu output)
#   df = DW * f                             (derivative w.r.t. up-projection)
#   de = DW * g * se * (1 + e * (1 - se))   (derivative w.r.t. gate)
#
# PATTERN: In-place backward saves memory by reusing input buffers for gradients.
# This is safe because the inputs are consumed in a single pass and not needed
# after the backward kernel runs.

@triton.jit
def _DWf_DW_dfg_kernel(
    DW,           # pointer to upstream gradient (input) → overwritten with h
    e,            # pointer to gate activation (input) → overwritten with df
    g,            # pointer to up-projection (input) → overwritten with de
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
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

    # Recompute forward values (saves memory vs caching them)
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)

    h_row = f_row * g_row          # forward output (recomputed)
    df_row = DW_row * f_row        # gradient for up-projection path
    dg_row = DW_row * g_row        # intermediate for gate gradient

    # Gate gradient with full chain rule through silu
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives IN-PLACE into input buffers
    tl.store(DW + offsets, h_row, mask=mask)   # DW → h (forward recomputation)
    tl.store(e + offsets, df_row, mask=mask)    # e → df (gate gradient)
    tl.store(g + offsets, de_row, mask=mask)    # g → de (up-proj gradient)


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    """Python wrapper for the backward SwiGLU kernel."""
    batch_seq_len, hd = e.shape  # already flattened to 2D
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements,
                              BLOCK_SIZE=BLOCK_SIZE,
                              LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1)
    return DW, e, g

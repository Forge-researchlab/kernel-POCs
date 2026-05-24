"""
LayerNorm Triton kernels — two variants.

Ported from the standalone profiling suite. Both variants implement the same
mathematical LayerNorm (with affine weight/bias) but differ in backward
strategy:

  - Liger variant: backward computes dX, dW, dB. Uses partial accumulators
    (one row-group per SM) to avoid atomics. Suitable as the default.

  - Unsloth variant: backward computes ONLY dX, and writes it in-place into
    the dY buffer to save memory. Returns None for dW/dB — callers that
    train W/B must compute those gradients externally. Faster fwd+bwd when
    W/B grads are not needed (e.g., frozen norms).

5-step workflow (per hackathon convention):
  1. Forward kernel
  2. Backward kernel
  3. Gradcheck (fp64)
  4. autograd.Function wrap
  5. Benchmark vs PyTorch eager + torch.compile
"""
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


EPS_DEFAULT = 1e-6


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def layernorm_reference(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                        eps: float = EPS_DEFAULT) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps)


# ---------------------------------------------------------------------------
# Launch heuristic
# ---------------------------------------------------------------------------

def _calculate_settings(n_cols: int):
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    return BLOCK_SIZE, num_warps


# ===========================================================================
# Variant 1 — Liger-style: full dX + dW + dB backward, partial accumulators.
# ===========================================================================

@triton.jit
def _liger_layernorm_forward_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(ACC_DTYPE)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(ACC_DTYPE)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    x_hat = xc * rstd
    y = x_hat * w + b

    tl.store(Y_ptr + row * stride_x_row + offs, y.to(x.dtype), mask=mask)
    tl.store(Mean_ptr + row, mean)
    tl.store(RSTD_ptr + row, rstd)


@triton.jit
def _liger_layernorm_backward_kernel(
    DY_ptr, X_ptr, W_ptr, Mean_ptr, RSTD_ptr,
    DX_ptr, DW_ptr, DB_ptr,
    stride_row, N: tl.constexpr,
    M,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(ACC_DTYPE)

    acc_dw = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)
    acc_db = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, M)

    for row in range(row_start, row_end):
        dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(ACC_DTYPE)
        mean = tl.load(Mean_ptr + row)
        rstd = tl.load(RSTD_ptr + row)

        x_hat = (x - mean) * rstd
        dx_hat = dy * w

        c1 = tl.sum(dx_hat, axis=0) / N
        c2 = tl.sum(dx_hat * x_hat, axis=0) / N
        dx = rstd * (dx_hat - c1 - x_hat * c2)

        tl.store(DX_ptr + row * stride_row + offs, dx.to(dy.dtype), mask=mask)

        acc_dw += tl.where(mask, dy * x_hat, 0.0)
        acc_db += tl.where(mask, dy, 0.0)

    tl.store(DW_ptr + pid * N + offs, acc_dw, mask=mask)
    tl.store(DB_ptr + pid * N + offs, acc_db, mask=mask)


class ForgeLayerNormLigerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps):
        shape = X.shape
        dim = shape[-1]
        X_flat = X.view(-1, dim).contiguous()
        n_rows, n_cols = X_flat.shape

        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)

        is_fp64 = X.dtype == torch.float64
        acc_dtype = tl.float64 if is_fp64 else tl.float32
        buf_dtype = torch.float64 if is_fp64 else torch.float32

        Y = torch.empty_like(X_flat)
        Mean = torch.empty(n_rows, dtype=buf_dtype, device=X.device)
        RSTD = torch.empty(n_rows, dtype=buf_dtype, device=X.device)

        _liger_layernorm_forward_kernel[(n_rows,)](
            X_flat, Y, W, B, Mean, RSTD,
            X_flat.stride(0), n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            ACC_DTYPE=acc_dtype,
        )

        ctx.save_for_backward(X_flat, W, B, Mean, RSTD)
        ctx.shape = shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_cols = n_cols
        ctx.acc_dtype = acc_dtype
        ctx.buf_dtype = buf_dtype
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        X_flat, W, B, Mean, RSTD = ctx.saved_tensors
        n_rows, n_cols = X_flat.shape

        dY_flat = dY.view(-1, n_cols).contiguous()
        DX = torch.empty_like(dY_flat)

        sm_count = torch.cuda.get_device_properties(X_flat.device).multi_processor_count
        num_programs = min(n_rows, sm_count)
        rows_per_program = math.ceil(n_rows / num_programs)

        DW_partial = torch.zeros(num_programs, n_cols, dtype=ctx.buf_dtype, device=X_flat.device)
        DB_partial = torch.zeros(num_programs, n_cols, dtype=ctx.buf_dtype, device=X_flat.device)

        _liger_layernorm_backward_kernel[(num_programs,)](
            dY_flat, X_flat, W, Mean, RSTD,
            DX, DW_partial, DB_partial,
            X_flat.stride(0), n_cols,
            n_rows, rows_per_program,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
            ACC_DTYPE=ctx.acc_dtype,
        )

        dW = DW_partial.sum(0).to(W.dtype)
        dB = DB_partial.sum(0).to(B.dtype)

        return DX.view(*ctx.shape), dW, dB, None


class ForgeLayerNormLiger(nn.Module):
    def __init__(self, hidden: int, eps: float = EPS_DEFAULT,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(hidden, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ForgeLayerNormLigerFunction.apply(x, self.weight, self.bias, self.eps)


# ===========================================================================
# Variant 2 — Unsloth-style: dX-only backward, in-place into dY buffer.
#
# WARNING: backward returns None for dW and dB. The affine parameters will
# NOT receive gradients through this Function — use only when W/B are frozen
# or when dW/dB are computed externally (e.g., via a separate reduction).
# ===========================================================================

@triton.jit
def _unsloth_layernorm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, B_ptr, R_ptr, Mu_ptr,
    n_cols: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    Y_ptr += row * Y_row_stride
    X_ptr += row * X_row_stride
    R_ptr += row
    Mu_ptr += row

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)

    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    inv_var = tl.math.rsqrt(var + eps)

    tl.store(R_ptr, inv_var)
    tl.store(Mu_ptr, mean)

    y = (xc * inv_var) * w + b
    tl.store(Y_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _unsloth_layernorm_backward_kernel(
    DY_ptr, DY_row_stride,
    X_ptr, X_row_stride,
    W_ptr, R_ptr, Mu_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Computes ONLY dX. Writes dX in-place into the dY buffer.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    DY_ptr += row * DY_row_stride
    X_ptr += row * X_row_stride
    R_ptr += row
    Mu_ptr += row

    dy = tl.load(DY_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(ACC_DTYPE)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(ACC_DTYPE)
    inv_var = tl.load(R_ptr).to(ACC_DTYPE)
    mean = tl.load(Mu_ptr).to(ACC_DTYPE)

    normed = (x - mean) * inv_var
    dy_w = dy * w

    c1 = tl.sum(dy_w, axis=0) / n_cols
    c2 = tl.sum(dy_w * normed, axis=0) / n_cols
    dx = inv_var * (dy_w - c1 - normed * c2)

    tl.store(DY_ptr + offs, dx.to(dy.dtype), mask=mask)


class ForgeLayerNormUnslothFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps):
        shape = X.shape
        dim = shape[-1]
        X_flat = X.view(-1, dim).contiguous()
        n_rows, n_cols = X_flat.shape

        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)

        is_fp64 = X.dtype == torch.float64
        acc_dtype = tl.float64 if is_fp64 else tl.float32
        buf_dtype = torch.float64 if is_fp64 else torch.float32

        Y = torch.empty_like(X_flat)
        R = torch.empty(n_rows, dtype=buf_dtype, device=X.device)
        Mu = torch.empty(n_rows, dtype=buf_dtype, device=X.device)

        _unsloth_layernorm_forward_kernel[(n_rows,)](
            Y, Y.stride(0),
            X_flat, X_flat.stride(0),
            W, B, R, Mu,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            ACC_DTYPE=acc_dtype,
        )

        ctx.save_for_backward(X_flat, W, B, R, Mu)
        ctx.shape = shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_cols = n_cols
        ctx.acc_dtype = acc_dtype
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        X_flat, W, B, R, Mu = ctx.saved_tensors
        n_rows, n_cols = X_flat.shape

        dY_flat = dY.view(-1, n_cols).contiguous()

        _unsloth_layernorm_backward_kernel[(n_rows,)](
            dY_flat, dY_flat.stride(0),
            X_flat, X_flat.stride(0),
            W, R, Mu,
            n_cols,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
            ACC_DTYPE=ctx.acc_dtype,
        )

        # dY_flat now contains dX (in-place overwrite). dW, dB intentionally None.
        return dY_flat.view(*ctx.shape), None, None, None


class ForgeLayerNormUnsloth(nn.Module):
    def __init__(self, hidden: int, eps: float = EPS_DEFAULT,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(hidden, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ForgeLayerNormUnslothFunction.apply(x, self.weight, self.bias, self.eps)

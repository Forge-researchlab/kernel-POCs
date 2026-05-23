# These are placeholder files for testing patching.

import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 131072
BACKWARD_ROWS_PER_PROGRAM = 16


@triton.jit
def _rmsnorm_forward_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    x_row = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_dtype = x_row.dtype
    x_fp32 = x_row.to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)

    mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)
    tl.store(rstd_ptr + row_idx, rstd)

    # Match the Llama/Qwen HF pattern: normalize in fp32, cast back to input
    # dtype, then apply the learned scale.
    y = (x_fp32 * rstd).to(x_dtype) * weight
    tl.store(y_ptr + row_start + col_offsets, y, mask=mask)


@triton.jit
def _rmsnorm_backward_kernel(
    dx_ptr,
    dweight_partial_ptr,
    dy_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    n_rows,
    n_cols: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    dweight_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row_offset in range(ROWS_PER_PROGRAM):
        row_idx = row_block_idx * ROWS_PER_PROGRAM + row_offset
        active_row = row_idx < n_rows
        row_mask = mask & active_row
        row_start = row_idx * n_cols

        dy = tl.load(dy_ptr + row_start + col_offsets, mask=row_mask, other=0.0)
        x_row = tl.load(x_ptr + row_start + col_offsets, mask=row_mask, other=0.0)
        x_dtype = x_row.dtype
        x = x_row.to(tl.float32)
        rstd = tl.load(rstd_ptr + row_idx, mask=active_row, other=0.0).to(tl.float32)

        normed = x * rstd
        normed_cast = normed.to(x_dtype)
        scaled_dy = (dy * weight).to(tl.float32)
        dot = tl.sum(scaled_dy * x, axis=0)
        dx = rstd * (scaled_dy - x * (rstd * rstd) * dot / n_cols)

        dweight_acc += dy * normed_cast
        tl.store(dx_ptr + row_start + col_offsets, dx, mask=row_mask)

    tl.store(
        dweight_partial_ptr + row_block_idx * n_cols + col_offsets,
        dweight_acc,
        mask=mask,
    )


def torch_rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(variance + eps)
    return normed.to(dtype=x.dtype) * weight


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


def _block_size(n_cols: int) -> int:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_BLOCK_SIZE:
        raise RuntimeError(
            f"RMSNorm hidden size {n_cols} is too large for this placeholder kernel; "
            f"max supported block size is {MAX_BLOCK_SIZE}."
        )
    return block_size


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if weight.dim() != 1:
        raise ValueError("weight must be a 1D tensor")
    if x.shape[-1] != weight.numel():
        raise ValueError(f"x.shape[-1] ({x.shape[-1]}) must equal weight.numel() ({weight.numel()})")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("x and weight must be floating point tensors")


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    _check_inputs(x, weight)
    if not x.is_cuda:
        raise RuntimeError("rmsnorm_forward requires CUDA; use rmsnorm() for the CPU fallback")

    original_shape = x.shape
    n_cols = original_shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    weight_1d = weight.contiguous()
    n_rows = x_2d.shape[0]

    y = torch.empty(x_2d.shape, device=x.device, dtype=torch.result_type(x, weight))
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    block_size = _block_size(n_cols)

    _rmsnorm_forward_kernel[(n_rows,)](
        y,
        x_2d,
        weight_1d,
        rstd,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return y.view(original_shape), x_2d, weight_1d, rstd


def rmsnorm_backward(
    dy: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
):
    original_shape = dy.shape
    n_cols = original_shape[-1]
    dy_2d = dy.contiguous().view(-1, n_cols)
    n_rows = dy_2d.shape[0]

    dx = torch.empty_like(x_2d)
    block_size = _block_size(n_cols)
    n_row_blocks = triton.cdiv(n_rows, BACKWARD_ROWS_PER_PROGRAM)
    dweight_partial = torch.empty((n_row_blocks, n_cols), device=dy.device, dtype=torch.float32)

    _rmsnorm_backward_kernel[(n_row_blocks,)](
        dx,
        dweight_partial,
        dy_2d,
        x_2d,
        weight,
        rstd,
        n_rows,
        n_cols,
        ROWS_PER_PROGRAM=BACKWARD_ROWS_PER_PROGRAM,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )

    dweight = dweight_partial.sum(dim=0).to(dtype=weight.dtype)
    return dx.view(original_shape), dweight


class ForgeRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        y, x_2d, weight_1d, rstd = rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x_2d, weight_1d, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        dx, dweight = rmsnorm_backward(dy, x_2d, weight, rstd)
        return dx, dweight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _check_inputs(x, weight)
    if not x.is_cuda:
        return torch_rmsnorm_reference(x, weight, eps)
    return ForgeRMSNormFunction.apply(x, weight, eps)

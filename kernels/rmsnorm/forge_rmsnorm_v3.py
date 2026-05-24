"""ForgeRMSNorm V3 — V2 + @triton.autotune over (num_warps, num_stages).

V2 picks `num_warps` heuristically from `BLOCK_SIZE` (4 for ≥512, 8 for ≥2048,
16 for ≥8192, 32 for ≥32768). That schedule is reasonable but never explored —
at Qwen3 H=4096 we always get num_warps=8, at Gemma2 H=2304 we get 4, at
Llama3-70B H=8192 we get 16. V3 lets Triton autotune pick the (num_warps,
num_stages) pair per (n_cols, ACC_DTYPE, CASTING_MODE) combination from a small
grid of configs.

Tuning surface:
  num_warps  ∈ {4, 8, 16}
  num_stages ∈ {2, 3}                          # async load pipelining

Cache key:
  ("n_cols", "ACC_DTYPE")                       # rebench when hidden size or dtype changes

Expected gain over v2:
  5-15% on shapes where v2's heuristic picks suboptimal warps. The biggest
  wins are at small-hidden Gemma shapes (H=2304 → v2 picks 4 warps, v3 may
  find 8 warps is better) and at Qwen3-8B train (H=4096 → v2 picks 8, v3 may
  find 4 or 16 wins on a given GPU).

First call per (n_cols, ACC_DTYPE) is slow (six configs compiled + measured);
thereafter Triton caches the winner.

Kernel bodies are IDENTICAL to v2 — only the @triton.autotune decorator and
the host call (no num_warps arg) differ.
"""

import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 131072

_CASTING_LLAMA = 0
_CASTING_GEMMA = 1
_CASTING_NONE = 2

_CASTING_MODE_BY_NAME = {
    "llama": _CASTING_LLAMA,
    "gemma": _CASTING_GEMMA,
    "none":  _CASTING_NONE,
}


# ----------------------------------------------------------------------------
# Autotune configs — small grid keeps first-call compile cost bounded.
# ----------------------------------------------------------------------------

_AUTOTUNE_CONFIGS_FWD = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in (4, 8, 16)
    for ns in (2, 3)
]
_AUTOTUNE_CONFIGS_BWD = list(_AUTOTUNE_CONFIGS_FWD)


# ----------------------------------------------------------------------------
# Forward Triton kernel — identical body to v2's _rmsnorm_v2_forward_kernel,
# decorated with @triton.autotune.
# ----------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS_FWD, key=["n_cols", "ACC_DTYPE"])
@triton.jit
def _rmsnorm_v3_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr, RSTD_row_stride,
    n_cols,
    eps,
    OFFSET: tl.constexpr,
    CASTING_MODE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    X_row = X_ptr + row * X_row_stride
    Y_row = Y_ptr + row * Y_row_stride
    RSTD_row = RSTD_ptr + row * RSTD_row_stride

    x = tl.load(X_row + col, mask=mask, other=0.0)
    w = tl.load(W_ptr + col, mask=mask, other=0.0)

    x_acc = x.to(ACC_DTYPE)
    mean_square = tl.sum(x_acc * x_acc, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)
    tl.store(RSTD_row, rstd)

    if CASTING_MODE == 0:  # LLAMA
        normed_native = (x_acc * rstd).to(x.dtype)
        y = normed_native * (w + OFFSET)
    elif CASTING_MODE == 1:  # GEMMA
        w_acc = w.to(ACC_DTYPE)
        y = (x_acc * rstd) * (w_acc + OFFSET)
    else:  # NONE
        rstd_native = rstd.to(x.dtype)
        y = (x * rstd_native) * (w + OFFSET)

    tl.store(Y_row + col, y, mask=mask)


# ----------------------------------------------------------------------------
# Backward Triton kernel — identical body to v2's, autotuned.
# ----------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS_BWD, key=["n_cols", "ACC_DTYPE"])
@triton.jit
def _rmsnorm_v3_backward_kernel(
    dX_ptr, dX_row_stride,
    dW_partial_ptr,
    dY_ptr, dY_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr, RSTD_row_stride,
    n_rows,
    n_cols,
    rows_per_program,
    OFFSET: tl.constexpr,
    CASTING_MODE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    w = tl.load(W_ptr + col, mask=mask, other=0.0).to(ACC_DTYPE)
    w_offset = w + OFFSET
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=ACC_DTYPE)

    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)

    for row in range(row_start, row_end):
        X_row = X_ptr + row * X_row_stride
        dY_row = dY_ptr + row * dY_row_stride
        dX_row = dX_ptr + row * dX_row_stride
        rstd = tl.load(RSTD_ptr + row * RSTD_row_stride).to(ACC_DTYPE)

        x = tl.load(X_row + col, mask=mask, other=0.0).to(ACC_DTYPE)
        dy = tl.load(dY_row + col, mask=mask, other=0.0)
        dy_acc = dy.to(ACC_DTYPE)

        x_hat = x * rstd
        scaled_dy = dy_acc * w_offset
        dot = tl.sum(scaled_dy * x, axis=0)
        dx = rstd * (scaled_dy - x * rstd * rstd * dot / n_cols)

        tl.store(dX_row + col, dx.to(dy.dtype), mask=mask)
        dw_acc += dy_acc * x_hat

    tl.store(dW_partial_ptr + pid * n_cols + col, dw_acc, mask=mask)


# ----------------------------------------------------------------------------
# Host helpers — copied from v2; the only difference is the host launcher
# below does NOT pass num_warps (autotune handles it).
# ----------------------------------------------------------------------------

def _resolve_casting_mode(casting_mode) -> int:
    if isinstance(casting_mode, int):
        if casting_mode not in (_CASTING_LLAMA, _CASTING_GEMMA, _CASTING_NONE):
            raise ValueError(f"Invalid casting_mode int: {casting_mode}")
        return casting_mode
    if isinstance(casting_mode, str):
        try:
            return _CASTING_MODE_BY_NAME[casting_mode]
        except KeyError as exc:
            raise ValueError(
                f"Invalid casting_mode {casting_mode!r}; "
                f"expected one of {sorted(_CASTING_MODE_BY_NAME)}"
            ) from exc
    raise TypeError(f"casting_mode must be int or str, got {type(casting_mode).__name__}")


def _block_size(n_cols: int) -> int:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_BLOCK_SIZE:
        raise RuntimeError(
            f"RMSNorm v3: hidden size {n_cols} requires block size {block_size} "
            f"which exceeds MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}."
        )
    return block_size


def _acc_dtypes(input_dtype: torch.dtype):
    if input_dtype == torch.float64:
        return torch.float64, tl.float64
    return torch.float32, tl.float32


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if weight.dim() != 1:
        raise ValueError("weight must be a 1D tensor")
    if x.shape[-1] != weight.numel():
        raise ValueError(
            f"x.shape[-1] ({x.shape[-1]}) must equal weight.numel() ({weight.numel()})"
        )
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("x and weight must be floating point tensors")


# Re-export the v2 oracle so v3 tests/bench can use the same reference.
from .forge_rmsnorm_v2 import torch_rmsnorm_reference_v2 as torch_rmsnorm_reference_v3  # noqa: E402


# ----------------------------------------------------------------------------
# Host launchers
# ----------------------------------------------------------------------------

def rmsnorm_v3_forward(x, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
    _check_inputs(x, weight)
    if not x.is_cuda:
        raise RuntimeError("rmsnorm_v3_forward requires CUDA")
    mode = _resolve_casting_mode(casting_mode)
    original_shape = x.shape
    n_cols = original_shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    weight_1d = weight.contiguous()
    n_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)
    acc_torch, acc_tl = _acc_dtypes(x.dtype)
    rstd = torch.empty((n_rows,), device=x.device, dtype=acc_torch)
    block_size = _block_size(n_cols)

    _rmsnorm_v3_forward_kernel[(n_rows,)](
        y, y.stride(0),
        x_2d, x_2d.stride(0),
        weight_1d,
        rstd, rstd.stride(0),
        n_cols,
        eps,
        OFFSET=float(offset),
        CASTING_MODE=mode,
        ACC_DTYPE=acc_tl,
        BLOCK_SIZE=block_size,
    )
    return y.view(original_shape), x_2d, weight_1d, rstd, mode


def rmsnorm_v3_backward(dy, x_2d, weight, rstd, offset=0.0, casting_mode_int=_CASTING_LLAMA):
    original_shape = dy.shape
    n_cols = original_shape[-1]
    dy_2d = dy.contiguous().view(-1, n_cols)
    n_rows = dy_2d.shape[0]

    dx = torch.empty_like(x_2d)
    block_size = _block_size(n_cols)
    acc_torch, acc_tl = _acc_dtypes(x_2d.dtype)

    sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
    num_programs = min(n_rows, sm_count)
    rows_per_program = triton.cdiv(n_rows, num_programs)

    dw_partial = torch.empty(
        (num_programs, n_cols),
        device=dy.device,
        dtype=acc_torch,
    )

    _rmsnorm_v3_backward_kernel[(num_programs,)](
        dx, dx.stride(0),
        dw_partial,
        dy_2d, dy_2d.stride(0),
        x_2d, x_2d.stride(0),
        weight,
        rstd, rstd.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        OFFSET=float(offset),
        CASTING_MODE=casting_mode_int,
        ACC_DTYPE=acc_tl,
        BLOCK_SIZE=block_size,
    )

    dw = dw_partial.sum(dim=0).to(dtype=weight.dtype)
    return dx.view(original_shape), dw


# ----------------------------------------------------------------------------
# autograd.Function + nn.Module + entry point
# ----------------------------------------------------------------------------

class ForgeRMSNormv3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
        y, x_2d, weight_1d, rstd, mode = rmsnorm_v3_forward(x, weight, eps, offset, casting_mode)
        ctx.save_for_backward(x_2d, weight_1d, rstd)
        ctx.eps = eps
        ctx.offset = float(offset)
        ctx.casting_mode_int = mode
        return y

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight_1d, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_v3_backward(
            dy, x_2d, weight_1d, rstd,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
        )
        return dx, dw, None, None, None


class ForgeRMSNormv3(torch.nn.Module):
    """v3 nn.Module wrapper — same API as ForgeRMSNormv2 but uses the autotuned kernels."""

    def __init__(self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama",
                 weight_init="ones", dtype=None, device=None):
        super().__init__()
        if weight_init not in ("ones", "zeros"):
            raise ValueError(f"weight_init must be 'ones' or 'zeros', got {weight_init!r}")
        self.variance_epsilon = float(eps)
        self.eps = self.variance_epsilon
        self.offset = float(offset)
        self.casting_mode = casting_mode
        init_fn = torch.ones if weight_init == "ones" else torch.zeros
        self.weight = torch.nn.Parameter(init_fn(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return apply_rmsnorm_v3(
            x, self.weight,
            eps=self.variance_epsilon,
            offset=self.offset,
            casting_mode=self.casting_mode,
        )

    def extra_repr(self):
        return (
            f"hidden_size={self.weight.numel()}, eps={self.variance_epsilon}, "
            f"offset={self.offset}, casting_mode={self.casting_mode!r}"
        )


def apply_rmsnorm_v3(x, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
    """v3 entry point — autotuned variant of v2.

    On CUDA: ForgeRMSNormv3Function.apply (Triton-autotuned).
    On CPU: torch_rmsnorm_reference_v3 (shared with v2).
    """
    _check_inputs(x, weight)
    if not x.is_cuda:
        return torch_rmsnorm_reference_v3(x, weight, eps, offset, casting_mode)
    return ForgeRMSNormv3Function.apply(x, weight, eps, offset, casting_mode)


__all__ = [
    "apply_rmsnorm_v3",
    "ForgeRMSNormv3",
    "ForgeRMSNormv3Function",
    "torch_rmsnorm_reference_v3",
    "rmsnorm_v3_forward",
    "rmsnorm_v3_backward",
]

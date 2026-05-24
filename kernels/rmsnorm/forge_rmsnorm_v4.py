"""ForgeRMSNorm V4 — V3 + in-place dY → dX backward (Unsloth-style memory saver).

The motivation:
- V3 always allocates a fresh `dX` buffer in backward. At Qwen3-8B train shape
  (b=2, s=2048, h=4096, bf16) that's a 33 MB alloc + write per RMSNorm backward
  call. Per layer with 4 RMSNorms (Gemma2) × ~32 layers ≈ several GB of
  activation memory and HBM traffic over a single backward pass.
- Unsloth (and Liger with `in_place=True` default) reuse the `dY` buffer as the
  `dX` output — no fresh allocation, dy is loaded into registers once then
  overwritten with dx. Cuts backward latency by roughly 2× on bandwidth-bound
  shapes (which RMSNorm is).
- Our V3-vs-Unsloth backward gap (~2× at most shapes — verified in
  benchmarks/results/v3_results.json) is closed here.

The catch — when in_place is NOT safe:
- Gemma2 uses a residual-paired RMSNorm pattern: the `dY` flowing into one
  RMSNorm backward may also feed another node's backward via the residual
  fan-out. Modifying `dY` in-place corrupts that path. Liger's docstring
  explicitly notes this:
    > gemma2 uses two rmsnorm sequentially with residual in between. The
    > residual part needs dY so it cannot be modified in-place.
- The kernel body itself is in-place-safe (loads `dy` into registers BEFORE
  storing `dx`; no re-reads). The unsafe-ness lives in the caller's autograd
  graph, not the kernel.

Contract:
- `apply_rmsnorm_v4(x, weight, eps, offset, casting_mode, in_place=True)`
- The autograd Function saves `in_place` on ctx; backward consults it to decide
  whether to pass `dy_2d` as the `dX_ptr` or allocate fresh.
- The closure factory in `forge.patching.core._make_rmsnorm_forward` passes
  `in_place=True` for Qwen2/3 (no fan-out conflict) and `in_place=False` for
  Gemma2 (residual conflict). See `forge/forge/patching/core.py`.

Kernel body is byte-identical to V3 — only the host launcher and Function
differ. The autotune decorator is retained from V3.
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
# Autotune configs — same as V3.
# ----------------------------------------------------------------------------

_AUTOTUNE_CONFIGS_FWD = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in (4, 8, 16)
    for ns in (2, 3)
]
_AUTOTUNE_CONFIGS_BWD = list(_AUTOTUNE_CONFIGS_FWD)


# ----------------------------------------------------------------------------
# Forward Triton kernel (identical to V3).
# ----------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS_FWD, key=["n_cols", "ACC_DTYPE"])
@triton.jit
def _rmsnorm_v4_forward_kernel(
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
# Backward Triton kernel (identical body to V3 — already in-place-safe because
# we load `dy` into registers BEFORE storing `dx`, with no later re-read of dy).
# ----------------------------------------------------------------------------

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS_BWD,
    key=["n_cols", "ACC_DTYPE"],
    # When in_place=True, the host passes dY_ptr aliased to dX_ptr. Autotune
    # runs the kernel multiple times to benchmark candidate configs, which
    # would corrupt dY across runs (each run writes dx into dy's buffer).
    # `restore_value` tells Triton to snapshot+restore these buffers between
    # benchmark iterations so each config sees the same dy. The final post-
    # tune run with the chosen config writes dx into dy as expected.
    restore_value=["dX_ptr", "dY_ptr"],
)
@triton.jit
def _rmsnorm_v4_backward_kernel(
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
        # CRITICAL ORDERING: load dy into registers FIRST, before computing dx.
        # This is what makes the kernel in-place safe even when dX_ptr == dY_ptr.
        dy = tl.load(dY_row + col, mask=mask, other=0.0)
        dy_acc = dy.to(ACC_DTYPE)

        x_hat = x * rstd
        scaled_dy = dy_acc * w_offset
        dot = tl.sum(scaled_dy * x, axis=0)
        dx = rstd * (scaled_dy - x * rstd * rstd * dot / n_cols)

        # Store dx — when dX_ptr == dY_ptr this overwrites dy's slot, but dy
        # is already in registers so it's safe.
        tl.store(dX_row + col, dx.to(dy.dtype), mask=mask)
        dw_acc += dy_acc * x_hat

    tl.store(dW_partial_ptr + pid * n_cols + col, dw_acc, mask=mask)


# ----------------------------------------------------------------------------
# Host helpers
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
            f"RMSNorm v4: hidden size {n_cols} requires block size {block_size} "
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


# Re-export the v2 oracle so v4 tests/bench can use the same reference.
from .forge_rmsnorm_v2 import torch_rmsnorm_reference_v2 as torch_rmsnorm_reference_v4  # noqa: E402


# ----------------------------------------------------------------------------
# Host launchers
# ----------------------------------------------------------------------------

def rmsnorm_v4_forward(x, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
    _check_inputs(x, weight)
    if not x.is_cuda:
        raise RuntimeError("rmsnorm_v4_forward requires CUDA")
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

    _rmsnorm_v4_forward_kernel[(n_rows,)](
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


def rmsnorm_v4_backward(
    dy,
    x_2d,
    weight,
    rstd,
    offset=0.0,
    casting_mode_int=_CASTING_LLAMA,
    in_place=True,
):
    """Backward with optional in-place dY → dX.

    When `in_place=True`, the kernel writes `dx` into the same memory as `dy`
    (saves one allocation + one buffer's worth of HBM traffic). Safe ONLY when
    the autograd graph doesn't re-read `dy` after our backward returns —
    typically true for Qwen3/Llama-style sequential blocks, false for Gemma2's
    residual-paired RMSNorm pattern.
    """
    original_shape = dy.shape
    n_cols = original_shape[-1]
    dy_2d = dy.contiguous().view(-1, n_cols)
    n_rows = dy_2d.shape[0]

    if in_place:
        # Reuse dy_2d as the dX buffer — kernel loads dy into registers before
        # storing dx, so no clobbering risk inside the kernel.
        dx_2d = dy_2d
    else:
        dx_2d = torch.empty_like(x_2d)

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

    _rmsnorm_v4_backward_kernel[(num_programs,)](
        dx_2d, dx_2d.stride(0),
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
    return dx_2d.view(original_shape), dw


# ----------------------------------------------------------------------------
# autograd.Function + nn.Module + entry point
# ----------------------------------------------------------------------------

class ForgeRMSNormv4Function(torch.autograd.Function):
    """Autograd interface for ForgeRMSNorm v4.

    forward(ctx, x, weight, eps, offset=0.0, casting_mode="llama", in_place=True)
    backward returns (dx, dw, None, None, None, None)

    The `in_place` flag is saved on `ctx` and consulted during backward to
    decide whether `dx` reuses `dy`'s storage.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        offset: float = 0.0,
        casting_mode: str = "llama",
        in_place: bool = True,
    ):
        y, x_2d, weight_1d, rstd, mode = rmsnorm_v4_forward(
            x, weight, eps, offset, casting_mode
        )
        ctx.save_for_backward(x_2d, weight_1d, rstd)
        ctx.eps = eps
        ctx.offset = float(offset)
        ctx.casting_mode_int = mode
        ctx.in_place = bool(in_place)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight_1d, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_v4_backward(
            dy, x_2d, weight_1d, rstd,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
            in_place=ctx.in_place,
        )
        # forward signature: (x, weight, eps, offset, casting_mode, in_place)
        return dx, dw, None, None, None, None


class ForgeRMSNormv4(torch.nn.Module):
    """v4 nn.Module wrapper. `in_place=True` (Unsloth-style dY→dX) by default;
    set False for residual-paired patterns (Gemma2)."""

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        in_place=True,
        weight_init="ones",
        dtype=None,
        device=None,
    ):
        super().__init__()
        if weight_init not in ("ones", "zeros"):
            raise ValueError(f"weight_init must be 'ones' or 'zeros', got {weight_init!r}")
        self.variance_epsilon = float(eps)
        self.eps = self.variance_epsilon
        self.offset = float(offset)
        self.casting_mode = casting_mode
        self.in_place = bool(in_place)
        init_fn = torch.ones if weight_init == "ones" else torch.zeros
        self.weight = torch.nn.Parameter(init_fn(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return apply_rmsnorm_v4(
            x, self.weight,
            eps=self.variance_epsilon,
            offset=self.offset,
            casting_mode=self.casting_mode,
            in_place=self.in_place,
        )

    def extra_repr(self):
        return (
            f"hidden_size={self.weight.numel()}, eps={self.variance_epsilon}, "
            f"offset={self.offset}, casting_mode={self.casting_mode!r}, "
            f"in_place={self.in_place}"
        )


def apply_rmsnorm_v4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
) -> torch.Tensor:
    """v4 entry point — v3 plus the in-place dY→dX backward optimization.

    On CUDA: ForgeRMSNormv4Function.apply (Triton-autotuned, optional in-place).
    On CPU: torch_rmsnorm_reference_v4 (shared with v2/v3).

    Args:
        x: (..., H) input.
        weight: (H,) affine scale.
        eps: rsqrt epsilon.
        offset: 0.0 (Llama/Qwen) or 1.0 (Gemma).
        casting_mode: "llama" | "gemma" | "none".
        in_place: when True (default), backward writes `dx` into `dy`'s buffer
                  for a ~2× backward speedup. **Unsafe in residual-paired
                  contexts where `dy` is read by another backward node** —
                  set False for Gemma2's RMSNorm pattern.

    Returns:
        Tensor with the same shape and dtype as x.
    """
    _check_inputs(x, weight)
    if not x.is_cuda:
        return torch_rmsnorm_reference_v4(x, weight, eps, offset, casting_mode)
    return ForgeRMSNormv4Function.apply(x, weight, eps, offset, casting_mode, in_place)


__all__ = [
    "apply_rmsnorm_v4",
    "ForgeRMSNormv4",
    "ForgeRMSNormv4Function",
    "torch_rmsnorm_reference_v4",
    "rmsnorm_v4_forward",
    "rmsnorm_v4_backward",
]

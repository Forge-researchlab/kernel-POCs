"""ForgeRMSNorm V2 — offset constexpr + casting modes + SM-proportional dW partials.

Three deltas vs v1:

  (1) OFFSET as `tl.constexpr` — fuses the Gemma `+1` into the existing fp32
      weight load. Zero runtime cost (Triton specializes a separate binary per
      OFFSET value). Closure-factory tensor materialization rejected: would
      allocate + HBM-write a `[H]` tensor each forward, or freeze weight at
      patch time and break LoRA.

  (2) CASTING_MODE as `tl.constexpr` (3 modes):
        LLAMA (0): rstd in fp32, then cast x*rstd back to input dtype, then
                   apply (weight + OFFSET). Matches LlamaRMSNorm / Qwen3RMSNorm.
        GEMMA (1): everything in fp32 through the affine multiply, cast at
                   store. Matches Gemma2RMSNorm where `1.0 + weight.float()`
                   loses precision in bf16 at near-zero weight init.
        NONE  (2): native dtype throughout (cheapest, debug only).

  (3) SM-proportional dW partials backward — `(num_programs ≈ SM_count, n_cols)`
      partial buffer + Python `partial.sum(0).to(weight.dtype)` final reduction.
      Replaces v1's `(ceil(n_rows / 16), n_cols)` buffer which over-allocates
      at large n_rows. On A100 (108 SMs) at Qwen3-8B train shape
      (n_rows=4096, H=4096, bf16): 1.7 MB partial buffer vs v1's 4 MB.
      Atomics-free, matches the team-locked pattern from
      kernels/layernorm/context.md §3.

Design references in `../rmsnorm_knowledge_base/02_liger/rms_norm.py` and the
detailed walkthrough in `docs/evolution_report.md`.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

MAX_BLOCK_SIZE = 131072  # Triton block-size cap

# casting_mode: tl.constexpr int. String -> int dispatch lives at the host edge
# (see _resolve_casting_mode). Inside the kernel we test against these ints.
_CASTING_LLAMA = 0
_CASTING_GEMMA = 1
_CASTING_NONE = 2

_CASTING_MODE_BY_NAME = {
    "llama": _CASTING_LLAMA,
    "gemma": _CASTING_GEMMA,
    "none":  _CASTING_NONE,
}


# ----------------------------------------------------------------------------
# Forward Triton kernel
# ----------------------------------------------------------------------------

@triton.jit
def _rmsnorm_v2_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr, RSTD_row_stride,
    n_cols,
    eps,
    OFFSET: tl.constexpr,
    CASTING_MODE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,        # fp32 for bf16/fp16/fp32, fp64 for fp64 (gradcheck)
    BLOCK_SIZE: tl.constexpr,
):
    """Single row per program. Grid: (n_rows,).

    ACC_DTYPE is the dtype for the reduction and rstd — fp32 for the common
    bf16/fp16 case, fp64 when the input is fp64 (so torch.autograd.gradcheck
    can perturb at fp64 precision without losing the perturbation to a fp32
    downcast inside the kernel).
    """
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    X_row = X_ptr + row * X_row_stride
    Y_row = Y_ptr + row * Y_row_stride
    RSTD_row = RSTD_ptr + row * RSTD_row_stride

    x = tl.load(X_row + col, mask=mask, other=0.0)
    w = tl.load(W_ptr + col, mask=mask, other=0.0)

    # Promote to ACC_DTYPE for the rsqrt — never downcasts because ACC_DTYPE
    # is fp64 when the input is fp64 (set by the host).
    x_acc = x.to(ACC_DTYPE)
    mean_square = tl.sum(x_acc * x_acc, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)
    tl.store(RSTD_row, rstd)

    # Apply affine: y = (x * rstd) * (w + OFFSET), with cast-back behavior
    # depending on CASTING_MODE.
    if CASTING_MODE == 0:  # LLAMA: cast x*rstd back to input dtype BEFORE affine
        normed_native = (x_acc * rstd).to(x.dtype)
        # w is loaded in input dtype; OFFSET (compile-time fp constant) promotes.
        y = normed_native * (w + OFFSET)
    elif CASTING_MODE == 1:  # GEMMA: keep ACC dtype through affine, cast at store
        w_acc = w.to(ACC_DTYPE)
        y = (x_acc * rstd) * (w_acc + OFFSET)
    else:  # NONE: native dtype (numerically worse, debug only)
        rstd_native = rstd.to(x.dtype)
        y = (x * rstd_native) * (w + OFFSET)

    tl.store(Y_row + col, y, mask=mask)


# ----------------------------------------------------------------------------
# Backward Triton kernel
# ----------------------------------------------------------------------------

@triton.jit
def _rmsnorm_v2_backward_kernel(
    dX_ptr, dX_row_stride,
    dW_partial_ptr,                 # shape (num_programs, n_cols), ACC_DTYPE
    dY_ptr, dY_row_stride,
    X_ptr, X_row_stride,
    W_ptr,
    RSTD_ptr, RSTD_row_stride,
    n_rows,
    n_cols,
    rows_per_program,               # runtime int — different per launch
    OFFSET: tl.constexpr,
    CASTING_MODE: tl.constexpr,     # accepted for future per-mode tweaks
    ACC_DTYPE: tl.constexpr,        # fp32 for bf16/fp16/fp32, fp64 for fp64 gradcheck
    BLOCK_SIZE: tl.constexpr,
):
    """Grid: (num_programs ≈ SM_count,). Each program handles a strip of rows."""
    pid = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    # w + OFFSET in ACC_DTYPE — used both for dx computation (across all rows in
    # this program's strip) and broadcasts cleanly across rows.
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

        # x_hat = x * rstd (the "normalized" intermediate that w_offset scales).
        x_hat = x * rstd

        # scaled_dy = dy * (w + OFFSET) in ACC_DTYPE
        scaled_dy = dy_acc * w_offset

        # dot = sum_over_H(scaled_dy * x)
        dot = tl.sum(scaled_dy * x, axis=0)

        # dx = rstd * (scaled_dy - x * rstd² * dot / H)
        dx = rstd * (scaled_dy - x * rstd * rstd * dot / n_cols)

        # Cast back to dY's dtype (and dX's, which match by allocation) at store.
        tl.store(dX_row + col, dx.to(dy.dtype), mask=mask)

        # dW contribution for this row: dy * x_hat. Accumulate in ACC_DTYPE
        # across all rows in this program's strip.
        dw_acc += dy_acc * x_hat

    # Write this program's strip of dW partials (ACC_DTYPE).
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


def _calculate_settings(n_cols: int):
    """Liger-style heuristic: power-of-2 block, warps scale with size."""
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_BLOCK_SIZE:
        raise RuntimeError(
            f"RMSNorm v2: hidden size {n_cols} requires block size {block_size} "
            f"which exceeds MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}."
        )
    num_warps = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    elif block_size < 512:
        num_warps = 1
    return block_size, num_warps


def _acc_dtypes(input_dtype: torch.dtype):
    """Pick the kernel accumulation dtype based on the input dtype.

    Returns (torch_dtype, triton_dtype) — fp32 for bf16/fp16/fp32, fp64 for fp64.
    fp64 path is for gradcheck precision; fp32 covers the production bf16/fp16
    use cases without losing accuracy on the rsqrt reduction.
    """
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


def torch_rmsnorm_reference_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> torch.Tensor:
    """Pure-PyTorch oracle matching HF's LlamaRMSNorm (offset=0) and Gemma2RMSNorm (offset=1).

    Used as the correctness oracle in tests/benchmarks and as the CPU fallback
    for `apply_rmsnorm_v2`.
    """
    mode = _resolve_casting_mode(casting_mode)
    input_dtype = x.dtype

    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)

    if mode == _CASTING_LLAMA:
        normed = (x_fp32 * rstd).to(input_dtype)
        # Llama: (weight + offset) lives in input dtype during the affine multiply
        return normed * (weight + offset)
    elif mode == _CASTING_GEMMA:
        # Gemma: all-fp32 through affine, cast at return
        return ((x_fp32 * rstd) * (weight.float() + offset)).to(input_dtype)
    else:  # NONE: native dtype
        rstd_native = rstd.to(input_dtype)
        return (x * rstd_native) * (weight + offset)


# ----------------------------------------------------------------------------
# Host launchers
# ----------------------------------------------------------------------------

def rmsnorm_v2_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode="llama",
):
    """Run the forward Triton kernel. Returns (y, x_2d, weight_1d, rstd, mode).

    The trailing (x_2d, weight_1d, rstd, mode) are saved for backward.
    """
    _check_inputs(x, weight)
    if not x.is_cuda:
        raise RuntimeError(
            "rmsnorm_v2_forward requires CUDA; use apply_rmsnorm_v2() for the CPU fallback."
        )
    mode = _resolve_casting_mode(casting_mode)
    original_shape = x.shape
    n_cols = original_shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    weight_1d = weight.contiguous()
    n_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)
    acc_torch, acc_tl = _acc_dtypes(x.dtype)
    rstd = torch.empty((n_rows,), device=x.device, dtype=acc_torch)
    block_size, num_warps = _calculate_settings(n_cols)

    _rmsnorm_v2_forward_kernel[(n_rows,)](
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
        num_warps=num_warps,
    )
    return y.view(original_shape), x_2d, weight_1d, rstd, mode


def rmsnorm_v2_backward(
    dy: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    offset: float = 0.0,
    casting_mode_int: int = _CASTING_LLAMA,
):
    """Run the backward Triton kernel. Returns (dx, dw)."""
    original_shape = dy.shape
    n_cols = original_shape[-1]
    dy_2d = dy.contiguous().view(-1, n_cols)
    n_rows = dy_2d.shape[0]

    dx = torch.empty_like(x_2d)
    block_size, num_warps = _calculate_settings(n_cols)
    acc_torch, acc_tl = _acc_dtypes(x_2d.dtype)

    # SM-proportional partial buffer.
    sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
    num_programs = min(n_rows, sm_count)
    rows_per_program = triton.cdiv(n_rows, num_programs)

    dw_partial = torch.empty(
        (num_programs, n_cols),
        device=dy.device,
        dtype=acc_torch,
    )

    _rmsnorm_v2_backward_kernel[(num_programs,)](
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
        num_warps=num_warps,
    )

    dw = dw_partial.sum(dim=0).to(dtype=weight.dtype)
    return dx.view(original_shape), dw


# ----------------------------------------------------------------------------
# autograd.Function
# ----------------------------------------------------------------------------

class ForgeRMSNormv2Function(torch.autograd.Function):
    """Autograd interface for ForgeRMSNorm v2.

    forward(ctx, x, weight, eps, offset=0.0, casting_mode="llama")
    backward returns (dx, dw, None, None, None)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        offset: float = 0.0,
        casting_mode: str = "llama",
    ):
        y, x_2d, weight_1d, rstd, mode = rmsnorm_v2_forward(
            x, weight, eps, offset, casting_mode
        )
        ctx.save_for_backward(x_2d, weight_1d, rstd)
        ctx.eps = eps
        ctx.offset = float(offset)
        ctx.casting_mode_int = mode
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight_1d, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_v2_backward(
            dy, x_2d, weight_1d, rstd,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
        )
        # forward signature: (x, weight, eps, offset, casting_mode)
        return dx, dw, None, None, None


# ----------------------------------------------------------------------------
# nn.Module wrapper
# ----------------------------------------------------------------------------

class ForgeRMSNormv2(torch.nn.Module):
    """Drop-in module replacement for HF RMSNorm classes.

    Args:
        hidden_size: H — length of the affine weight vector.
        eps: rsqrt epsilon.
        offset: 0.0 (Llama/Qwen) or 1.0 (Gemma).
        casting_mode: "llama" | "gemma" | "none".
        weight_init: how to initialize the weight tensor.
                     "ones" for Llama/Qwen; "zeros" for Gemma (since the +1 offset
                     makes a zero-weight init equivalent to identity at start).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 0.0,
        casting_mode: str = "llama",
        weight_init: str = "ones",
        dtype: torch.dtype = None,
        device=None,
    ):
        super().__init__()
        if weight_init not in ("ones", "zeros"):
            raise ValueError(f"weight_init must be 'ones' or 'zeros', got {weight_init!r}")
        # Use the same attribute names HF uses (variance_epsilon) so this module
        # is drop-in for `forge.patch` closures that read either.
        self.variance_epsilon = float(eps)
        self.eps = self.variance_epsilon  # alias for Gemma-style attribute lookup
        self.offset = float(offset)
        self.casting_mode = casting_mode
        init_fn = torch.ones if weight_init == "ones" else torch.zeros
        self.weight = torch.nn.Parameter(
            init_fn(hidden_size, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_rmsnorm_v2(
            x, self.weight,
            eps=self.variance_epsilon,
            offset=self.offset,
            casting_mode=self.casting_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.weight.numel()}, eps={self.variance_epsilon}, "
            f"offset={self.offset}, casting_mode={self.casting_mode!r}"
        )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def apply_rmsnorm_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> torch.Tensor:
    """Apply Forge v2 RMSNorm.

    On CUDA: dispatches to the Triton kernel via `ForgeRMSNormv2Function.apply`.
    On CPU: falls back to `torch_rmsnorm_reference_v2` (the pure-PyTorch oracle).

    Args:
        x: (..., H) input. Last dim is the normalized axis.
        weight: (H,) affine scale.
        eps: rsqrt epsilon.
        offset: 0.0 (Llama/Qwen) or 1.0 (Gemma — applies (weight + 1) inside the kernel).
        casting_mode: "llama" (rstd-only fp32, default), "gemma" (all-fp32 through affine),
                      or "none" (native dtype, debug).

    Returns:
        Tensor with the same shape and dtype as x.
    """
    _check_inputs(x, weight)
    if not x.is_cuda:
        return torch_rmsnorm_reference_v2(x, weight, eps, offset, casting_mode)
    return ForgeRMSNormv2Function.apply(x, weight, eps, offset, casting_mode)


__all__ = [
    "apply_rmsnorm_v2",
    "ForgeRMSNormv2",
    "ForgeRMSNormv2Function",
    "torch_rmsnorm_reference_v2",
    "rmsnorm_v2_forward",
    "rmsnorm_v2_backward",
]

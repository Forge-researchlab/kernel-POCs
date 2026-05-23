import numbers

import torch
import triton
import triton.language as tl


MAX_ROW_BLOCK_SIZE = 65536


def _as_float(name: str, value: float) -> float:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Python scalar, not a tensor")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real Python scalar")
    return float(value)


def _calculate_settings(n_cols: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_ROW_BLOCK_SIZE:
        raise RuntimeError(
            f"SwiGLU hidden size {n_cols} requires block size {block_size}, "
            f"which exceeds the row-wise limit {MAX_ROW_BLOCK_SIZE}."
        )

    num_warps = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


def _check_same_shape_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    if gate.device != up.device:
        raise ValueError("gate and up must be on the same device")
    if gate.dtype != up.dtype:
        raise TypeError(f"gate and up must have the same dtype, got {gate.dtype} and {up.dtype}")
    if not gate.is_floating_point() or not up.is_floating_point():
        raise TypeError("gate and up must be floating point tensors")


def _check_cuda_dtype(x: torch.Tensor) -> None:
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"CUDA SwiGLU supports fp16, bf16, and fp32, got {x.dtype}")


def _check_packed_input(gate_up: torch.Tensor) -> None:
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"packed gate_up last dimension must be even, got {gate_up.shape[-1]}")
    if not gate_up.is_floating_point():
        raise TypeError("gate_up must be a floating point tensor")


@triton.jit
def _swiglu_forward_kernel(
    out_ptr,
    gate_ptr,
    up_ptr,
    n_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_idx * n_cols

    gate = tl.load(gate_ptr + row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32) * gate_multiplier
    up = tl.load(up_ptr + row_start + offsets, mask=mask, other=0.0)

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)
    out = (silu_gate.to(gate_dtype) * up) * down_multiplier
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dgate_ptr,
    dup_ptr,
    dout_ptr,
    gate_ptr,
    up_ptr,
    n_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_idx * n_cols

    dout = tl.load(dout_ptr + row_start + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32) * gate_multiplier
    up = tl.load(up_ptr + row_start + offsets, mask=mask, other=0.0)

    sig = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sig
    dout_scaled = dout * down_multiplier

    dup = dout_scaled * silu_gate.to(gate_dtype)
    dgate = dout_scaled * up * (silu_gate * (1.0 - sig) + sig) * gate_multiplier

    tl.store(dgate_ptr + row_start + offsets, dgate, mask=mask)
    tl.store(dup_ptr + row_start + offsets, dup, mask=mask)


@triton.jit
def _swiglu_packed_forward_kernel(
    out_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row_start = row_idx * total_cols
    output_row_start = row_idx * n_cols

    gate = tl.load(gate_up_ptr + input_row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    up = tl.load(gate_up_ptr + input_row_start + n_cols + offsets, mask=mask, other=0.0)
    gate_fp32 = gate.to(tl.float32) * gate_multiplier

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)
    out = (silu_gate.to(gate_dtype) * up) * down_multiplier
    tl.store(out_ptr + output_row_start + offsets, out, mask=mask)


@triton.jit
def _swiglu_packed_backward_kernel(
    dgate_up_ptr,
    dout_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row_start = row_idx * total_cols
    output_row_start = row_idx * n_cols

    dout = tl.load(dout_ptr + output_row_start + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_up_ptr + input_row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    up = tl.load(gate_up_ptr + input_row_start + n_cols + offsets, mask=mask, other=0.0)
    gate_fp32 = gate.to(tl.float32) * gate_multiplier

    sig = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sig
    dout_scaled = dout * down_multiplier

    dup = dout_scaled * silu_gate.to(gate_dtype)
    dgate = dout_scaled * up * (silu_gate * (1.0 - sig) + sig) * gate_multiplier

    tl.store(dgate_up_ptr + input_row_start + offsets, dgate, mask=mask)
    tl.store(dgate_up_ptr + input_row_start + n_cols + offsets, dup, mask=mask)


def torch_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    silu_gate = torch.nn.functional.silu(gate.float() * gate_multiplier).to(dtype=gate.dtype)
    return (silu_gate * up) * down_multiplier


def torch_swiglu_packed_reference(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    _check_packed_input(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return torch_swiglu_reference(gate, up, gate_multiplier, down_multiplier)


def swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
):
    _check_same_shape_inputs(gate, up)
    _check_cuda_dtype(gate)
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)

    original_shape = gate.shape
    n_cols = original_shape[-1]
    gate_2d = gate.contiguous().view(-1, n_cols)
    up_2d = up.contiguous().view(-1, n_cols)
    out = torch.empty_like(gate_2d)
    block_size, num_warps = _calculate_settings(n_cols)

    _swiglu_forward_kernel[(gate_2d.shape[0],)](
        out,
        gate_2d,
        up_2d,
        n_cols=n_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.view(original_shape), gate_2d, up_2d


def swiglu_backward(
    dout: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
):
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    original_shape = dout.shape
    n_cols = original_shape[-1]
    dout_2d = dout.contiguous().view(-1, n_cols)
    block_size, num_warps = _calculate_settings(n_cols)

    dgate = torch.empty_like(gate) if preserve_inputs else gate
    dup = torch.empty_like(up) if preserve_inputs else up

    _swiglu_backward_kernel[(dout_2d.shape[0],)](
        dgate,
        dup,
        dout_2d,
        gate,
        up,
        n_cols=n_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dgate.view(original_shape), dup.view(original_shape)


def swiglu_packed_forward(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
):
    _check_packed_input(gate_up)
    _check_cuda_dtype(gate_up)
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)

    original_shape = gate_up.shape
    total_cols = original_shape[-1]
    n_cols = total_cols // 2
    gate_up_2d = gate_up.contiguous().view(-1, total_cols)
    out = torch.empty((gate_up_2d.shape[0], n_cols), device=gate_up.device, dtype=gate_up.dtype)
    block_size, num_warps = _calculate_settings(n_cols)

    _swiglu_packed_forward_kernel[(gate_up_2d.shape[0],)](
        out,
        gate_up_2d,
        n_cols=n_cols,
        total_cols=total_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.view(*original_shape[:-1], n_cols), gate_up_2d


def swiglu_packed_backward(
    dout: torch.Tensor,
    gate_up: torch.Tensor,
    input_shape: torch.Size,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
):
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    n_cols = dout.shape[-1]
    total_cols = input_shape[-1]
    dout_2d = dout.contiguous().view(-1, n_cols)
    block_size, num_warps = _calculate_settings(n_cols)

    dgate_up = torch.empty_like(gate_up) if preserve_inputs else gate_up

    _swiglu_packed_backward_kernel[(dout_2d.shape[0],)](
        dgate_up,
        dout_2d,
        gate_up,
        n_cols=n_cols,
        total_cols=total_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dgate_up.view(input_shape)


class ForgeSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        gate_multiplier: float = 1.0,
        down_multiplier: float = 1.0,
        preserve_inputs: bool = False,
    ):
        y, gate_2d, up_2d = swiglu_forward(gate, up, gate_multiplier, down_multiplier)
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
        ctx.down_multiplier = _as_float("down_multiplier", down_multiplier)
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate, dup = swiglu_backward(
            dout,
            gate,
            up,
            ctx.gate_multiplier,
            ctx.down_multiplier,
            ctx.preserve_inputs,
        )
        return dgate, dup, None, None, None


class ForgePackedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate_up: torch.Tensor,
        gate_multiplier: float = 1.0,
        down_multiplier: float = 1.0,
        preserve_inputs: bool = False,
    ):
        y, gate_up_2d = swiglu_packed_forward(gate_up, gate_multiplier, down_multiplier)
        ctx.save_for_backward(gate_up_2d)
        ctx.input_shape = gate_up.shape
        ctx.gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
        ctx.down_multiplier = _as_float("down_multiplier", down_multiplier)
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        dgate_up = swiglu_packed_backward(
            dout,
            gate_up,
            ctx.input_shape,
            ctx.gate_multiplier,
            ctx.down_multiplier,
            ctx.preserve_inputs,
        )
        return dgate_up, None, None, None


def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
) -> torch.Tensor:
    _check_same_shape_inputs(gate, up)
    if not gate.is_cuda:
        return torch_swiglu_reference(gate, up, gate_multiplier, down_multiplier)
    return ForgeSwiGLUFunction.apply(gate, up, gate_multiplier, down_multiplier, preserve_inputs)


def swiglu_packed(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
) -> torch.Tensor:
    _check_packed_input(gate_up)
    if not gate_up.is_cuda:
        return torch_swiglu_packed_reference(gate_up, gate_multiplier, down_multiplier)
    return ForgePackedSwiGLUFunction.apply(gate_up, gate_multiplier, down_multiplier, preserve_inputs)

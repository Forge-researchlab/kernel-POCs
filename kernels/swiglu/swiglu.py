# These are placeholder files for testing patching.

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _swiglu_forward_kernel(
    y_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0).to(tl.int64)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)
    y = silu_gate.to(gate_dtype) * up
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dgate_ptr,
    dup_ptr,
    dy_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0).to(tl.int64)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    sigmoid_gate = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sigmoid_gate
    silu_gate_cast = silu_gate.to(gate_dtype)

    dup = dy * silu_gate_cast
    dgate = (dy * up).to(tl.float32) * sigmoid_gate * (1.0 + gate_fp32 * (1.0 - sigmoid_gate))

    tl.store(dgate_ptr + offsets, dgate, mask=mask)
    tl.store(dup_ptr + offsets, dup, mask=mask)


def torch_swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    silu_gate = torch.nn.functional.silu(gate.float()).to(dtype=gate.dtype)
    return silu_gate * up


def _check_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    if gate.device != up.device:
        raise ValueError("gate and up must be on the same device")
    if not gate.is_floating_point() or not up.is_floating_point():
        raise TypeError("gate and up must be floating point tensors")


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor):
    _check_inputs(gate, up)
    if not gate.is_cuda:
        raise RuntimeError("swiglu_forward requires CUDA; use swiglu() for the CPU fallback")

    gate_contig = gate.contiguous()
    up_contig = up.contiguous()
    y = torch.empty(gate_contig.shape, device=gate.device, dtype=torch.result_type(gate, up))
    n_elements = gate_contig.numel()

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_forward_kernel[grid](
        y,
        gate_contig,
        up_contig,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, gate_contig, up_contig


def swiglu_backward(dy: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    dy_contig = dy.contiguous()
    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)
    n_elements = gate.numel()

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_backward_kernel[grid](
        dgate,
        dup,
        dy_contig,
        gate,
        up,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dgate, dup


class ForgeSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor):
        y, gate_contig, up_contig = swiglu_forward(gate, up)
        ctx.save_for_backward(gate_contig, up_contig)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate, dup = swiglu_backward(dy, gate, up)
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    _check_inputs(gate, up)
    if not gate.is_cuda:
        return torch_swiglu_reference(gate, up)
    return ForgeSwiGLUFunction.apply(gate, up)

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import tanh
except ModuleNotFoundError:
    try:
        from triton.language.extra.cuda.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.math import tanh


FLAT_BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = 2**31 - FLAT_BLOCK_SIZE * 4


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
        raise TypeError(f"CUDA GEGLU supports fp16, bf16, and fp32, got {x.dtype}")


def _check_packed_input(gate_up: torch.Tensor) -> None:
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"packed gate_up last dimension must be even, got {gate_up.shape[-1]}")
    if not gate_up.is_floating_point():
        raise TypeError("gate_up must be a floating point tensor")


def _check_approximate(approximate: str) -> bool:
    if approximate not in {"tanh", "none"}:
        raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")
    return approximate == "tanh"


def _check_linear_weight(
    name: str,
    weight: torch.Tensor,
    in_features: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be 2D [out_features, in_features], got shape {tuple(weight.shape)}")
    if weight.shape[1] != in_features:
        raise ValueError(f"{name} input features {weight.shape[1]} do not match x last dimension {in_features}")
    if not weight.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if device is not None and weight.device != device:
        raise ValueError(f"{name} must be on device {device}, got {weight.device}")
    if dtype is not None and weight.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {weight.dtype}")


def _check_optional_bias(
    name: str,
    bias: torch.Tensor | None,
    out_features: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if bias is None:
        return
    if bias.dim() != 1 or bias.shape[0] != out_features:
        raise ValueError(f"{name} must have shape [{out_features}], got {tuple(bias.shape)}")
    if not bias.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if device is not None and bias.device != device:
        raise ValueError(f"{name} must be on device {device}, got {bias.device}")
    if dtype is not None and bias.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {bias.dtype}")


def _uses_long_indexing(num_elements: int) -> bool:
    return num_elements > INT32_SAFETY_BUFFER


@triton.jit
def _gelu_activation(gate_fp32, APPROXIMATE_TANH: tl.constexpr):
    if APPROXIMATE_TANH:
        sqrt_2_over_pi = 0.7978845608028654
        tanh_arg = sqrt_2_over_pi * gate_fp32 * (1.0 + 0.044715 * gate_fp32 * gate_fp32)
        t = tanh(tanh_arg)
        activated = 0.5 * gate_fp32 * (1.0 + t)
    else:
        inv_sqrt_2 = 0.7071067811865476
        activated = 0.5 * gate_fp32 * (1.0 + tl.math.erf(gate_fp32 * inv_sqrt_2))

    activated = tl.where(gate_fp32 == -float("inf"), 0.0, activated)
    activated = tl.where(gate_fp32 == float("inf"), gate_fp32, activated)
    return activated


@triton.jit
def _gelu_activation_and_grad(gate_fp32, APPROXIMATE_TANH: tl.constexpr):
    if APPROXIMATE_TANH:
        sqrt_2_over_pi = 0.7978845608028654
        gate_sq = gate_fp32 * gate_fp32
        tanh_arg = sqrt_2_over_pi * gate_fp32 * (1.0 + 0.044715 * gate_sq)
        t = tanh(tanh_arg)
        activated = 0.5 * gate_fp32 * (1.0 + t)
        grad = 0.5 * (1.0 + t) + 0.5 * gate_fp32 * (1.0 - t * t) * sqrt_2_over_pi * (
            1.0 + 3.0 * 0.044715 * gate_sq
        )
    else:
        inv_sqrt_2 = 0.7071067811865476
        inv_sqrt_2pi = 0.3989422804014327
        erf_term = tl.math.erf(gate_fp32 * inv_sqrt_2)
        cdf = 0.5 * (1.0 + erf_term)
        activated = gate_fp32 * cdf
        grad = cdf + inv_sqrt_2pi * gate_fp32 * tl.exp(-0.5 * gate_fp32 * gate_fp32)

    activated = tl.where(gate_fp32 == -float("inf"), 0.0, activated)
    activated = tl.where(gate_fp32 == float("inf"), gate_fp32, activated)
    grad = tl.where(gate_fp32 == -float("inf"), 0.0, grad)
    grad = tl.where(gate_fp32 == float("inf"), 1.0, grad)
    return activated, grad


@triton.jit
def _geglu_forward_kernel(
    out_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
    activated = _gelu_activation(gate.to(tl.float32), APPROXIMATE_TANH)
    out = activated.to(gate.dtype) * up
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _geglu_backward_kernel(
    dgate_ptr,
    dup_ptr,
    dout_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    activated, activated_grad = _gelu_activation_and_grad(gate.to(tl.float32), APPROXIMATE_TANH)
    activated_cast_fp32 = activated.to(gate.dtype).to(tl.float32)
    dout_fp32 = dout.to(tl.float32)

    dup = dout_fp32 * activated_cast_fp32
    dgate = dout_fp32 * up.to(tl.float32) * activated_grad
    tl.store(dgate_ptr + offsets, dgate, mask=mask)
    tl.store(dup_ptr + offsets, dup, mask=mask)


@triton.jit
def _geglu_packed_forward_kernel(
    out_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        output_offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        output_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < n_elements

    row_idx = output_offsets // n_cols
    col_idx = output_offsets - row_idx * n_cols
    gate_offsets = row_idx * total_cols + col_idx
    up_offsets = gate_offsets + n_cols

    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + up_offsets, mask=mask, other=0.0)
    activated = _gelu_activation(gate.to(tl.float32), APPROXIMATE_TANH)
    out = activated.to(gate.dtype) * up
    tl.store(out_ptr + output_offsets, out, mask=mask)


@triton.jit
def _geglu_packed_backward_kernel(
    dgate_up_ptr,
    dout_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        output_offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        output_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < n_elements

    row_idx = output_offsets // n_cols
    col_idx = output_offsets - row_idx * n_cols
    gate_offsets = row_idx * total_cols + col_idx
    up_offsets = gate_offsets + n_cols

    dout = tl.load(dout_ptr + output_offsets, mask=mask, other=0.0)
    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + up_offsets, mask=mask, other=0.0)

    activated, activated_grad = _gelu_activation_and_grad(gate.to(tl.float32), APPROXIMATE_TANH)
    activated_cast_fp32 = activated.to(gate.dtype).to(tl.float32)
    dout_fp32 = dout.to(tl.float32)

    dup = dout_fp32 * activated_cast_fp32
    dgate = dout_fp32 * up.to(tl.float32) * activated_grad
    tl.store(dgate_up_ptr + gate_offsets, dgate, mask=mask)
    tl.store(dgate_up_ptr + up_offsets, dup, mask=mask)


def torch_geglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: str = "tanh",
) -> torch.Tensor:
    _check_same_shape_inputs(gate, up)
    _check_approximate(approximate)
    activated = torch.nn.functional.gelu(gate.float(), approximate=approximate).to(dtype=gate.dtype)
    return activated * up


def torch_geglu_packed_reference(
    gate_up: torch.Tensor,
    approximate: str = "tanh",
) -> torch.Tensor:
    _check_packed_input(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return torch_geglu_reference(gate, up, approximate)


def pack_geglu_gate_up_weight(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """Inputs: gate/up linear weights. Outputs: packed [2I, H] weight. Logic: enable one gate+up GEMM."""
    if gate_weight.shape != up_weight.shape:
        raise ValueError(f"gate_weight and up_weight must have the same shape, got {gate_weight.shape} and {up_weight.shape}")
    if gate_weight.device != up_weight.device:
        raise ValueError("gate_weight and up_weight must be on the same device")
    if gate_weight.dtype != up_weight.dtype:
        raise TypeError(f"gate_weight and up_weight must have the same dtype, got {gate_weight.dtype} and {up_weight.dtype}")
    if gate_weight.dim() != 2:
        raise ValueError(f"gate_weight and up_weight must be 2D, got {gate_weight.dim()}D")
    return torch.cat([gate_weight, up_weight], dim=0).contiguous()


def pack_geglu_gate_up_bias(
    gate_bias: torch.Tensor | None,
    up_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Inputs: optional gate/up biases. Outputs: packed bias or None. Logic: mirror packed weight layout."""
    if gate_bias is None and up_bias is None:
        return None
    if gate_bias is None or up_bias is None:
        raise ValueError("gate_bias and up_bias must either both be None or both be tensors for packed gate+up")
    if gate_bias.dim() != 1 or up_bias.dim() != 1:
        raise ValueError("gate_bias and up_bias must be 1D tensors")
    if gate_bias.shape != up_bias.shape:
        raise ValueError(f"gate_bias and up_bias must have the same shape, got {gate_bias.shape} and {up_bias.shape}")
    if gate_bias.device != up_bias.device:
        raise ValueError("gate_bias and up_bias must be on the same device")
    if gate_bias.dtype != up_bias.dtype:
        raise TypeError(f"gate_bias and up_bias must have the same dtype, got {gate_bias.dtype} and {up_bias.dtype}")
    return torch.cat([gate_bias, up_bias], dim=0).contiguous()


def torch_geglu_mlp_reference(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    approximate: str = "tanh",
) -> torch.Tensor:
    """Inputs: x and separate MLP weights. Outputs: PyTorch GEGLU MLP. Logic: correctness baseline."""
    _check_approximate(approximate)
    gate = F.linear(x, gate_weight, gate_bias)
    up = F.linear(x, up_weight, up_bias)
    hidden = torch_geglu_reference(gate, up, approximate)
    return F.linear(hidden, down_weight, down_bias)


def geglu_mlp(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor | None = None,
    up_weight: torch.Tensor | None = None,
    packed_gate_up_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    packed_gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    approximate: str = "tanh",
    preserve_inputs: bool = False,
) -> torch.Tensor:
    """Inputs: x and MLP weights. Outputs: down(gelu(gate) * up). Logic: use packed gate+up when provided."""
    _check_approximate(approximate)
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dimensions, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("x must be a floating point tensor")

    in_features = x.shape[-1]
    if packed_gate_up_weight is not None:
        if gate_weight is not None or up_weight is not None:
            raise ValueError("provide either packed_gate_up_weight or separate gate_weight/up_weight, not both")
        _check_linear_weight("packed_gate_up_weight", packed_gate_up_weight, in_features, x.device, x.dtype)
        if packed_gate_up_weight.shape[0] % 2 != 0:
            raise ValueError(f"packed_gate_up_weight output dimension must be even, got {packed_gate_up_weight.shape[0]}")
        intermediate = packed_gate_up_weight.shape[0] // 2
        _check_optional_bias("packed_gate_up_bias", packed_gate_up_bias, packed_gate_up_weight.shape[0], x.device, x.dtype)
        if gate_bias is not None or up_bias is not None:
            raise ValueError("use packed_gate_up_bias with packed_gate_up_weight")
        gate_up = F.linear(x, packed_gate_up_weight, packed_gate_up_bias)
        hidden = geglu_packed(gate_up, approximate=approximate, preserve_inputs=preserve_inputs)
    else:
        if gate_weight is None or up_weight is None:
            raise ValueError("separate path requires gate_weight and up_weight")
        _check_linear_weight("gate_weight", gate_weight, in_features, x.device, x.dtype)
        _check_linear_weight("up_weight", up_weight, in_features, x.device, x.dtype)
        if gate_weight.shape != up_weight.shape:
            raise ValueError(f"gate_weight and up_weight must have the same shape, got {gate_weight.shape} and {up_weight.shape}")
        intermediate = gate_weight.shape[0]
        _check_optional_bias("gate_bias", gate_bias, intermediate, x.device, x.dtype)
        _check_optional_bias("up_bias", up_bias, intermediate, x.device, x.dtype)
        if packed_gate_up_bias is not None:
            raise ValueError("packed_gate_up_bias requires packed_gate_up_weight")
        gate = F.linear(x, gate_weight, gate_bias)
        up = F.linear(x, up_weight, up_bias)
        hidden = geglu(gate, up, approximate=approximate, preserve_inputs=preserve_inputs)

    _check_linear_weight("down_weight", down_weight, intermediate, x.device, x.dtype)
    _check_optional_bias("down_bias", down_bias, down_weight.shape[0], x.device, x.dtype)
    return F.linear(hidden, down_weight, down_bias)


def geglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: str = "tanh",
):
    _check_same_shape_inputs(gate, up)
    _check_cuda_dtype(gate)
    approximate_tanh = _check_approximate(approximate)

    original_shape = gate.shape
    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    out = torch.empty_like(gate_flat)
    n_elements = gate_flat.numel()
    grid = (triton.cdiv(n_elements, FLAT_BLOCK_SIZE),)

    _geglu_forward_kernel[grid](
        out,
        gate_flat,
        up_flat,
        n_elements,
        BLOCK_SIZE=FLAT_BLOCK_SIZE,
        APPROXIMATE_TANH=approximate_tanh,
        LONG_INDEXING=_uses_long_indexing(n_elements),
        num_warps=4,
    )
    return out.view(original_shape), gate_flat, up_flat


def geglu_backward(
    dout: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: str = "tanh",
    preserve_inputs: bool = False,
):
    approximate_tanh = _check_approximate(approximate)
    original_shape = dout.shape
    dout_flat = dout.contiguous().view(-1)
    n_elements = dout_flat.numel()
    grid = (triton.cdiv(n_elements, FLAT_BLOCK_SIZE),)

    dgate = torch.empty_like(gate) if preserve_inputs else gate
    dup = torch.empty_like(up) if preserve_inputs else up

    _geglu_backward_kernel[grid](
        dgate,
        dup,
        dout_flat,
        gate,
        up,
        n_elements,
        BLOCK_SIZE=FLAT_BLOCK_SIZE,
        APPROXIMATE_TANH=approximate_tanh,
        LONG_INDEXING=_uses_long_indexing(n_elements),
        num_warps=4,
    )
    return dgate.view(original_shape), dup.view(original_shape)


def geglu_packed_forward(
    gate_up: torch.Tensor,
    approximate: str = "tanh",
):
    _check_packed_input(gate_up)
    _check_cuda_dtype(gate_up)
    approximate_tanh = _check_approximate(approximate)

    original_shape = gate_up.shape
    total_cols = original_shape[-1]
    n_cols = total_cols // 2
    gate_up_2d = gate_up.contiguous().view(-1, total_cols)
    n_elements = gate_up_2d.shape[0] * n_cols
    out = torch.empty((n_elements,), device=gate_up.device, dtype=gate_up.dtype)
    grid = (triton.cdiv(n_elements, FLAT_BLOCK_SIZE),)

    _geglu_packed_forward_kernel[grid](
        out,
        gate_up_2d,
        n_cols=n_cols,
        total_cols=total_cols,
        n_elements=n_elements,
        BLOCK_SIZE=FLAT_BLOCK_SIZE,
        APPROXIMATE_TANH=approximate_tanh,
        LONG_INDEXING=_uses_long_indexing(n_elements),
        num_warps=4,
    )
    return out.view(*original_shape[:-1], n_cols), gate_up_2d


def geglu_packed_backward(
    dout: torch.Tensor,
    gate_up: torch.Tensor,
    input_shape: torch.Size,
    approximate: str = "tanh",
    preserve_inputs: bool = False,
):
    approximate_tanh = _check_approximate(approximate)
    n_cols = dout.shape[-1]
    total_cols = input_shape[-1]
    dout_flat = dout.contiguous().view(-1)
    n_elements = dout_flat.numel()
    grid = (triton.cdiv(n_elements, FLAT_BLOCK_SIZE),)

    dgate_up = torch.empty_like(gate_up) if preserve_inputs else gate_up

    _geglu_packed_backward_kernel[grid](
        dgate_up,
        dout_flat,
        gate_up,
        n_cols=n_cols,
        total_cols=total_cols,
        n_elements=n_elements,
        BLOCK_SIZE=FLAT_BLOCK_SIZE,
        APPROXIMATE_TANH=approximate_tanh,
        LONG_INDEXING=_uses_long_indexing(n_elements),
        num_warps=4,
    )
    return dgate_up.view(input_shape)


class ForgeGEGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        approximate: str = "tanh",
        preserve_inputs: bool = False,
    ):
        y, gate_flat, up_flat = geglu_forward(gate, up, approximate)
        ctx.save_for_backward(gate_flat, up_flat)
        ctx.approximate = approximate
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate, dup = geglu_backward(dout, gate, up, ctx.approximate, ctx.preserve_inputs)
        return dgate, dup, None, None


class ForgePackedGEGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate_up: torch.Tensor,
        approximate: str = "tanh",
        preserve_inputs: bool = False,
    ):
        y, gate_up_2d = geglu_packed_forward(gate_up, approximate)
        ctx.save_for_backward(gate_up_2d)
        ctx.input_shape = gate_up.shape
        ctx.approximate = approximate
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        dgate_up = geglu_packed_backward(
            dout,
            gate_up,
            ctx.input_shape,
            ctx.approximate,
            ctx.preserve_inputs,
        )
        return dgate_up, None, None


def geglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: str = "tanh",
    preserve_inputs: bool = False,
) -> torch.Tensor:
    _check_same_shape_inputs(gate, up)
    _check_approximate(approximate)
    if not gate.is_cuda:
        return torch_geglu_reference(gate, up, approximate)
    return ForgeGEGLUFunction.apply(gate, up, approximate, preserve_inputs)


def geglu_packed(
    gate_up: torch.Tensor,
    approximate: str = "tanh",
    preserve_inputs: bool = False,
) -> torch.Tensor:
    _check_packed_input(gate_up)
    _check_approximate(approximate)
    if not gate_up.is_cuda:
        return torch_geglu_packed_reference(gate_up, approximate)
    return ForgePackedGEGLUFunction.apply(gate_up, approximate, preserve_inputs)

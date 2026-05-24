from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.geglu import geglu
from kernels.geglu import geglu_mlp
from kernels.geglu import geglu_packed
from kernels.geglu import pack_geglu_gate_up_bias
from kernels.geglu import pack_geglu_gate_up_weight
from kernels.geglu import torch_geglu_mlp_reference
from kernels.geglu import torch_geglu_packed_reference
from kernels.geglu import torch_geglu_reference


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_dtype_available(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    if dtype is torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


@pytest.mark.parametrize("shape", [(2, 8, 128), (1, 3, 257), (4, 5, 1024), (1, 2, 11008)])
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.parametrize("preserve_inputs", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-5, 2e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEGLU path requires CUDA")
def test_geglu_forward_backward(shape, approximate, preserve_inputs, dtype, atol, rtol):
    if not _cuda_dtype_available(dtype):
        pytest.skip(f"{dtype} is not supported on this CUDA device")

    torch.manual_seed(0)
    gate_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    up_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_forge = gate_ref.detach().clone().requires_grad_(True)
    up_forge = up_ref.detach().clone().requires_grad_(True)
    gate_before = gate_forge.detach().clone()
    up_before = up_forge.detach().clone()

    out_ref = torch_geglu_reference(gate_ref, up_ref, approximate)
    out_forge = geglu(gate_forge, up_forge, approximate, preserve_inputs)
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)

    torch.testing.assert_close(gate_forge.grad, gate_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(up_forge.grad, up_ref.grad, atol=atol, rtol=rtol)

    if preserve_inputs:
        torch.testing.assert_close(gate_forge.detach(), gate_before, atol=0, rtol=0)
        torch.testing.assert_close(up_forge.detach(), up_before, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(2, 8, 256), (1, 3, 514)])
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.parametrize("preserve_inputs", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-5, 2e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton packed GEGLU path requires CUDA")
def test_geglu_packed_forward_backward(shape, approximate, preserve_inputs, dtype, atol, rtol):
    if not _cuda_dtype_available(dtype):
        pytest.skip(f"{dtype} is not supported on this CUDA device")

    torch.manual_seed(1)
    gate_up_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_up_forge = gate_up_ref.detach().clone().requires_grad_(True)
    gate_up_before = gate_up_forge.detach().clone()

    out_ref = torch_geglu_packed_reference(gate_up_ref, approximate)
    out_forge = geglu_packed(gate_up_forge, approximate, preserve_inputs)
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)

    torch.testing.assert_close(gate_up_forge.grad, gate_up_ref.grad, atol=atol, rtol=rtol)
    if preserve_inputs:
        torch.testing.assert_close(gate_up_forge.detach(), gate_up_before, atol=0, rtol=0)


@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEGLU path requires CUDA")
def test_geglu_forward_special_values(approximate):
    gate = torch.tensor(
        [[[0.0, float("inf"), float("-inf"), float("nan"), 20.0, -20.0]]],
        device=DEVICE,
        dtype=torch.float32,
    )
    up = torch.tensor([[[1.0, 2.0, 3.0, 4.0, -1.0, -2.0]]], device=DEVICE, dtype=torch.float32)

    out_ref = torch_geglu_reference(gate, up, approximate)
    out_forge = geglu(gate, up, approximate, preserve_inputs=True)
    torch.testing.assert_close(out_forge, out_ref, atol=1e-12, rtol=1e-7, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEGLU path requires CUDA")
def test_geglu_flat_path_supports_hidden_above_row_limit():
    torch.manual_seed(2)
    shape = (1, 1, 65537)
    gate_ref = torch.randn(*shape, device=DEVICE, dtype=torch.float32, requires_grad=True)
    up_ref = torch.randn(*shape, device=DEVICE, dtype=torch.float32, requires_grad=True)
    gate_forge = gate_ref.detach().clone().requires_grad_(True)
    up_forge = up_ref.detach().clone().requires_grad_(True)

    out_ref = torch_geglu_reference(gate_ref, up_ref, "tanh")
    out_forge = geglu(gate_forge, up_forge, "tanh", preserve_inputs=True)
    torch.testing.assert_close(out_forge, out_ref, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)
    torch.testing.assert_close(gate_forge.grad, gate_ref.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(up_forge.grad, up_ref.grad, atol=2e-5, rtol=2e-5)


def test_geglu_cpu_fallback_matches_reference():
    torch.manual_seed(0)
    gate = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    up = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)

    out = geglu(gate, up, approximate="none")
    out_ref = torch_geglu_reference(gate_ref, up_ref, approximate="none")
    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-5, 2e-5),
        (torch.float16, 3e-2, 3e-2),
        (torch.bfloat16, 4e-2, 4e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEGLU MLP path requires CUDA")
def test_geglu_mlp_packed_matches_separate_and_reference(use_bias, approximate, dtype, atol, rtol):
    if not _cuda_dtype_available(dtype):
        pytest.skip(f"{dtype} is not supported on this CUDA device")

    torch.manual_seed(3)
    batch, seq, hidden, intermediate = 2, 7, 128, 257
    x_ref = torch.randn(batch, seq, hidden, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_weight_ref = torch.randn(intermediate, hidden, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
    up_weight_ref = torch.randn(intermediate, hidden, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
    down_weight_ref = torch.randn(hidden, intermediate, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
    gate_weight_ref.retain_grad()
    up_weight_ref.retain_grad()
    down_weight_ref.retain_grad()

    if use_bias:
        gate_bias_ref = torch.randn(intermediate, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
        up_bias_ref = torch.randn(intermediate, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
        down_bias_ref = torch.randn(hidden, device=DEVICE, dtype=dtype, requires_grad=True) * 0.02
        gate_bias_ref.retain_grad()
        up_bias_ref.retain_grad()
        down_bias_ref.retain_grad()
    else:
        gate_bias_ref = up_bias_ref = down_bias_ref = None

    x_sep = x_ref.detach().clone().requires_grad_(True)
    gate_weight_sep = gate_weight_ref.detach().clone().requires_grad_(True)
    up_weight_sep = up_weight_ref.detach().clone().requires_grad_(True)
    down_weight_sep = down_weight_ref.detach().clone().requires_grad_(True)
    gate_bias_sep = gate_bias_ref.detach().clone().requires_grad_(True) if gate_bias_ref is not None else None
    up_bias_sep = up_bias_ref.detach().clone().requires_grad_(True) if up_bias_ref is not None else None
    down_bias_sep = down_bias_ref.detach().clone().requires_grad_(True) if down_bias_ref is not None else None

    x_packed = x_ref.detach().clone().requires_grad_(True)
    packed_weight = pack_geglu_gate_up_weight(
        gate_weight_ref.detach(),
        up_weight_ref.detach(),
    ).requires_grad_(True)
    packed_bias = pack_geglu_gate_up_bias(
        gate_bias_ref.detach() if gate_bias_ref is not None else None,
        up_bias_ref.detach() if up_bias_ref is not None else None,
    )
    if packed_bias is not None:
        packed_bias.requires_grad_(True)
    down_weight_packed = down_weight_ref.detach().clone().requires_grad_(True)
    down_bias_packed = down_bias_ref.detach().clone().requires_grad_(True) if down_bias_ref is not None else None

    out_ref = torch_geglu_mlp_reference(
        x_ref,
        gate_weight_ref,
        up_weight_ref,
        down_weight_ref,
        gate_bias_ref,
        up_bias_ref,
        down_bias_ref,
        approximate,
    )
    out_sep = geglu_mlp(
        x_sep,
        down_weight_sep,
        gate_weight=gate_weight_sep,
        up_weight=up_weight_sep,
        gate_bias=gate_bias_sep,
        up_bias=up_bias_sep,
        down_bias=down_bias_sep,
        approximate=approximate,
        preserve_inputs=False,
    )
    out_packed = geglu_mlp(
        x_packed,
        down_weight_packed,
        packed_gate_up_weight=packed_weight,
        packed_gate_up_bias=packed_bias,
        down_bias=down_bias_packed,
        approximate=approximate,
        preserve_inputs=False,
    )

    torch.testing.assert_close(out_sep, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_packed, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_sep.backward(grad)
    out_packed.backward(grad)

    torch.testing.assert_close(x_sep.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_packed.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(gate_weight_sep.grad, gate_weight_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(up_weight_sep.grad, up_weight_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(down_weight_sep.grad, down_weight_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(packed_weight.grad[:intermediate], gate_weight_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(packed_weight.grad[intermediate:], up_weight_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(down_weight_packed.grad, down_weight_ref.grad, atol=atol, rtol=rtol)

    if use_bias:
        torch.testing.assert_close(gate_bias_sep.grad, gate_bias_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(up_bias_sep.grad, up_bias_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(down_bias_sep.grad, down_bias_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(packed_bias.grad[:intermediate], gate_bias_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(packed_bias.grad[intermediate:], up_bias_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(down_bias_packed.grad, down_bias_ref.grad, atol=atol, rtol=rtol)


def test_geglu_rejects_invalid_inputs():
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)
    with pytest.raises(ValueError, match="same shape"):
        geglu(gate, up)

    gate_up = torch.randn(2, 5)
    with pytest.raises(ValueError, match="even"):
        geglu_packed(gate_up)

    with pytest.raises(ValueError, match="approximate"):
        geglu(torch.randn(2, 3), torch.randn(2, 3), approximate="fast")

    with pytest.raises(TypeError, match="floating point"):
        geglu(torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.int64))

    with pytest.raises(ValueError, match="either packed_gate_up_weight or separate"):
        geglu_mlp(
            torch.randn(2, 3),
            torch.randn(3, 4),
            gate_weight=torch.randn(4, 3),
            up_weight=torch.randn(4, 3),
            packed_gate_up_weight=torch.randn(8, 3),
        )

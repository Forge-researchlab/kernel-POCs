from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.swiglu import swiglu
from kernels.swiglu import swiglu_packed
from kernels.swiglu import torch_swiglu_packed_reference
from kernels.swiglu import torch_swiglu_reference


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_dtype_available(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    if dtype is torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


@pytest.mark.parametrize("shape", [(2, 8, 128), (1, 3, 257), (4, 5, 1024), (1, 2, 11008)])
@pytest.mark.parametrize("gate_multiplier, down_multiplier", [(1.0, 1.0), (0.7, 1.3)])
@pytest.mark.parametrize("preserve_inputs", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton SwiGLU path requires CUDA")
def test_swiglu_forward_backward(shape, gate_multiplier, down_multiplier, preserve_inputs, dtype, atol, rtol):
    if not _cuda_dtype_available(dtype):
        pytest.skip(f"{dtype} is not supported on this CUDA device")

    torch.manual_seed(0)
    gate_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    up_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_forge = gate_ref.detach().clone().requires_grad_(True)
    up_forge = up_ref.detach().clone().requires_grad_(True)
    gate_before = gate_forge.detach().clone()
    up_before = up_forge.detach().clone()

    out_ref = torch_swiglu_reference(gate_ref, up_ref, gate_multiplier, down_multiplier)
    out_forge = swiglu(gate_forge, up_forge, gate_multiplier, down_multiplier, preserve_inputs)
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
@pytest.mark.parametrize("gate_multiplier, down_multiplier", [(1.0, 1.0), (1.5, 0.5)])
@pytest.mark.parametrize("preserve_inputs", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton packed SwiGLU path requires CUDA")
def test_swiglu_packed_forward_backward(shape, gate_multiplier, down_multiplier, preserve_inputs, dtype, atol, rtol):
    if not _cuda_dtype_available(dtype):
        pytest.skip(f"{dtype} is not supported on this CUDA device")

    torch.manual_seed(1)
    gate_up_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_up_forge = gate_up_ref.detach().clone().requires_grad_(True)
    gate_up_before = gate_up_forge.detach().clone()

    out_ref = torch_swiglu_packed_reference(gate_up_ref, gate_multiplier, down_multiplier)
    out_forge = swiglu_packed(gate_up_forge, gate_multiplier, down_multiplier, preserve_inputs)
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)

    torch.testing.assert_close(gate_up_forge.grad, gate_up_ref.grad, atol=atol, rtol=rtol)
    if preserve_inputs:
        torch.testing.assert_close(gate_up_forge.detach(), gate_up_before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton SwiGLU path requires CUDA")
def test_swiglu_special_values_forward():
    gate = torch.tensor(
        [[[0.0, float("inf"), float("-inf"), float("nan"), 20.0, -20.0]]],
        device=DEVICE,
        dtype=torch.float32,
    )
    up = torch.tensor([[[1.0, 2.0, 3.0, 4.0, -1.0, -2.0]]], device=DEVICE, dtype=torch.float32)

    out_ref = torch_swiglu_reference(gate, up)
    out_forge = swiglu(gate, up, preserve_inputs=True)
    torch.testing.assert_close(out_forge, out_ref, atol=0, rtol=0, equal_nan=True)


def test_swiglu_cpu_fallback_matches_reference():
    torch.manual_seed(0)
    gate = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    up = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)

    out = swiglu(gate, up, gate_multiplier=0.7, down_multiplier=1.3)
    out_ref = torch_swiglu_reference(gate_ref, up_ref, gate_multiplier=0.7, down_multiplier=1.3)
    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)


def test_swiglu_rejects_invalid_inputs():
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)
    with pytest.raises(ValueError, match="same shape"):
        swiglu(gate, up)

    gate_up = torch.randn(2, 5)
    with pytest.raises(ValueError, match="even"):
        swiglu_packed(gate_up)

    with pytest.raises(TypeError, match="Python scalar"):
        swiglu(torch.randn(2, 3), torch.randn(2, 3), gate_multiplier=torch.tensor(1.0))

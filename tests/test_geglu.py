from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.geglu import geglu
from kernels.geglu import geglu_packed
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

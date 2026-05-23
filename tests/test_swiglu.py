from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.swiglu import ForgeSwiGLUFunction
from kernels.swiglu import swiglu
from kernels.swiglu import torch_swiglu_reference


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("shape", [(2, 8, 128), (1, 3, 257), (4, 5, 1024)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton SwiGLU path requires CUDA")
def test_swiglu_forward_backward(shape, dtype, atol, rtol):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(0)
    gate_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    up_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    gate_forge = gate_ref.detach().clone().requires_grad_(True)
    up_forge = up_ref.detach().clone().requires_grad_(True)

    out_ref = torch_swiglu_reference(gate_ref, up_ref)
    out_forge = ForgeSwiGLUFunction.apply(gate_forge, up_forge)
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)

    torch.testing.assert_close(gate_forge.grad, gate_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(up_forge.grad, up_ref.grad, atol=atol, rtol=rtol)


def test_swiglu_cpu_fallback_matches_reference():
    torch.manual_seed(0)
    gate = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    up = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)

    out = swiglu(gate, up)
    out_ref = torch_swiglu_reference(gate_ref, up_ref)
    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)

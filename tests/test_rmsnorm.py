from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.rmsnorm import ForgeRMSNormFunction
from kernels.rmsnorm import rmsnorm
from kernels.rmsnorm import torch_rmsnorm_reference

# v2 smoke for the Gemma offset path — pulled directly so we don't trigger the
# v3 import (autotune compilation) for this lightweight test.
from kernels.rmsnorm.forge_rmsnorm_v2 import (
    apply_rmsnorm_v2,
    torch_rmsnorm_reference_v2,
)


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RMSNorm path requires CUDA")
def test_rmsnorm_forward_backward(shape, dtype, atol, rtol):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(0)
    x_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    x_forge = x_ref.detach().clone().requires_grad_(True)
    weight_ref = torch.randn(shape[-1], device=DEVICE, dtype=dtype, requires_grad=True)
    weight_forge = weight_ref.detach().clone().requires_grad_(True)

    out_ref = torch_rmsnorm_reference(x_ref, weight_ref)
    out_forge = ForgeRMSNormFunction.apply(x_forge, weight_forge, 1e-6)
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)

    torch.testing.assert_close(x_forge.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(weight_forge.grad, weight_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(2, 8, 128), (4, 32, 2304)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RMSNorm requires CUDA")
def test_rmsnorm_v2_gemma_offset(shape, dtype, atol, rtol):
    """Gemma path: offset=1.0, casting_mode='gemma' matches the all-fp32 oracle."""
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this CUDA device")
    torch.manual_seed(0)
    x_ref = torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)
    x_forge = x_ref.detach().clone().requires_grad_(True)
    # Gemma-style weight init: near zero, with the +1 offset baked in by the kernel.
    weight_ref = (0.05 * torch.randn(shape[-1], device=DEVICE, dtype=dtype)).requires_grad_(True)
    weight_forge = weight_ref.detach().clone().requires_grad_(True)

    out_ref = torch_rmsnorm_reference_v2(x_ref, weight_ref, 1e-6, offset=1.0, casting_mode="gemma")
    out_forge = apply_rmsnorm_v2(x_forge, weight_forge, eps=1e-6, offset=1.0, casting_mode="gemma")
    torch.testing.assert_close(out_forge, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_forge.backward(grad)
    torch.testing.assert_close(x_forge.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(weight_forge.grad, weight_ref.grad, atol=atol, rtol=rtol)


def test_rmsnorm_cpu_fallback_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 17, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(17, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    out = rmsnorm(x, weight)
    out_ref = torch_rmsnorm_reference(x_ref, weight_ref)
    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)

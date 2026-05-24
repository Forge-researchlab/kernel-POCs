"""fp64 gradcheck for both LayerNorm variants.

Liger: gradcheck against (X, W, B) — all three should be differentiated.
Unsloth: gradcheck against X only (W, B frozen). Documents the dW=dB=None
contract via an xfail that runs gradcheck on the full input set.
"""
from __future__ import annotations

import pytest
import torch

from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)


GRADCHECK_SHAPE = (2, 4, 8)
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-5
GRADCHECK_RTOL = 1e-3


def _mk_inputs(requires_grad_x: bool, requires_grad_w: bool, requires_grad_b: bool):
    B, S, H = GRADCHECK_SHAPE
    X = torch.randn(B, S, H, dtype=torch.float64, device="cuda",
                    requires_grad=requires_grad_x)
    W = (torch.randn(H, dtype=torch.float64, device="cuda") * 0.5 + 1.0
        ).detach().requires_grad_(requires_grad_w)
    Bp = (torch.randn(H, dtype=torch.float64, device="cuda") * 0.1
        ).detach().requires_grad_(requires_grad_b)
    return X, W, Bp


def test_liger_gradcheck_full():
    """Liger differentiates X, W, B — gradcheck against all three."""
    X, W, B = _mk_inputs(True, True, True)
    assert torch.autograd.gradcheck(
        ForgeLayerNormLigerFunction.apply,
        (X, W, B, GRADCHECK_EPS),
        eps=1e-6,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        raise_exception=True,
    )


def test_unsloth_gradcheck_x_only():
    """Unsloth only differentiates X; W, B must be frozen for gradcheck to pass."""
    X, W, B = _mk_inputs(True, False, False)
    assert torch.autograd.gradcheck(
        ForgeLayerNormUnslothFunction.apply,
        (X, W, B, GRADCHECK_EPS),
        eps=1e-6,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        raise_exception=True,
    )


@pytest.mark.xfail(
    reason="Unsloth backward returns None for dW/dB by design; full gradcheck must fail",
    strict=True,
)
def test_unsloth_full_gradcheck_documents_none_contract():
    """Document the contract: gradcheck with W, B requiring grad must fail."""
    X, W, B = _mk_inputs(True, True, True)
    torch.autograd.gradcheck(
        ForgeLayerNormUnslothFunction.apply,
        (X, W, B, GRADCHECK_EPS),
        eps=1e-6,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        raise_exception=True,
    )

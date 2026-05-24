"""Liger vs Unsloth: forward equivalence, dX equivalence with frozen W/B,
saved-tensor shapes, and the in-place dY-overwrite contract of Unsloth bwd.
"""
from __future__ import annotations

import pytest
import torch

from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import make_inputs, clone_leaf


SHAPE = (4, 2048, 4096)
EPS = 1e-6


def test_forward_outputs_match_in_fp32():
    """Both variants implement the same math — fp32 outputs should match tightly."""
    X, W, B, _ = make_inputs(SHAPE, torch.float32, requires_grad=False)
    out_l = ForgeLayerNormLigerFunction.apply(X, W, B, EPS)
    out_u = ForgeLayerNormUnslothFunction.apply(X, W, B, EPS)
    torch.testing.assert_close(out_l, out_u, rtol=1e-5, atol=1e-5)


def test_dx_matches_with_frozen_wb():
    """When W, B are frozen, dX should be the same from both variants."""
    X, W, B, dY = make_inputs(SHAPE, torch.float32, requires_grad=False)

    X_l = clone_leaf(X, requires_grad=True)
    W_l = clone_leaf(W, requires_grad=False)
    B_l = clone_leaf(B, requires_grad=False)
    out_l = ForgeLayerNormLigerFunction.apply(X_l, W_l, B_l, EPS)
    out_l.backward(dY)

    X_u = clone_leaf(X, requires_grad=True)
    W_u = clone_leaf(W, requires_grad=False)
    B_u = clone_leaf(B, requires_grad=False)
    out_u = ForgeLayerNormUnslothFunction.apply(X_u, W_u, B_u, EPS)
    out_u.backward(dY.clone())

    torch.testing.assert_close(X_l.grad, X_u.grad, rtol=1e-5, atol=1e-5)


def test_saved_tensors_shape():
    """Document the saved-tensor signatures of each variant.

    Both save [X_flat, W, B, stat1, stat2] where stat1/stat2 are per-row fp32.
    Liger: (Mean, RSTD). Unsloth: (R=inv_var, Mu=mean). Shapes match.
    """
    X, W, B, _ = make_inputs(SHAPE, torch.float32)
    X.requires_grad_(True)
    W = W.detach().requires_grad_(True)
    B = B.detach().requires_grad_(True)

    out_l = ForgeLayerNormLigerFunction.apply(X, W, B, EPS)
    saved_l = out_l.grad_fn.saved_tensors
    assert len(saved_l) == 5
    X_flat_l, W_s_l, B_s_l, stat1_l, stat2_l = saved_l
    assert stat1_l.dtype == torch.float32
    assert stat2_l.dtype == torch.float32
    assert stat1_l.shape == (SHAPE[0] * SHAPE[1],)
    assert stat2_l.shape == (SHAPE[0] * SHAPE[1],)

    X2 = clone_leaf(X, requires_grad=True)
    W2 = clone_leaf(W, requires_grad=True)
    B2 = clone_leaf(B, requires_grad=True)
    out_u = ForgeLayerNormUnslothFunction.apply(X2, W2, B2, EPS)
    saved_u = out_u.grad_fn.saved_tensors
    assert len(saved_u) == 5
    _, _, _, stat1_u, stat2_u = saved_u
    assert stat1_u.shape == (SHAPE[0] * SHAPE[1],)
    assert stat2_u.shape == (SHAPE[0] * SHAPE[1],)


def test_unsloth_backward_overwrites_dy_buffer():
    """Unsloth bwd writes dX in-place into the dY buffer — verify by checking
    that the upstream gradient tensor has been mutated to equal X.grad."""
    X, W, B, _ = make_inputs(SHAPE, torch.float32, requires_grad=False)
    eps = EPS

    X_f = clone_leaf(X, requires_grad=True)
    W_f = clone_leaf(W, requires_grad=False)
    B_f = clone_leaf(B, requires_grad=False)
    out = ForgeLayerNormUnslothFunction.apply(X_f, W_f, B_f, eps)

    # Manually build a dY tensor we can keep a reference to. Use the
    # 2D .view(-1, H) form because that is what bwd's `dY_flat.contiguous()`
    # returns (and on contiguous input, contiguous() returns the same tensor).
    dY_2d = torch.randn(SHAPE[0] * SHAPE[1], SHAPE[2], dtype=torch.float32,
                        device="cuda").contiguous()
    dY_before = dY_2d.clone()
    dY_3d = dY_2d.view(*SHAPE)

    out.backward(dY_3d)

    # After bwd, dY_2d (the buffer the kernel wrote into) should differ from
    # its pre-bwd contents — i.e. it now holds dX.
    assert not torch.equal(dY_2d, dY_before), \
        "Unsloth backward should overwrite dY in-place"
    torch.testing.assert_close(dY_2d.view(*SHAPE), X_f.grad, rtol=0.0, atol=0.0)

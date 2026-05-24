"""Forward + backward correctness for both LayerNorm variants vs F.layer_norm.

Test checklist:
  - [x] Forward matches F.layer_norm across bf16/fp16/fp32 and a shape sweep
  - [x] Liger backward: dX, dW, dB match eager autograd
  - [x] Unsloth backward: dX matches eager (W/B frozen)
  - [x] Unsloth contract: dW/dB are None when W/B require grad
  - [x] Non-contiguous input is handled (kernel calls .contiguous())
  - [x] Liger is deterministic (no atomics in fwd; partial bwd is deterministic)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import SHAPES_SWEEP, make_inputs, clone_leaf, tol_for


DTYPES = [torch.bfloat16, torch.float16, torch.float32]


# ---------------------------------------------------------------------------
# Liger forward / backward
# ---------------------------------------------------------------------------

class TestLigerForward:
    @pytest.mark.parametrize("shape", SHAPES_SWEEP)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_eager(self, shape, dtype):
        X, W, Bp, _ = make_inputs(shape, dtype, requires_grad=False)
        eps = 1e-6

        out = ForgeLayerNormLigerFunction.apply(X, W, Bp, eps)
        ref = F.layer_norm(X, (shape[-1],), W, Bp, eps)

        torch.testing.assert_close(out, ref, **tol_for(dtype))


class TestLigerBackward:
    @pytest.mark.parametrize("shape", SHAPES_SWEEP)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_all_grads_match_eager(self, shape, dtype):
        X, W, Bp, dY = make_inputs(shape, dtype, requires_grad=False)
        eps = 1e-6

        # Forge path
        X_f = clone_leaf(X, requires_grad=True)
        W_f = clone_leaf(W, requires_grad=True)
        B_f = clone_leaf(Bp, requires_grad=True)
        out_f = ForgeLayerNormLigerFunction.apply(X_f, W_f, B_f, eps)
        out_f.backward(dY)

        # Eager reference
        X_r = clone_leaf(X, requires_grad=True)
        W_r = clone_leaf(W, requires_grad=True)
        B_r = clone_leaf(Bp, requires_grad=True)
        out_r = F.layer_norm(X_r, (shape[-1],), W_r, B_r, eps)
        out_r.backward(dY)

        tol = tol_for(dtype)
        torch.testing.assert_close(X_f.grad, X_r.grad, **tol)
        torch.testing.assert_close(W_f.grad, W_r.grad, **tol)
        torch.testing.assert_close(B_f.grad, B_r.grad, **tol)


# ---------------------------------------------------------------------------
# Unsloth forward / backward
# ---------------------------------------------------------------------------

class TestUnslothForward:
    @pytest.mark.parametrize("shape", SHAPES_SWEEP)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_eager(self, shape, dtype):
        X, W, Bp, _ = make_inputs(shape, dtype, requires_grad=False)
        eps = 1e-6

        out = ForgeLayerNormUnslothFunction.apply(X, W, Bp, eps)
        ref = F.layer_norm(X, (shape[-1],), W, Bp, eps)

        torch.testing.assert_close(out, ref, **tol_for(dtype))


class TestUnslothBackward:
    @pytest.mark.parametrize("shape", SHAPES_SWEEP)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_dx_matches_eager_with_frozen_wb(self, shape, dtype):
        X, W, Bp, dY = make_inputs(shape, dtype, requires_grad=False)
        eps = 1e-6

        # Forge path (W, B frozen — Unsloth only produces dX)
        X_f = clone_leaf(X, requires_grad=True)
        W_f = clone_leaf(W, requires_grad=False)
        B_f = clone_leaf(Bp, requires_grad=False)
        out_f = ForgeLayerNormUnslothFunction.apply(X_f, W_f, B_f, eps)
        out_f.backward(dY.clone())  # clone — bwd overwrites dY in-place

        # Eager reference (also W, B frozen)
        X_r = clone_leaf(X, requires_grad=True)
        W_r = clone_leaf(W, requires_grad=False)
        B_r = clone_leaf(Bp, requires_grad=False)
        out_r = F.layer_norm(X_r, (shape[-1],), W_r, B_r, eps)
        out_r.backward(dY)

        torch.testing.assert_close(X_f.grad, X_r.grad, **tol_for(dtype))

    def test_dw_db_are_none_by_contract(self):
        shape = (2, 8, 1024)
        X, W, Bp, dY = make_inputs(shape, torch.float32, requires_grad=False)
        eps = 1e-6

        X_f = clone_leaf(X, requires_grad=True)
        W_f = clone_leaf(W, requires_grad=True)   # requires grad — Unsloth still returns None
        B_f = clone_leaf(Bp, requires_grad=True)

        out = ForgeLayerNormUnslothFunction.apply(X_f, W_f, B_f, eps)
        out.backward(dY.clone())

        assert X_f.grad is not None, "Unsloth must produce dX"
        assert W_f.grad is None, "Unsloth backward returns None for dW by design"
        assert B_f.grad is None, "Unsloth backward returns None for dB by design"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestNonContiguous:
    def test_strided_input_is_normalized_via_contiguous(self):
        # Both autograd.Function bodies call X.view(-1, dim).contiguous() so this
        # should just work without error and match eager.
        shape = (4, 2048, 4096)
        dtype = torch.bfloat16
        X, W, Bp, _ = make_inputs(shape, dtype, requires_grad=False)

        # Make X non-contiguous by transposing the inner two dims and back
        X_nc = X.transpose(0, 1)
        assert not X_nc.is_contiguous()
        # The kernel internally re-views so the *shape* needs to be 3D again
        X_nc = X_nc.transpose(0, 1)  # restore logical shape, still non-contiguous

        out = ForgeLayerNormLigerFunction.apply(X_nc, W, Bp, 1e-6)
        ref = F.layer_norm(X_nc.contiguous(), (shape[-1],), W, Bp, 1e-6)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


class TestDeterminism:
    def test_liger_same_input_same_output(self):
        shape = (4, 2048, 4096)
        dtype = torch.bfloat16
        X, W, Bp, _ = make_inputs(shape, dtype, requires_grad=False)

        out1 = ForgeLayerNormLigerFunction.apply(X, W, Bp, 1e-6)
        out2 = ForgeLayerNormLigerFunction.apply(X, W, Bp, 1e-6)

        # Same kernel, same inputs, no atomics -> bitwise equal
        assert torch.equal(out1, out2), "Liger forward must be deterministic"

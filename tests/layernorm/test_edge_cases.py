"""Edge cases: eps sweep, minimal shapes, non-power-of-2 hidden, near-block-cap,
constant input, state_dict round-trip."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.layernorm import (
    ForgeLayerNormLiger,
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnsloth,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import make_inputs, clone_leaf, TOL_BF16


VARIANTS = {
    "liger": ForgeLayerNormLigerFunction,
    "unsloth": ForgeLayerNormUnslothFunction,
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("eps", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
def test_eps_sweep(variant, eps):
    shape = (2, 8, 1024)
    X, W, B, _ = make_inputs(shape, torch.bfloat16, requires_grad=False)
    out = VARIANTS[variant].apply(X, W, B, eps)
    ref = F.layer_norm(X, (shape[-1],), W, B, eps)
    torch.testing.assert_close(out, ref, **TOL_BF16)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_minimal_shape_1_1_128(variant):
    X, W, B, _ = make_inputs((1, 1, 128), torch.bfloat16, requires_grad=False)
    out = VARIANTS[variant].apply(X, W, B, 1e-6)
    ref = F.layer_norm(X, (128,), W, B, 1e-6)
    torch.testing.assert_close(out, ref, **TOL_BF16)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("hidden", [3, 17, 4097, 8193])
def test_non_power_of_2_hidden(variant, hidden):
    shape = (2, 4, hidden)
    X, W, B, _ = make_inputs(shape, torch.float32, requires_grad=False)
    out = VARIANTS[variant].apply(X, W, B, 1e-6)
    ref = F.layer_norm(X, (hidden,), W, B, 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_single_row(variant):
    shape = (1, 1, 4096)
    X, W, B, _ = make_inputs(shape, torch.bfloat16, requires_grad=False)
    out = VARIANTS[variant].apply(X, W, B, 1e-6)
    ref = F.layer_norm(X, (4096,), W, B, 1e-6)
    torch.testing.assert_close(out, ref, **TOL_BF16)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_large_hidden_near_block_cap(variant):
    # BLOCK_SIZE is capped at 65536 in _calculate_settings; hidden=32768 sits
    # well under the cap but exercises a large per-row reduction.
    shape = (1, 2, 32768)
    X, W, B, _ = make_inputs(shape, torch.bfloat16, requires_grad=False)
    out = VARIANTS[variant].apply(X, W, B, 1e-6)
    ref = F.layer_norm(X, (32768,), W, B, 1e-6)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_constant_input_no_nan(variant):
    """If X is all-ones, var=0 and rstd = 1/sqrt(eps) — output must be finite."""
    shape = (2, 8, 256)
    X = torch.ones(shape, dtype=torch.float32, device="cuda")
    W = torch.ones(256, dtype=torch.float32, device="cuda")
    B = torch.zeros(256, dtype=torch.float32, device="cuda")
    out = VARIANTS[variant].apply(X, W, B, 1e-6)
    assert torch.isfinite(out).all(), f"{variant}: non-finite output for constant input"
    # Output should be B (since (X - mean) = 0)
    torch.testing.assert_close(out, B.expand_as(out), rtol=1e-5, atol=1e-5)


MODULES = {
    "liger": ForgeLayerNormLiger,
    "unsloth": ForgeLayerNormUnsloth,
}


@pytest.mark.parametrize("variant", list(MODULES))
def test_module_state_dict_roundtrip(variant):
    hidden = 1024
    mod1 = MODULES[variant](hidden, dtype=torch.bfloat16, device="cuda")
    # Randomize params so the round-trip isn't trivial
    with torch.no_grad():
        mod1.weight.copy_(torch.randn_like(mod1.weight))
        mod1.bias.copy_(torch.randn_like(mod1.bias))

    state = mod1.state_dict()
    mod2 = MODULES[variant](hidden, dtype=torch.bfloat16, device="cuda")
    mod2.load_state_dict(state)

    X = torch.randn(2, 8, hidden, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(mod1(X), mod2(X), rtol=0.0, atol=0.0)

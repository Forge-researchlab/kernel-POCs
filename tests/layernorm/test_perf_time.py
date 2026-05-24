"""Latency benchmarks (marked `bench`).

Forward / backward / fwd+bwd ms for eager F.layer_norm, torch.compile(F.layer_norm),
Liger, and Unsloth. Prints a table per shape. Soft thresholds — no hard assertions.

Run with:
    pytest tests/test_kernels/layernorm/test_perf_time.py -m bench -s
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from benchmarks.harness import _sync_and_time
from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import SHAPES_DESIGN, make_inputs, clone_leaf


pytestmark = pytest.mark.bench


def _liger_fwd(X, W, B, eps):
    return lambda: ForgeLayerNormLigerFunction.apply(X, W, B, eps)


def _unsloth_fwd(X, W, B, eps):
    return lambda: ForgeLayerNormUnslothFunction.apply(X, W, B, eps)


def _eager_fwd(X, W, B, eps):
    return lambda: F.layer_norm(X, (X.shape[-1],), W, B, eps)


def _compiled_fwd(X, W, B, eps):
    H = X.shape[-1]
    compiled = torch.compile(
        lambda x, w, b: F.layer_norm(x, (H,), w, b, eps),
        mode="reduce-overhead",
    )
    # Warm the compile path
    compiled(X, W, B)
    return lambda: compiled(X, W, B)


def _make_bwd_call(fwd_fn, dY):
    """Return a no-arg callable that runs fwd then backward(dY)."""
    def call():
        out = fwd_fn()
        out.backward(dY.clone(), retain_graph=False)
    return call


@pytest.mark.parametrize("shape", SHAPES_DESIGN)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_latency(shape, dtype, capsys):
    eps = 1e-6
    X, W, B, dY = make_inputs(shape, dtype, requires_grad=False)

    # Forward-only measurements
    fwd_results = {}
    for name, ctor in [
        ("eager",   _eager_fwd),
        ("compile", _compiled_fwd),
        ("liger",   _liger_fwd),
        ("unsloth", _unsloth_fwd),
    ]:
        fn = ctor(X, W, B, eps)
        fwd_results[name] = _sync_and_time(fn, warmup=25, repeats=100)

    # Fwd+bwd measurements (Liger uses grad on W/B; Unsloth on X only)
    def liger_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, True); Bg = clone_leaf(B, True)
        out = ForgeLayerNormLigerFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY)

    def unsloth_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, False); Bg = clone_leaf(B, False)
        out = ForgeLayerNormUnslothFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY.clone())

    def eager_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, True); Bg = clone_leaf(B, True)
        out = F.layer_norm(Xg, (shape[-1],), Wg, Bg, eps)
        out.backward(dY)

    fb_results = {
        "eager":   _sync_and_time(eager_fwd_bwd, warmup=25, repeats=50),
        "liger":   _sync_and_time(liger_fwd_bwd, warmup=25, repeats=50),
        "unsloth": _sync_and_time(unsloth_fwd_bwd, warmup=25, repeats=50),
    }

    with capsys.disabled():
        print(f"\n  --- LATENCY  shape={shape}  dtype={dtype} ---")
        print(f"  {'impl':<10} {'fwd (ms)':>10} {'fwd+bwd (ms)':>14} {'fwd speedup':>14} {'fb speedup':>12}")
        base_f = fwd_results["eager"]
        base_fb = fb_results["eager"]
        for name in ("eager", "compile", "liger", "unsloth"):
            f = fwd_results.get(name, float("nan"))
            fb = fb_results.get(name, float("nan"))
            sp_f = base_f / f if f else float("nan")
            sp_fb = base_fb / fb if fb and fb == fb else float("nan")
            print(f"  {name:<10} {f:>10.4f} {fb:>14.4f} "
                  f"{sp_f:>13.2f}x {sp_fb:>11.2f}x")

"""Peak VRAM benchmarks (marked `bench`).

Compares forward-only and fwd+bwd peak GPU memory across eager / Liger / Unsloth.
Specifically calls out Unsloth's in-place dY -> dX savings.

Run with:
    pytest tests/test_kernels/layernorm/test_perf_memory.py -m bench -s
"""
from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

from benchmarks.harness import _measure_peak_memory
from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import SHAPES_DESIGN, make_inputs, clone_leaf, _elem_size, n_rows


pytestmark = pytest.mark.bench


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape", SHAPES_DESIGN)
def test_peak_memory(shape, capsys):
    dtype = torch.bfloat16
    eps = 1e-6
    X, W, B, dY = make_inputs(shape, dtype, requires_grad=False)

    # --- forward-only ---
    fwd_peaks = {}
    for name, fn in [
        ("eager",   lambda: F.layer_norm(X, (shape[-1],), W, B, eps)),
        ("liger",   lambda: ForgeLayerNormLigerFunction.apply(X, W, B, eps)),
        ("unsloth", lambda: ForgeLayerNormUnslothFunction.apply(X, W, B, eps)),
    ]:
        _cleanup()
        fwd_peaks[name] = _measure_peak_memory(fn)

    # --- fwd+bwd ---
    def eager_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, True); Bg = clone_leaf(B, True)
        out = F.layer_norm(Xg, (shape[-1],), Wg, Bg, eps)
        out.backward(dY)

    def liger_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, True); Bg = clone_leaf(B, True)
        out = ForgeLayerNormLigerFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY)

    def unsloth_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, False); Bg = clone_leaf(B, False)
        out = ForgeLayerNormUnslothFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY.clone())

    fb_peaks = {}
    for name, fn in [("eager", eager_fwd_bwd), ("liger", liger_fwd_bwd),
                     ("unsloth", unsloth_fwd_bwd)]:
        _cleanup()
        fb_peaks[name] = _measure_peak_memory(fn)

    expected_inplace_savings_mb = (n_rows(shape) * shape[-1] * _elem_size(dtype)) / (1024 ** 2)

    with capsys.disabled():
        print(f"\n  --- PEAK VRAM  shape={shape}  dtype={dtype} ---")
        print(f"  {'impl':<10} {'fwd (MB)':>10} {'fwd+bwd (MB)':>14} {'vs eager fwd+bwd':>18}")
        base = fb_peaks["eager"]
        for name in ("eager", "liger", "unsloth"):
            f = fwd_peaks[name]
            fb = fb_peaks[name]
            delta = base - fb
            print(f"  {name:<10} {f:>10.1f} {fb:>14.1f} {delta:>+17.1f}")
        print(f"  expected Unsloth in-place savings vs Liger: {expected_inplace_savings_mb:.1f} MB")
        print(f"  observed Liger-Unsloth fwd+bwd:             {fb_peaks['liger'] - fb_peaks['unsloth']:+.1f} MB")

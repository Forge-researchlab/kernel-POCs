"""Memory-bandwidth utilization vs A100 40GB peak (1555 GB/s).

For each design shape, times fwd and bwd kernels in isolation and divides the
analytical byte count by elapsed time to report GB/s and % of peak.

Run with:
    pytest tests/test_kernels/layernorm/test_bandwidth.py -m bench -s
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.harness import _sync_and_time
from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import (
    A100_PEAK_BW,
    SHAPES_DESIGN,
    clone_leaf,
    compute_bwd_bytes_liger,
    compute_bwd_bytes_unsloth,
    compute_fwd_bytes,
    make_inputs,
    sm_count,
)


pytestmark = pytest.mark.bench


def _bw_row(label, time_ms, total_bytes):
    sec = time_ms / 1000.0
    gbps = total_bytes / sec / 1e9
    util = gbps / (A100_PEAK_BW / 1e9) * 100
    return f"  {label:<22} {time_ms:>9.4f} ms   {gbps:>8.1f} GB/s   {util:>5.1f}%"


@pytest.mark.parametrize("shape", SHAPES_DESIGN)
def test_bandwidth(shape, capsys):
    dtype = torch.bfloat16
    eps = 1e-6
    X, W, B, dY = make_inputs(shape, dtype, requires_grad=False)
    sm = sm_count()

    # --- forward times ---
    fwd_liger = _sync_and_time(
        lambda: ForgeLayerNormLigerFunction.apply(X, W, B, eps),
        warmup=25, repeats=100,
    )
    fwd_unsloth = _sync_and_time(
        lambda: ForgeLayerNormUnslothFunction.apply(X, W, B, eps),
        warmup=25, repeats=100,
    )

    # --- backward times (have to build a closure that runs fwd then bwd) ---
    def liger_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, True); Bg = clone_leaf(B, True)
        out = ForgeLayerNormLigerFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY)

    def unsloth_fwd_bwd():
        Xg = clone_leaf(X, True); Wg = clone_leaf(W, False); Bg = clone_leaf(B, False)
        out = ForgeLayerNormUnslothFunction.apply(Xg, Wg, Bg, eps)
        out.backward(dY.clone())

    fb_liger = _sync_and_time(liger_fwd_bwd, warmup=25, repeats=50)
    fb_unsloth = _sync_and_time(unsloth_fwd_bwd, warmup=25, repeats=50)

    # bwd ≈ fwd+bwd − fwd (rough; both are dominated by their own kernel)
    bwd_liger = max(fb_liger - fwd_liger, 0.001)
    bwd_unsloth = max(fb_unsloth - fwd_unsloth, 0.001)

    bytes_fwd = compute_fwd_bytes(shape, dtype)
    bytes_bwd_l = compute_bwd_bytes_liger(shape, dtype, sm)
    bytes_bwd_u = compute_bwd_bytes_unsloth(shape, dtype)

    with capsys.disabled():
        print(f"\n  --- BANDWIDTH  shape={shape}  dtype={dtype}  A100 peak={A100_PEAK_BW/1e9:.0f} GB/s ---")
        print(_bw_row("liger fwd",       fwd_liger,   bytes_fwd))
        print(_bw_row("unsloth fwd",     fwd_unsloth, bytes_fwd))
        print(_bw_row("liger bwd (est)", bwd_liger,   bytes_bwd_l))
        print(_bw_row("unsloth bwd (est)", bwd_unsloth, bytes_bwd_u))

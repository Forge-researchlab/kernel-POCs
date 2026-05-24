"""Alignment-impact diagnostic (marked `bench`).

When hidden is a power-of-two, BLOCK_SIZE matches it exactly. When it isn't,
BLOCK_SIZE rounds up to the next power-of-two and roughly half of every block
is masked out — wasting SRAM and bandwidth.

This test isn't a pass/fail — it just prints fwd time for matched and mismatched
hidden so the gap is visible.

Run with:
    pytest tests/test_kernels/layernorm/test_alignment_impact.py -m bench -s
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.harness import _sync_and_time
from kernels.layernorm import ForgeLayerNormLigerFunction

from ._helpers import make_inputs


pytestmark = pytest.mark.bench


def test_alignment_impact(capsys):
    dtype = torch.bfloat16
    eps = 1e-6
    n_rows = 4 * 2048
    rows = []
    for hidden in (4096, 4097, 8192, 8193):
        shape = (1, n_rows, hidden)
        X, W, B, _ = make_inputs(shape, dtype, requires_grad=False)
        t = _sync_and_time(
            lambda: ForgeLayerNormLigerFunction.apply(X, W, B, eps),
            warmup=25, repeats=100,
        )
        # Block size = next power of 2 of hidden (capped at 65536).
        bs = 1
        while bs < hidden:
            bs *= 2
        waste = (1 - hidden / bs) * 100
        rows.append((hidden, bs, waste, t))

    with capsys.disabled():
        print(f"\n  --- ALIGNMENT IMPACT  rows={n_rows}  dtype={dtype} ---")
        print(f"  {'hidden':>7} {'block':>7} {'masked':>8}    {'fwd (ms)':>10}")
        for hidden, bs, waste, t in rows:
            print(f"  {hidden:>7} {bs:>7} {waste:>7.1f}%   {t:>10.4f}")

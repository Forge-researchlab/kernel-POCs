"""CUDA kernel launch count via torch.profiler.

PyTorch eager LayerNorm fires ~10+ small kernels (mean/var reductions, affine
multiply, etc.). The Triton variants should fuse to ~1 fwd kernel + ~1 bwd
kernel. Useful for showing fusion wins.

Run with:
    pytest tests/test_kernels/layernorm/test_launch_count.py -m bench -s
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from kernels.layernorm import (
    ForgeLayerNormLigerFunction,
    ForgeLayerNormUnslothFunction,
)

from ._helpers import clone_leaf, make_inputs


pytestmark = pytest.mark.bench


def _count_cuda_kernels(fn) -> int:
    # Warm
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
    torch.cuda.synchronize()
    n = 0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            n += 1
    return n


def test_launch_count(capsys):
    shape = (4, 2048, 4096)
    dtype = torch.bfloat16
    eps = 1e-6
    X, W, B, dY = make_inputs(shape, dtype, requires_grad=False)

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

    counts = {
        "eager":   _count_cuda_kernels(eager_fwd_bwd),
        "liger":   _count_cuda_kernels(liger_fwd_bwd),
        "unsloth": _count_cuda_kernels(unsloth_fwd_bwd),
    }

    with capsys.disabled():
        print(f"\n  --- KERNEL LAUNCHES (fwd+bwd)  shape={shape}  dtype={dtype} ---")
        for name, n in counts.items():
            print(f"  {name:<10} {n} CUDA events")

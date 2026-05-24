"""Minimal vendored stubs of the Liger helpers that `rms_norm.py` imports.

Stripped down from `liger_kernel/ops/utils.py` to remove dependencies on the
liger_kernel package layout. Only includes what rms_norm.py uses on a single
GPU (no NPU, no XPU large-GRF mode, no version-comparator branching beyond
"Triton 3.0+ on CUDA").
"""
from contextlib import nullcontext
from functools import wraps
import operator

import torch
import triton


MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int):
    """Pick (BLOCK_SIZE, num_warps) for a given problem size n.

    Matches Liger's calculate_settings: power-of-two block, warps scale with size.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel: hidden size n={n} requires block "
            f"size {BLOCK_SIZE} which exceeds MAX_FUSED_SIZE={MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def compare_version(package_name: str, op, target_version: str) -> bool:
    """Compare an installed package version against a target via `op` (e.g. operator.ge)."""
    try:
        import importlib.metadata as md
    except ImportError:
        import importlib_metadata as md
    try:
        actual = md.version(package_name)
    except Exception:
        return False
    def _to_tuple(v: str):
        return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())
    return op(_to_tuple(actual), _to_tuple(target_version))


def ensure_contiguous(fn):
    """Decorator that forces all tensor arguments to be contiguous before the call."""
    @wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor) and not a.is_contiguous():
                a = a.contiguous()
            new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                v = v.contiguous()
            new_kwargs[k] = v
        return fn(ctx, *new_args, **new_kwargs)
    return wrapper


def is_npu_available() -> bool:
    """Single-GPU NVIDIA host — never NPU."""
    return False


def get_npu_core_count() -> int:
    """NPU core count; harmless default for the NVIDIA path (unused there)."""
    return 1


def set_large_grf_mode(kernel_args: dict) -> None:
    """XPU-only large-GRF flag; no-op on NVIDIA."""
    return None


# torch dtype -> Triton dtype mapping used by Liger when launching XPU-mode
# block kernels. The single-row path used on NVIDIA doesn't reference this, but
# the symbol must exist so the module import succeeds.
import triton.language as _tl
torch_to_triton_dtype = {
    torch.float32:  _tl.float32,
    torch.float16:  _tl.float16,
    torch.bfloat16: _tl.bfloat16,
}

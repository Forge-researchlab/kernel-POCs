"""Minimal vendored stubs of the Unsloth helpers that rope_embedding.py imports.

Stripped down from `unsloth/kernels/utils.py` and `unsloth/device_type.py` to remove
dependencies on `unsloth_zoo`, bitsandbytes, AMD/Intel device probes, etc.

Only includes what `rope_embedding.py` actually uses:
- DEVICE_COUNT
- calculate_settings
- torch_gpu_device
- torch_device_stream
"""

from contextlib import nullcontext
import triton
import torch


MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int):
    """Pick (BLOCK_SIZE, num_warps) for a given problem size n.

    Verbatim from Unsloth's `calculate_settings` in unsloth/kernels/utils.py.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


DEVICE_COUNT: int = torch.cuda.device_count() if torch.cuda.is_available() else 1


if DEVICE_COUNT > 1:
    torch_gpu_device = torch.cuda.device
else:
    def torch_gpu_device(device):
        return nullcontext()


torch_device_stream = torch.cuda.current_stream

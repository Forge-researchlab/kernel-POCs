from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernels.rmsnorm import rmsnorm
from kernels.rmsnorm import torch_rmsnorm_reference


def _time_ms(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def _run_case(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)

    def torch_forward():
        torch_rmsnorm_reference(x, weight)

    def forge_forward():
        rmsnorm(x, weight)

    def torch_full():
        x.grad = None
        weight.grad = None
        torch_rmsnorm_reference(x, weight).backward(grad)

    def forge_full():
        x.grad = None
        weight.grad = None
        rmsnorm(x, weight).backward(grad)

    return {
        "shape": shape,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_forward_ms": _time_ms(torch_forward),
        "forge_forward_ms": _time_ms(forge_forward),
        "torch_full_ms": _time_ms(torch_full),
        "forge_full_ms": _time_ms(forge_full),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton RMSNorm benchmark")

    cases = [
        ((2, 128, 1024), torch.float16),
        ((2, 128, 4096), torch.float16),
        ((4, 256, 4096), torch.bfloat16),
    ]

    for shape, dtype in cases:
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        result = _run_case(shape, dtype)
        print(result)


if __name__ == "__main__":
    main()

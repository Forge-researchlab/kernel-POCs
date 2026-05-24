"""Isolated Unsloth GEGLU benchmark.

Run this file as a separate Python process. Importing Unsloth can patch global
runtime behavior, so shared Forge/Liger/PyTorch benchmarks must not import this
module or benchmark Unsloth in-process.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
UNSLOTH_SRC = WORKSPACE / "unsloth"
if not UNSLOTH_SRC.exists():
    raise RuntimeError(f"expected Unsloth checkout at {UNSLOTH_SRC}")
sys.path.insert(0, str(UNSLOTH_SRC))

try:
    from unsloth.kernels.geglu import geglu_approx_forward_kernel
    from unsloth.kernels.geglu import geglu_exact_forward_kernel
except Exception as exc:
    raise RuntimeError("Unsloth import failed inside the isolated benchmark process") from exc


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

SUITES = {
    "smoke": [
        # batch, seq, hidden, intermediate
        (1, 128, 1024, 4096),
        (1, 32, 4096, 11008),
        (2, 7, 4096, 11009),
    ],
    "a100": [
        (1, 512, 4096, 11008),
        (2, 2048, 4096, 11008),
        (2, 7, 4096, 11009),
        (1, 128, 3072, 24576),
    ],
}


def _check_cuda() -> None:
    """Inputs: none. Outputs: raises on non-CUDA. Logic: keep this benchmark GPU-only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this isolated Unsloth benchmark has no CPU fallback")


def _supports_dtype(dtype: torch.dtype) -> bool:
    """Inputs: torch dtype. Outputs: CUDA support bool. Logic: gate bf16 on device capability."""
    if dtype is torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


def _time_cuda(fn, warmup: int = 10, rep: int = 30) -> float:
    """Inputs: callable/repeats. Outputs: mean milliseconds. Logic: CUDA events with sync."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    trimmed = times[2:-2] if len(times) > 8 else times
    return sum(trimmed) / len(trimmed)


def _peak_memory(fn) -> int:
    """Inputs: callable. Outputs: peak allocated bytes. Logic: CUDA memory stats around one run."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _make_weights(hidden_size: int, intermediate_size: int, dtype: torch.dtype):
    """Inputs: model dims/dtype. Outputs: scaled weights. Logic: stable MLP initialization."""
    scale_in = 1.0 / math.sqrt(hidden_size)
    scale_mid = 1.0 / math.sqrt(intermediate_size)
    gate_w = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * scale_in
    up_w = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * scale_in
    down_w = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * scale_mid
    return gate_w, up_w, down_w


def _make_inputs(shape, dtype: torch.dtype):
    """Inputs: benchmark shape/dtype. Outputs: x and weights. Logic: shared data for timed modes."""
    batch, seq, hidden_size, intermediate_size = shape
    x = torch.randn(batch, seq, hidden_size, device="cuda", dtype=dtype)
    gate_w, up_w, down_w = _make_weights(hidden_size, intermediate_size, dtype)
    return x, gate_w, up_w, down_w


def _torch_geglu_reference(gate: torch.Tensor, up: torch.Tensor, approximate: str) -> torch.Tensor:
    """Inputs: gate/up tensors. Outputs: reference GEGLU. Logic: PyTorch parity check only."""
    gelu_approx = "tanh" if approximate == "tanh" else "none"
    return F.gelu(gate, approximate=gelu_approx) * up


def _unsloth_geglu(gate: torch.Tensor, up: torch.Tensor, approximate: str) -> torch.Tensor:
    """Inputs: gate/up tensors. Outputs: Unsloth GEGLU. Logic: call standalone Unsloth forward kernel."""
    if approximate == "tanh":
        return geglu_approx_forward_kernel(gate, up)
    return geglu_exact_forward_kernel(gate, up)


def _forward(mode: str, x, gate_w, up_w, down_w, approximate: str):
    """Inputs: mode/tensors. Outputs: activation or MLP output. Logic: benchmark Unsloth gate/up path."""
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    act = _unsloth_geglu(gate, up, approximate)
    if mode == "gateup_forward":
        return act
    if mode == "mlp_forward":
        return F.linear(act, down_w)
    raise ValueError(f"unknown mode: {mode}")


def _check_outputs(shape, dtype: torch.dtype, approximate: str) -> None:
    """Inputs: shape/dtype/approx. Outputs: assertion or none. Logic: compare Unsloth forward to PyTorch."""
    x, gate_w, up_w, down_w = _make_inputs(shape, dtype)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    actual_gateup = _unsloth_geglu(gate, up, approximate)
    expected_gateup = _torch_geglu_reference(gate, up, approximate)
    actual_mlp = F.linear(actual_gateup, down_w)
    expected_mlp = F.linear(expected_gateup, down_w)
    atol = 3e-2 if dtype is not torch.float32 else 2e-5
    rtol = 3e-2 if dtype is not torch.float32 else 2e-5
    torch.testing.assert_close(actual_gateup, expected_gateup, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_mlp, expected_mlp, atol=atol, rtol=rtol)


def run(args) -> None:
    """Inputs: CLI args. Outputs: printed/CSV rows. Logic: isolated Unsloth forward benchmark."""
    _check_cuda()
    torch.manual_seed(args.seed)
    device_name = torch.cuda.get_device_name()
    rows = []

    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        if not _supports_dtype(dtype):
            print(f"warning: skipping {dtype_name}; unsupported on this GPU", file=sys.stderr)
            continue

        for approximate in args.approximate:
            for shape in SUITES[args.suite]:
                if args.check:
                    _check_outputs(shape, dtype, approximate)
                base = _make_inputs(shape, dtype)
                for mode in args.modes:
                    fn = lambda: _forward(mode, *base, approximate)
                    ms = _time_cuda(fn, warmup=args.warmup, rep=args.rep)
                    peak = _peak_memory(fn)
                    row = {
                        "device": device_name,
                        "suite": args.suite,
                        "provider": "unsloth",
                        "mode": mode,
                        "batch": shape[0],
                        "seq": shape[1],
                        "hidden": shape[2],
                        "intermediate": shape[3],
                        "dtype": dtype_name,
                        "approximate": approximate,
                        "time_ms": f"{ms:.6f}",
                        "peak_memory_mib": f"{peak / 2**20:.3f}",
                    }
                    rows.append(row)
                    print(row)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


def parse_args():
    """Inputs: CLI. Outputs: parsed args. Logic: expose isolated Unsloth benchmark controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--dtype", choices=sorted(DTYPES), nargs="+", default=["bf16"])
    parser.add_argument("--approximate", choices=["tanh", "none"], nargs="+", default=["tanh"])
    parser.add_argument("--modes", choices=["gateup_forward", "mlp_forward"], nargs="+", default=["gateup_forward", "mlp_forward"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

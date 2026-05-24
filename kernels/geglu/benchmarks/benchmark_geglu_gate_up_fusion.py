import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

LIGER_SRC = WORKSPACE / "Liger-Kernel" / "src"
if LIGER_SRC.exists():
    sys.path.insert(0, str(LIGER_SRC))

UNSLOTH_SRC = WORKSPACE / "unsloth"
if UNSLOTH_SRC.exists():
    sys.path.insert(0, str(UNSLOTH_SRC))

from kernels.geglu import geglu
from kernels.geglu import geglu_packed
from kernels.geglu import torch_geglu_packed_reference
from kernels.geglu import torch_geglu_reference


try:
    from liger_kernel.ops import LigerGELUMulFunction
except Exception as exc:
    LigerGELUMulFunction = None
    LIGER_IMPORT_ERROR = repr(exc)
else:
    LIGER_IMPORT_ERROR = None


try:
    from unsloth.kernels.geglu import geglu_approx_forward_kernel
    from unsloth.kernels.geglu import geglu_exact_forward_kernel
except Exception as exc:
    geglu_approx_forward_kernel = None
    geglu_exact_forward_kernel = None
    UNSLOTH_IMPORT_ERROR = repr(exc)
else:
    UNSLOTH_IMPORT_ERROR = None


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

PROVIDERS = {
    "separate_torch",
    "separate_forge",
    "packed_torch",
    "packed_forge",
    "liger",
    "unsloth",
}


def _check_cuda() -> None:
    """Inputs: none. Outputs: raises on non-CUDA. Logic: keep this experiment GPU-only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this gate+up fusion experiment intentionally has no CPU fallback")


def _supports_dtype(dtype: torch.dtype) -> bool:
    """Inputs: torch dtype. Outputs: CUDA support bool. Logic: gate bf16 on device capability."""
    if dtype is torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


def _time_cuda(fn, setup=None, warmup=10, rep=30) -> float:
    """Inputs: callable and optional setup. Outputs: mean milliseconds. Logic: CUDA events with sync."""
    for _ in range(warmup):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
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


def _peak_memory(fn, setup=None) -> int:
    """Inputs: callable and optional setup. Outputs: peak allocated bytes. Logic: CUDA memory stats."""
    torch.cuda.empty_cache()
    if setup is not None:
        setup()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _make_weights(hidden_size: int, intermediate_size: int, dtype: torch.dtype):
    """Inputs: model dims/dtype. Outputs: scaled MLP weights. Logic: deterministic stable initialization."""
    scale_in = 1.0 / math.sqrt(hidden_size)
    scale_mid = 1.0 / math.sqrt(intermediate_size)
    gate_w = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * scale_in
    up_w = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * scale_in
    packed_w = torch.cat([gate_w, up_w], dim=0).contiguous()
    down_w = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * scale_mid
    return gate_w, up_w, packed_w, down_w


def _make_inputs(shape, dtype: torch.dtype):
    """Inputs: benchmark shape/dtype. Outputs: x, weights, grads. Logic: shared data for fair variants."""
    batch, seq, hidden_size, intermediate_size = shape
    x = torch.randn(batch, seq, hidden_size, device="cuda", dtype=dtype)
    gate_w, up_w, packed_w, down_w = _make_weights(hidden_size, intermediate_size, dtype)
    gateup_grad = torch.randn(batch, seq, intermediate_size, device="cuda", dtype=dtype)
    mlp_grad = torch.randn(batch, seq, hidden_size, device="cuda", dtype=dtype)
    return x, gate_w, up_w, packed_w, down_w, gateup_grad, mlp_grad


def _activation(provider: str, x, gate_w, up_w, packed_w, approximate: str):
    """Inputs: provider and tensors. Outputs: GEGLU activation. Logic: compare separate vs packed gate/up."""
    if provider == "separate_torch":
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        return torch_geglu_reference(gate, up, approximate)
    if provider == "separate_forge":
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        return geglu(gate, up, approximate, preserve_inputs=False)
    if provider == "packed_torch":
        gate_up = F.linear(x, packed_w)
        return torch_geglu_packed_reference(gate_up, approximate)
    if provider == "packed_forge":
        gate_up = F.linear(x, packed_w)
        return geglu_packed(gate_up, approximate, preserve_inputs=False)
    if provider == "liger":
        if approximate != "tanh":
            raise RuntimeError("Liger GEGLU only supports approximate='tanh'")
        if LigerGELUMulFunction is None:
            raise RuntimeError(f"Liger is not importable: {LIGER_IMPORT_ERROR}")
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        return LigerGELUMulFunction.apply(gate, up)
    if provider == "unsloth":
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        if approximate == "tanh":
            if geglu_approx_forward_kernel is None:
                raise RuntimeError(f"Unsloth is not importable: {UNSLOTH_IMPORT_ERROR}")
            return geglu_approx_forward_kernel(gate, up)
        if geglu_exact_forward_kernel is None:
            raise RuntimeError(f"Unsloth is not importable: {UNSLOTH_IMPORT_ERROR}")
        return geglu_exact_forward_kernel(gate, up)
    raise ValueError(f"unknown provider: {provider}")


def _forward(provider: str, mode: str, x, gate_w, up_w, packed_w, down_w, approximate: str):
    """Inputs: provider/mode/tensors. Outputs: activation or down-projected MLP. Logic: isolate fusion value."""
    act = _activation(provider, x, gate_w, up_w, packed_w, approximate)
    if mode == "gateup_forward":
        return act
    if mode == "mlp_forward":
        return F.linear(act, down_w)
    raise ValueError(f"forward provider received invalid mode: {mode}")


def _clone_for_grad(*tensors):
    """Inputs: tensors. Outputs: detached grad-enabled clones. Logic: keep each timed run independent."""
    return [t.detach().clone().requires_grad_(True) for t in tensors]


def _bench_forward(provider: str, mode: str, base, approximate: str, warmup: int, rep: int):
    """Inputs: provider/mode/base tensors. Outputs: ms and peak bytes. Logic: forward-only timing."""
    x, gate_w, up_w, packed_w, down_w, _, _ = base
    fn = lambda: _forward(provider, mode, x, gate_w, up_w, packed_w, down_w, approximate)
    return _time_cuda(fn, warmup=warmup, rep=rep), _peak_memory(fn)


def _bench_full(provider: str, mode: str, base, approximate: str, warmup: int, rep: int):
    """Inputs: provider/mode/base tensors. Outputs: ms and peak bytes. Logic: full autograd timing."""
    if provider == "unsloth":
        return None, None

    base_x, base_gate_w, base_up_w, base_packed_w, base_down_w, gateup_grad, mlp_grad = base
    state = {}

    def setup():
        if provider.startswith("packed"):
            x, packed_w = _clone_for_grad(base_x, base_packed_w)
            state.clear()
            state.update({"x": x, "packed_w": packed_w, "down_w": base_down_w.detach().clone().requires_grad_(True)})
        else:
            x, gate_w, up_w = _clone_for_grad(base_x, base_gate_w, base_up_w)
            state.clear()
            state.update({"x": x, "gate_w": gate_w, "up_w": up_w, "down_w": base_down_w.detach().clone().requires_grad_(True)})

    def fn():
        if provider.startswith("packed"):
            out = _forward(
                provider,
                mode.replace("_full", "_forward"),
                state["x"],
                base_gate_w,
                base_up_w,
                state["packed_w"],
                state["down_w"],
                approximate,
            )
        else:
            out = _forward(
                provider,
                mode.replace("_full", "_forward"),
                state["x"],
                state["gate_w"],
                state["up_w"],
                base_packed_w,
                state["down_w"],
                approximate,
            )
        out.backward(gateup_grad if mode == "gateup_full" else mlp_grad)

    return _time_cuda(fn, setup=setup, warmup=warmup, rep=rep), _peak_memory(fn, setup=setup)


def _check_outputs(shape, dtype: torch.dtype, approximate: str, providers):
    """Inputs: shape/dtype/providers. Outputs: none or assertion. Logic: verify packed/separate forward parity."""
    base = _make_inputs(shape, dtype)
    x, gate_w, up_w, packed_w, down_w, _, _ = base
    ref = _forward("separate_torch", "mlp_forward", x, gate_w, up_w, packed_w, down_w, approximate)
    for provider in providers:
        if provider == "liger" and approximate != "tanh":
            continue
        out = _forward(provider, "mlp_forward", x, gate_w, up_w, packed_w, down_w, approximate)
        torch.testing.assert_close(out, ref, atol=3e-2 if dtype is not torch.float32 else 2e-5, rtol=3e-2 if dtype is not torch.float32 else 2e-5)


def run(args):
    """Inputs: CLI args. Outputs: printed/CSV rows. Logic: benchmark packed gate+up fusion on CUDA."""
    _check_cuda()
    device_name = torch.cuda.get_device_name()
    providers = args.providers
    if "liger" in providers and LigerGELUMulFunction is None:
        print(f"warning: skipping liger: {LIGER_IMPORT_ERROR}", file=sys.stderr)
        providers = [p for p in providers if p != "liger"]
    if "unsloth" in providers and (geglu_approx_forward_kernel is None or geglu_exact_forward_kernel is None):
        print(f"warning: skipping unsloth: {UNSLOTH_IMPORT_ERROR}", file=sys.stderr)
        providers = [p for p in providers if p != "unsloth"]

    rows = []
    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        if not _supports_dtype(dtype):
            print(f"warning: skipping {dtype_name}; unsupported on this GPU", file=sys.stderr)
            continue

        for approximate in args.approximate:
            active_providers = [p for p in providers if p != "liger" or approximate == "tanh"]
            for shape in SUITES[args.suite]:
                if args.check:
                    _check_outputs(shape, dtype, approximate, active_providers)
                base = _make_inputs(shape, dtype)
                for provider in active_providers:
                    for mode in args.modes:
                        if mode.endswith("_forward"):
                            ms, peak = _bench_forward(provider, mode, base, approximate, args.warmup, args.rep)
                        elif mode.endswith("_full"):
                            ms, peak = _bench_full(provider, mode, base, approximate, args.warmup, args.rep)
                            if ms is None:
                                if provider == "unsloth":
                                    print("warning: skipping unsloth full mode; standalone Unsloth GEGLU has no autograd wrapper", file=sys.stderr)
                                continue
                        else:
                            raise ValueError(f"unknown mode: {mode}")

                        row = {
                            "device": device_name,
                            "suite": args.suite,
                            "provider": provider,
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
    """Inputs: CLI. Outputs: parsed args. Logic: expose fast gate+up experiment controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--dtype", choices=sorted(DTYPES), nargs="+", default=["bf16"])
    parser.add_argument("--approximate", choices=["tanh", "none"], nargs="+", default=["tanh"])
    parser.add_argument(
        "--providers",
        choices=sorted(PROVIDERS),
        nargs="+",
        default=["separate_torch", "separate_forge", "packed_torch", "packed_forge", "liger", "unsloth"],
    )
    parser.add_argument(
        "--modes",
        choices=["gateup_forward", "gateup_full", "mlp_forward", "mlp_full"],
        nargs="+",
        default=["gateup_forward", "mlp_forward"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

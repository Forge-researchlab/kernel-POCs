import argparse
import csv
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

LIGER_SRC = WORKSPACE / "Liger-Kernel" / "src"
if LIGER_SRC.exists():
    sys.path.insert(0, str(LIGER_SRC))

from kernels.swiglu import swiglu
from kernels.swiglu import swiglu_backward
from kernels.swiglu import swiglu_forward
from kernels.swiglu import swiglu_packed
from kernels.swiglu import swiglu_packed_backward
from kernels.swiglu import swiglu_packed_forward
from kernels.swiglu import torch_swiglu_packed_reference
from kernels.swiglu import torch_swiglu_reference


def _patch_liger_dtensor_compat() -> None:
    try:
        import torch.distributed.tensor as dist_tensor
    except Exception:
        return

    if hasattr(dist_tensor, "DTensor"):
        return

    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        class DTensor:  # type: ignore[no-redef]
            pass

    dist_tensor.DTensor = DTensor


_patch_liger_dtensor_compat()


try:
    from liger_kernel.ops import LigerSiLUMulFunction
except Exception as exc:
    LigerSiLUMulFunction = None
    LIGER_IMPORT_ERROR = repr(exc)
else:
    LIGER_IMPORT_ERROR = None


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

SUITES = {
    "smoke": [
        (2, 128, 1024),
        (1, 256, 11008),
        (2, 7, 11009),
    ],
    "a100": [
        (1, 512, 11008),
        (2, 2048, 11008),
        (4, 2048, 11008),
        (2, 7, 11009),
        (1, 128, 32768),
        (1, 32, 65536),
    ],
}

MULTIPLIERS = {
    "default": [(1.0, 1.0)],
    "scaled": [(0.7, 1.3)],
    "both": [(1.0, 1.0), (0.7, 1.3)],
}


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")


def _supports_dtype(dtype: torch.dtype) -> bool:
    if dtype is torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


def _time_cuda(fn, setup=None, warmup=20, rep=50) -> float:
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
    torch.cuda.empty_cache()
    if setup is not None:
        setup()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _make_inputs(shape, dtype):
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape, device="cuda", dtype=dtype)
    gate_up = torch.cat([gate, up], dim=-1).contiguous()
    return gate, up, gate_up, grad


def _separate_provider(provider, gate, up, gate_multiplier, down_multiplier):
    if provider == "torch":
        return torch_swiglu_reference(gate, up, gate_multiplier, down_multiplier)
    if provider == "forge_fast":
        return swiglu(gate, up, gate_multiplier, down_multiplier, preserve_inputs=False)
    if provider == "forge_safe":
        return swiglu(gate, up, gate_multiplier, down_multiplier, preserve_inputs=True)
    if provider == "liger":
        if LigerSiLUMulFunction is None:
            raise RuntimeError("Liger is not importable")
        return LigerSiLUMulFunction.apply(gate, up, gate_multiplier, down_multiplier)
    raise ValueError(f"unknown separate provider: {provider}")


def _packed_provider(provider, gate_up, gate_multiplier, down_multiplier):
    if provider == "torch_packed":
        return torch_swiglu_packed_reference(gate_up, gate_multiplier, down_multiplier)
    if provider == "forge_packed_fast":
        return swiglu_packed(gate_up, gate_multiplier, down_multiplier, preserve_inputs=False)
    if provider == "forge_packed_safe":
        return swiglu_packed(gate_up, gate_multiplier, down_multiplier, preserve_inputs=True)
    raise ValueError(f"unknown packed provider: {provider}")


def _bench_forward(provider, layout, shape, dtype, gate_multiplier, down_multiplier, warmup, rep):
    gate, up, gate_up, _ = _make_inputs(shape, dtype)
    if layout == "separate":
        fn = lambda: _separate_provider(provider, gate, up, gate_multiplier, down_multiplier)
    else:
        fn = lambda: _packed_provider(provider, gate_up, gate_multiplier, down_multiplier)
    return _time_cuda(fn, warmup=warmup, rep=rep), _peak_memory(fn)


def _bench_full(provider, layout, shape, dtype, gate_multiplier, down_multiplier, warmup, rep):
    base_gate, base_up, base_gate_up, grad = _make_inputs(shape, dtype)
    state = {}

    def setup():
        if layout == "separate":
            state["gate"] = base_gate.detach().clone().requires_grad_(True)
            state["up"] = base_up.detach().clone().requires_grad_(True)
        else:
            state["gate_up"] = base_gate_up.detach().clone().requires_grad_(True)

    def fn():
        if layout == "separate":
            out = _separate_provider(provider, state["gate"], state["up"], gate_multiplier, down_multiplier)
        else:
            out = _packed_provider(provider, state["gate_up"], gate_multiplier, down_multiplier)
        out.backward(grad)

    return _time_cuda(fn, setup=setup, warmup=warmup, rep=rep), _peak_memory(fn, setup=setup)


def _bench_backward(provider, layout, shape, dtype, gate_multiplier, down_multiplier, warmup, rep):
    if provider not in {"forge_fast", "forge_safe", "forge_packed_fast", "forge_packed_safe"}:
        return None, None

    base_gate, base_up, base_gate_up, grad = _make_inputs(shape, dtype)
    state = {}

    def setup():
        if layout == "separate":
            gate_input = base_gate.detach().clone()
            up_input = base_up.detach().clone()
            _, gate_saved, up_saved = swiglu_forward(gate_input, up_input, gate_multiplier, down_multiplier)
            state["gate"] = gate_saved
            state["up"] = up_saved
        else:
            gate_up_input = base_gate_up.detach().clone()
            _, gate_up_saved = swiglu_packed_forward(gate_up_input, gate_multiplier, down_multiplier)
            state["gate_up"] = gate_up_saved

    def fn():
        if layout == "separate":
            swiglu_backward(
                grad,
                state["gate"],
                state["up"],
                gate_multiplier,
                down_multiplier,
                preserve_inputs=provider == "forge_safe",
            )
        else:
            swiglu_packed_backward(
                grad,
                state["gate_up"],
                base_gate_up.shape,
                gate_multiplier,
                down_multiplier,
                preserve_inputs=provider == "forge_packed_safe",
            )

    return _time_cuda(fn, setup=setup, warmup=warmup, rep=rep), _peak_memory(fn, setup=setup)


def _provider_layout(provider):
    if provider in {"torch", "forge_fast", "forge_safe", "liger"}:
        return "separate"
    if provider in {"torch_packed", "forge_packed_fast", "forge_packed_safe"}:
        return "packed"
    raise ValueError(f"unknown provider: {provider}")


def run(args):
    _check_cuda()
    device_name = torch.cuda.get_device_name()
    rows = []
    providers = args.providers
    if "liger" in providers and LigerSiLUMulFunction is None:
        print(
            f"warning: skipping liger provider because liger_kernel is not importable: {LIGER_IMPORT_ERROR}",
            file=sys.stderr,
        )
        providers = [p for p in providers if p != "liger"]

    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        if not _supports_dtype(dtype):
            print(f"warning: skipping {dtype_name}; unsupported on this GPU", file=sys.stderr)
            continue

        for shape in SUITES[args.suite]:
            for gate_multiplier, down_multiplier in MULTIPLIERS[args.multipliers]:
                for provider in providers:
                    layout = _provider_layout(provider)
                    for mode in args.modes:
                        if mode == "forward":
                            ms, peak = _bench_forward(
                                provider,
                                layout,
                                shape,
                                dtype,
                                gate_multiplier,
                                down_multiplier,
                                args.warmup,
                                args.rep,
                            )
                        elif mode == "full":
                            ms, peak = _bench_full(
                                provider,
                                layout,
                                shape,
                                dtype,
                                gate_multiplier,
                                down_multiplier,
                                args.warmup,
                                args.rep,
                            )
                        elif mode == "backward":
                            ms, peak = _bench_backward(
                                provider,
                                layout,
                                shape,
                                dtype,
                                gate_multiplier,
                                down_multiplier,
                                args.warmup,
                                args.rep,
                            )
                            if ms is None:
                                continue
                        else:
                            raise ValueError(f"unknown mode: {mode}")

                        row = {
                            "device": device_name,
                            "suite": args.suite,
                            "provider": provider,
                            "layout": layout,
                            "mode": mode,
                            "batch": shape[0],
                            "seq": shape[1],
                            "hidden": shape[2],
                            "dtype": dtype_name,
                            "gate_multiplier": gate_multiplier,
                            "down_multiplier": down_multiplier,
                            "time_ms": f"{ms:.6f}",
                            "peak_memory_mib": f"{peak / 2**20:.3f}",
                            "warmup": args.warmup,
                            "rep": args.rep,
                        }
                        rows.append(row)
                        print(row)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {len(rows)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--dtype", nargs="+", choices=sorted(DTYPES), default=["bf16"])
    parser.add_argument("--multipliers", choices=sorted(MULTIPLIERS), default="both")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["torch", "forge_fast", "forge_safe", "torch_packed", "forge_packed_fast", "forge_packed_safe", "liger"],
    )
    parser.add_argument("--modes", nargs="+", choices=["forward", "backward", "full"], default=["forward", "backward", "full"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--save", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

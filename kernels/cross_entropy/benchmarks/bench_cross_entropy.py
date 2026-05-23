"""
Cross entropy benchmark harness.

Compares:
  - torch: torch.nn.functional.cross_entropy
  - forge: kernels/cross_entropy/experiments/v2
  - liger: liger_kernel.transformers.cross_entropy.LigerCrossEntropyLoss

Timing uses CUDA events and reports median latency. Memory uses CUDA peak
allocated bytes during the measured operation, excluding input tensor creation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v2 import forge_cross_entropy  # noqa: E402


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Provider:
    name: str
    loss_fn: LossFn


@dataclass(frozen=True)
class Measurement:
    bt: int
    vocab: int
    provider: str
    mode: str
    latency_ms: float
    memory_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cross entropy providers.")
    parser.add_argument("--bt", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--vocab", type=int, nargs="+", default=[128256])
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--providers", nargs="+", default=["torch", "forge", "liger"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["forward", "backward", "full", "no-grad-forward"],
        choices=["forward", "backward", "full", "no-grad-forward"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--reduction", choices=["mean", "sum", "none"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument(
        "--liger-path",
        type=str,
        default=os.environ.get("LIGER_KERNEL_PATH"),
        help="Optional path to a Liger-Kernel checkout if liger_kernel is not installed.",
    )
    return parser.parse_args()


def torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def maybe_add_liger_path(liger_path: str | None) -> None:
    if not liger_path:
        return
    path = Path(liger_path).expanduser().resolve()
    if path.exists():
        sys.path.insert(0, str(path))


def build_provider(name: str, args: argparse.Namespace) -> Provider:
    if name == "torch":
        return Provider(
            name="torch",
            loss_fn=lambda logits, target: F.cross_entropy(
                logits,
                target,
                ignore_index=args.ignore_index,
                reduction=args.reduction,
                label_smoothing=args.label_smoothing,
            ),
        )

    if name == "forge":
        return Provider(
            name="forge",
            loss_fn=lambda logits, target: forge_cross_entropy(
                logits,
                target,
                ignore_index=args.ignore_index,
                reduction=args.reduction,
                label_smoothing=args.label_smoothing,
            ),
        )

    if name == "liger":
        maybe_add_liger_path(args.liger_path)
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        loss_module = LigerCrossEntropyLoss(
            ignore_index=args.ignore_index,
            label_smoothing=args.label_smoothing,
            reduction=args.reduction,
        )
        return Provider(name="liger", loss_fn=loss_module)

    raise ValueError(f"Unknown provider: {name}")


def make_inputs(bt: int, vocab: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(bt, vocab, device=device, dtype=dtype)
    target = torch.randint(0, vocab, (bt,), device=device)
    return logits, target


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss if loss.ndim == 0 else loss.sum()


def build_trial(provider: Provider, mode: str, base_logits: torch.Tensor, target: torch.Tensor) -> Callable[[], None]:
    if mode == "no-grad-forward":
        logits = base_logits.detach().clone()

        def run_no_grad_forward() -> None:
            with torch.no_grad():
                provider.loss_fn(logits, target)

        return run_no_grad_forward

    logits = base_logits.detach().clone().requires_grad_(True)

    if mode == "forward":
        return lambda: provider.loss_fn(logits, target)

    if mode == "backward":
        loss = provider.loss_fn(logits, target)
        loss = scalar_loss(loss)
        torch.cuda.synchronize()
        return loss.backward

    if mode == "full":

        def run_full() -> None:
            loss = provider.loss_fn(logits, target)
            scalar_loss(loss).backward()

        return run_full

    raise ValueError(f"Unknown mode: {mode}")


def warm_kernel(provider: Provider, mode: str, base_logits: torch.Tensor, target: torch.Tensor, warmup: int) -> None:
    for _ in range(warmup):
        build_trial(provider, mode, base_logits, target)()
    torch.cuda.synchronize()


def bench_latency(
    provider: Provider,
    mode: str,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    warmup: int,
    rep: int,
) -> float:
    warm_kernel(provider, mode, base_logits, target, warmup)

    times = []
    for _ in range(rep):
        trial = build_trial(provider, mode, base_logits, target)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        trial()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return median(times)


def bench_memory(
    provider: Provider,
    mode: str,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    warmup: int,
) -> float:
    warm_kernel(provider, mode, base_logits, target, warmup)
    torch.cuda.empty_cache()
    trial = build_trial(provider, mode, base_logits, target)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    trial()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - before) / 1024 / 1024


def write_csv(rows: list[dict[str, str]], save_path: str) -> Path:
    path = Path(save_path)
    if path.suffix.lower() != ".csv":
        path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = path / f"cross_entropy_{timestamp}.csv"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def format_mb(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.3f}"


def baseline_vs_forge(baseline: float | None, forge: float | None) -> float | None:
    if baseline is None or forge is None or baseline <= 0 or forge <= 0:
        return None
    return baseline / forge


def format_comparison(ratio: float | None) -> str:
    if ratio is None:
        return "-"
    return f"{ratio:.2f}x"


def render_table(headers: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> str:
    right_align = right_align or set()
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def border(left: str, fill: str, join: str, right: str) -> str:
        return left + join.join(fill * (width + 2) for width in widths) + right

    def format_row(row: list[str]) -> str:
        cells = []
        for header, width, cell in zip(headers, widths, row):
            text = cell.rjust(width) if header in right_align else cell.ljust(width)
            cells.append(f" {text} ")
        return "│" + "│".join(cells) + "│"

    lines = [
        border("┌", "─", "┬", "┐"),
        format_row(headers),
        border("├", "─", "┼", "┤"),
    ]
    lines.extend(format_row(row) for row in rows)
    lines.append(border("└", "─", "┴", "┘"))
    return "\n".join(lines)


def index_measurements(measurements: list[Measurement]) -> dict[tuple[int, int, str, str], Measurement]:
    return {
        (measurement.vocab, measurement.bt, measurement.mode, measurement.provider): measurement
        for measurement in measurements
    }


def get_value(
    indexed: dict[tuple[int, int, str, str], Measurement],
    vocab: int,
    bt: int,
    mode: str,
    provider: str,
    attr: str,
) -> float | None:
    measurement = indexed.get((vocab, bt, mode, provider))
    return getattr(measurement, attr) if measurement else None


def print_summary_tables(
    measurements: list[Measurement],
    vocab_values: list[int],
    bt_values: list[int],
    modes: list[str],
) -> None:
    indexed = index_measurements(measurements)
    include_vocab = len(vocab_values) > 1

    for vocab in vocab_values:
        vocab_suffix = f", V={vocab}" if include_vocab else ""
        for bt in bt_values:
            section = f"BT={bt}{vocab_suffix}"

            latency_headers = ["Mode", "Forge", "Torch", "Liger", "Torch/Forge", "Liger/Forge"]
            latency_rows = []
            for mode in modes:
                forge = get_value(indexed, vocab, bt, mode, "forge", "latency_ms")
                torch_value = get_value(indexed, vocab, bt, mode, "torch", "latency_ms")
                liger = get_value(indexed, vocab, bt, mode, "liger", "latency_ms")
                latency_rows.append(
                    [
                        mode,
                        format_ms(forge),
                        format_ms(torch_value),
                        format_ms(liger),
                        format_comparison(baseline_vs_forge(torch_value, forge)),
                        format_comparison(baseline_vs_forge(liger, forge)),
                    ]
                )

            print(f"\nLatency, median CUDA time in ms ({section}):")
            print(
                render_table(
                    latency_headers,
                    latency_rows,
                    right_align={"Forge", "Torch", "Liger", "Torch/Forge", "Liger/Forge"},
                )
            )

            memory_modes = ["full"] if "full" in modes else modes
            memory_headers = ["Mode", "Forge MB", "Torch MB", "Liger MB", "Torch/Forge", "Liger/Forge"]
            memory_rows = []
            for mode in memory_modes:
                forge = get_value(indexed, vocab, bt, mode, "forge", "memory_mb")
                torch_value = get_value(indexed, vocab, bt, mode, "torch", "memory_mb")
                liger = get_value(indexed, vocab, bt, mode, "liger", "memory_mb")
                memory_rows.append(
                    [
                        mode,
                        format_mb(forge),
                        format_mb(torch_value),
                        format_mb(liger),
                        format_comparison(baseline_vs_forge(torch_value, forge)),
                        format_comparison(baseline_vs_forge(liger, forge)),
                    ]
                )

            print(f"\nPeak memory in MB ({section}):")
            print(
                render_table(
                    memory_headers,
                    memory_rows,
                    right_align={"Forge MB", "Torch MB", "Liger MB", "Torch/Forge", "Liger/Forge"},
                )
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch_dtype(args.dtype)

    providers = []
    for name in args.providers:
        try:
            providers.append(build_provider(name, args))
        except Exception as exc:
            print(f"Skipping provider {name}: {exc}")

    if not providers:
        raise RuntimeError("No benchmark providers are available.")

    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print(f"dtype={args.dtype} reduction={args.reduction} label_smoothing={args.label_smoothing}")

    rows: list[dict[str, str]] = []
    all_measurements: list[Measurement] = []
    for vocab in args.vocab:
        for bt in args.bt:
            base_logits, target = make_inputs(bt, vocab, dtype, device)
            print(f"Benchmarking BT={bt} V={vocab} ...")
            for mode in args.modes:
                for provider in providers:
                    latency_ms = bench_latency(provider, mode, base_logits, target, args.warmup, args.rep)
                    memory_mb = bench_memory(provider, mode, base_logits, target, args.warmup)
                    all_measurements.append(Measurement(bt, vocab, provider.name, mode, latency_ms, memory_mb))

    print_summary_tables(all_measurements, args.vocab, args.bt, args.modes)

    indexed = index_measurements(all_measurements)
    for measurement in all_measurements:
        forge_latency = get_value(indexed, measurement.vocab, measurement.bt, measurement.mode, "forge", "latency_ms")
        forge_memory = get_value(indexed, measurement.vocab, measurement.bt, measurement.mode, "forge", "memory_mb")
        rows.append(
            {
                "provider": measurement.provider,
                "mode": measurement.mode,
                "bt": str(measurement.bt),
                "vocab": str(measurement.vocab),
                "dtype": args.dtype,
                "latency_ms": f"{measurement.latency_ms:.4f}",
                "memory_mb": f"{measurement.memory_mb:.2f}",
                "latency_vs_forge": (
                    f"{baseline_vs_forge(measurement.latency_ms, forge_latency):.3f}"
                    if baseline_vs_forge(measurement.latency_ms, forge_latency)
                    else "N/A"
                ),
                "memory_vs_forge": (
                    f"{baseline_vs_forge(measurement.memory_mb, forge_memory):.3f}"
                    if baseline_vs_forge(measurement.memory_mb, forge_memory)
                    else "N/A"
                ),
            }
        )

    if args.save and rows:
        path = write_csv(rows, args.save)
        print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()

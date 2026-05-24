"""
Standalone v6 LoRA MLP latency benchmark.

Runs the v6 kernel (sync and streams variants) N times (default 10),
each run using triton.testing.do_bench with warmup+rep internally. Reports
per-run latency, mean, std, and a "clean mean" that drops outlier runs
(>2x the median) caused by GPU throttling or contention.

This file does NOT import anything from the Unsloth baseline or any other
kernel version (v1, v3, v5). All weight tensors are created directly.

Usage:
    python benchmarks/bench_v6.py                          # defaults
    python benchmarks/bench_v6.py --runs 20 --batch 4 --seq 2048
    python benchmarks/bench_v6.py --rank 32 --save results/
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.v6.lora_mlp_kernel_v6 import (
    lora_mlp_v6,
    stack_lora_a,
    stack_gate_lora_a,
    stack_down_lora_a,
)

DEVICE = "cuda"
WARMUP = 10
REP = 50


def make_v6_params(hidden, intermediate, rank, dtype=torch.bfloat16, device="cuda"):
    """Create weight tensors directly — no unsloth dependency."""
    H, I, r = hidden, intermediate, rank

    W_gate = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_up = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(H, I, dtype=dtype, device=device) * 0.02

    A_gate = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    B_gate = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    A_up = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    B_up = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    A_down = torch.randn(r, I, dtype=dtype, device=device) * 0.02
    B_down = torch.randn(H, r, dtype=dtype, device=device) * 0.02

    s_gate = 1.0
    s_up = 1.0
    s_down = 1.0

    A_stack = stack_lora_a(A_gate, A_up)
    W_gate_stack = stack_gate_lora_a(W_gate, A_gate, A_up)
    W_down_stack = stack_down_lora_a(W_down, A_down)

    return {
        "W_gate": W_gate, "A_gate": A_gate, "B_gate": B_gate, "s_gate": s_gate,
        "W_up": W_up, "A_up": A_up, "B_up": B_up, "s_up": s_up,
        "W_down": W_down, "A_down": A_down, "B_down": B_down, "s_down": s_down,
        "A_stack": A_stack,
        "W_gate_stack": W_gate_stack,
        "W_down_stack": W_down_stack,
    }


def bench_single_run(X, params, enable_streams):
    """One do_bench measurement of v6 forward."""
    def fn():
        return lora_mlp_v6(X, **params, enable_streams=enable_streams)
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def compute_stats(times):
    """Compute mean, std, clean_mean, clean_std from a list of times."""
    n = len(times)
    mean = sum(times) / n
    variance = sum((t - mean) ** 2 for t in times) / n
    std = math.sqrt(variance)

    median = sorted(times)[n // 2]
    clean = [t for t in times if t < 2.0 * median]
    clean_mean = sum(clean) / len(clean) if clean else mean
    clean_var = sum((t - clean_mean) ** 2 for t in clean) / len(clean) if clean else variance
    clean_std = math.sqrt(clean_var)
    n_dropped = n - len(clean)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "clean_mean": round(clean_mean, 4),
        "clean_std": round(clean_std, 4),
        "n_clean": len(clean),
        "n_dropped": n_dropped,
    }


def bench_v6(batch, seq_len, hidden, intermediate, rank, dtype, n_runs):
    """Run v6_sync and v6_streams benchmarks n_runs times each."""
    params = make_v6_params(hidden, intermediate, rank, dtype=dtype, device=DEVICE)
    X = torch.randn(batch, seq_len, hidden, dtype=dtype, device=DEVICE)
    M = batch * seq_len

    print(f"Config: batch={batch} seq={seq_len} hidden={hidden} "
          f"intermediate={intermediate} rank={rank} M={M} dtype={dtype}")
    print(f"Running {n_runs} benchmark runs ({WARMUP} warmup + {REP} reps each)\n")

    # --- v6_sync ---
    print("=== v6_sync (enable_streams=False) ===")
    sync_times = []
    for i in range(n_runs):
        ms = bench_single_run(X, params, enable_streams=False)
        sync_times.append(round(ms, 4))
        print(f"  Run {i+1:2d}: {ms:.4f} ms")

    sync_stats = compute_stats(sync_times)
    print(f"\n{'─' * 50}")
    print(f"  All {n_runs} runs:  mean={sync_stats['mean']:.4f} ms  std={sync_stats['std']:.4f} ms")
    print(f"  Clean runs ({sync_stats['n_clean']}/{n_runs}): "
          f"mean={sync_stats['clean_mean']:.4f} ms  std={sync_stats['clean_std']:.4f} ms")
    if sync_stats["n_dropped"]:
        print(f"  Dropped {sync_stats['n_dropped']} outlier run(s) (>2x median of {sync_stats['median']:.4f} ms)")
    print(f"{'─' * 50}\n")

    # --- v6_streams ---
    print("=== v6_streams (enable_streams=True) ===")
    streams_times = []
    for i in range(n_runs):
        ms = bench_single_run(X, params, enable_streams=True)
        streams_times.append(round(ms, 4))
        print(f"  Run {i+1:2d}: {ms:.4f} ms")

    streams_stats = compute_stats(streams_times)
    print(f"\n{'─' * 50}")
    print(f"  All {n_runs} runs:  mean={streams_stats['mean']:.4f} ms  std={streams_stats['std']:.4f} ms")
    print(f"  Clean runs ({streams_stats['n_clean']}/{n_runs}): "
          f"mean={streams_stats['clean_mean']:.4f} ms  std={streams_stats['clean_std']:.4f} ms")
    if streams_stats["n_dropped"]:
        print(f"  Dropped {streams_stats['n_dropped']} outlier run(s) (>2x median of {streams_stats['median']:.4f} ms)")
    print(f"{'─' * 50}\n")

    return {
        "batch": batch,
        "seq_len": seq_len,
        "hidden": hidden,
        "intermediate": intermediate,
        "rank": rank,
        "M": M,
        "dtype": str(dtype),
        "n_runs": n_runs,
        "v6_sync": {"runs_ms": sync_times, **sync_stats},
        "v6_streams": {"runs_ms": streams_times, **streams_stats},
    }


def save_results(result, path):
    """Save per-run results as CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows = []
    for i in range(result["n_runs"]):
        rows.append({
            "run": i + 1,
            "v6_sync_ms": result["v6_sync"]["runs_ms"][i],
            "v6_streams_ms": result["v6_streams"]["runs_ms"][i],
            "batch": result["batch"],
            "seq_len": result["seq_len"],
            "hidden": result["hidden"],
            "intermediate": result["intermediate"],
            "rank": result["rank"],
            "M": result["M"],
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = path.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"v6 LoRA MLP Benchmark Summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Config: batch={result['batch']} seq={result['seq_len']} "
                f"H={result['hidden']} I={result['intermediate']} "
                f"r={result['rank']} M={result['M']}\n")
        f.write(f"Runs: {result['n_runs']}\n\n")
        for variant in ["v6_sync", "v6_streams"]:
            s = result[variant]
            f.write(f"{variant}:\n")
            f.write(f"  Mean:       {s['mean']:.4f} ms (std={s['std']:.4f})\n")
            f.write(f"  Median:     {s['median']:.4f} ms\n")
            f.write(f"  Clean mean: {s['clean_mean']:.4f} ms "
                    f"(std={s['clean_std']:.4f}, {s['n_clean']}/{result['n_runs']} runs)\n\n")
    print(f"Results saved to {path}")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="v6 LoRA MLP latency benchmark (isolated)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--save", type=str, default=None, help="Directory to save CSV")
    args = parser.parse_args()

    result = bench_v6(
        args.batch, args.seq, args.hidden, args.intermediate,
        args.rank, torch.bfloat16, args.runs,
    )

    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.save, f"v6_{timestamp}.csv")
        save_results(result, path)


if __name__ == "__main__":
    main()

"""
Standalone Unsloth LoRA MLP latency benchmark.

Runs the Unsloth apply_lora_mlp_swiglu forward pass N times (default 10),
each run using triton.testing.do_bench with warmup+rep internally. Reports
per-run latency, mean, std, and a "clean mean" that drops outlier runs
(>2x the median) caused by GPU throttling or contention.

Usage:
    python benchmarks/bench_unsloth.py                          # defaults
    python benchmarks/bench_unsloth.py --runs 20 --batch 4 --seq 2048
    python benchmarks/bench_unsloth.py --rank 32 --save results/
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

from reference.unsloth_baseline import (
    apply_lora_mlp_swiglu as unsloth_lora_mlp,
    make_lora_mlp_params as unsloth_make_params,
)

DEVICE = "cuda"
WARMUP = 10
REP = 50


def bench_single_run(X, params):
    """One do_bench measurement of Unsloth forward."""
    def fn():
        return unsloth_lora_mlp(X, **params)
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def bench_unsloth(batch, seq_len, hidden, intermediate, rank, dtype, n_runs):
    """Run Unsloth benchmark n_runs times, return per-run and summary stats."""
    params = unsloth_make_params(
        hidden, intermediate, rank, dtype=dtype, device=DEVICE, requires_grad=False
    )
    X = torch.randn(batch, seq_len, hidden, dtype=dtype, device=DEVICE)
    M = batch * seq_len

    print(f"Config: batch={batch} seq={seq_len} hidden={hidden} "
          f"intermediate={intermediate} rank={rank} M={M} dtype={dtype}")
    print(f"Running {n_runs} benchmark runs ({WARMUP} warmup + {REP} reps each)\n")

    times = []
    for i in range(n_runs):
        ms = bench_single_run(X, params)
        times.append(round(ms, 4))
        print(f"  Run {i+1:2d}: {ms:.4f} ms")

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    std = math.sqrt(variance)

    median = sorted(times)[len(times) // 2]
    clean = [t for t in times if t < 2.0 * median]
    clean_mean = sum(clean) / len(clean) if clean else mean
    clean_var = sum((t - clean_mean) ** 2 for t in clean) / len(clean) if clean else variance
    clean_std = math.sqrt(clean_var)
    n_dropped = len(times) - len(clean)

    print(f"\n{'─' * 50}")
    print(f"  All {n_runs} runs:  mean={mean:.4f} ms  std={std:.4f} ms")
    print(f"  Clean runs ({len(clean)}/{n_runs}): mean={clean_mean:.4f} ms  std={clean_std:.4f} ms")
    if n_dropped:
        print(f"  Dropped {n_dropped} outlier run(s) (>2x median of {median:.4f} ms)")
    print(f"{'─' * 50}")

    return {
        "batch": batch,
        "seq_len": seq_len,
        "hidden": hidden,
        "intermediate": intermediate,
        "rank": rank,
        "M": M,
        "dtype": str(dtype),
        "n_runs": n_runs,
        "runs_ms": times,
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "median_ms": round(median, 4),
        "clean_mean_ms": round(clean_mean, 4),
        "clean_std_ms": round(clean_std, 4),
        "n_clean": len(clean),
        "n_dropped": n_dropped,
    }


def save_results(result, path):
    """Save per-run results as CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    rows = []
    for i, ms in enumerate(result["runs_ms"]):
        rows.append({
            "run": i + 1,
            "latency_ms": ms,
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
        f.write(f"Unsloth LoRA MLP Benchmark Summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Config: batch={result['batch']} seq={result['seq_len']} "
                f"H={result['hidden']} I={result['intermediate']} "
                f"r={result['rank']} M={result['M']}\n")
        f.write(f"Runs: {result['n_runs']}\n")
        f.write(f"Mean:       {result['mean_ms']:.4f} ms (std={result['std_ms']:.4f})\n")
        f.write(f"Median:     {result['median_ms']:.4f} ms\n")
        f.write(f"Clean mean: {result['clean_mean_ms']:.4f} ms "
                f"(std={result['clean_std_ms']:.4f}, {result['n_clean']}/{result['n_runs']} runs)\n")
    print(f"Results saved to {path}")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Unsloth LoRA MLP latency benchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--save", type=str, default=None, help="Directory to save CSV")
    args = parser.parse_args()

    result = bench_unsloth(
        args.batch, args.seq, args.hidden, args.intermediate,
        args.rank, torch.bfloat16, args.runs,
    )

    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.save, f"unsloth_{timestamp}.csv")
        save_results(result, path)


if __name__ == "__main__":
    main()

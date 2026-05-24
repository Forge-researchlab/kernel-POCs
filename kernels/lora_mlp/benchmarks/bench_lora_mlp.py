"""
Benchmark harness for LoRA MLP kernel experiments.

Compares:
  - PyTorch reference (matmul_lora / lora_swiglu_mlp) — mirrors Unsloth's approach
  - Triton v1 fused LoRA matmul

Two benchmark modes:
  - projection: single matmul_lora call (v1 vs Unsloth per-projection)
  - mlp: full MLP forward (gate + up + SwiGLU + down)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.unsloth_baseline import (
    matmul_lora as unsloth_matmul_lora,
    apply_lora_mlp_swiglu as unsloth_lora_mlp,
    make_lora_mlp_params as unsloth_make_params,
    swiglu_fg_kernel,
)
from experiments.v1.lora_mlp_kernel_v1 import fused_lora_matmul


DEVICE = "cuda"
WARMUP = 10
REP = 50


def bench_projection(M, N, K, rank, dtype, lora_scale=1.0):
    """Benchmark a single LoRA projection: X @ W + s * (X @ A) @ B."""
    X = torch.randn(M, K, device=DEVICE, dtype=dtype)
    W = torch.randn(N, K, device=DEVICE, dtype=dtype) * 0.02
    A = torch.randn(rank, K, device=DEVICE, dtype=dtype) * 0.02
    B = torch.randn(N, rank, device=DEVICE, dtype=dtype) * 0.02

    # Unsloth baseline (exact code: torch.matmul + addmm_, 3 cuBLAS calls)
    def unsloth_fn():
        return unsloth_matmul_lora(X, W, None, A, B, lora_scale)

    # Triton v1 (1 kernel launch)
    def triton_fn():
        return fused_lora_matmul(X, W, A, B, lora_scale)

    ms_unsloth = triton.testing.do_bench(unsloth_fn, warmup=WARMUP, rep=REP)
    ms_triton = triton.testing.do_bench(triton_fn, warmup=WARMUP, rep=REP)

    speedup = ms_unsloth / ms_triton if ms_triton > 0 else float("inf")
    return {
        "mode": "projection",
        "M": M, "N": N, "K": K, "rank": rank, "dtype": str(dtype),
        "unsloth_ms": round(ms_unsloth, 4),
        "triton_v1_ms": round(ms_triton, 4),
        "speedup": round(speedup, 3),
    }


def bench_mlp(batch, seq_len, hidden, intermediate, rank, dtype, lora_scale=1.0):
    """Benchmark full MLP forward: gate + up + SwiGLU + down."""
    M = batch * seq_len
    params = unsloth_make_params(
        hidden, intermediate, rank, dtype=dtype, device=DEVICE, requires_grad=False
    )
    X = torch.randn(batch, seq_len, hidden, dtype=dtype, device=DEVICE)

    # Unsloth baseline (exact code: LoRA_MLP.apply with Triton SwiGLU)
    def unsloth_fn():
        return unsloth_lora_mlp(X, **params)

    # Triton v1: 3x fused_lora_matmul + Unsloth's Triton SwiGLU kernel
    gp, up, dp = params["gate_proj"], params["up_proj"], params["down_proj"]
    def triton_fn():
        e = fused_lora_matmul(X, gp["W"], gp["A"], gp["B"], gp["s"])
        g = fused_lora_matmul(X, up["W"], up["A"], up["B"], up["s"])
        h = swiglu_fg_kernel(e, g)
        return fused_lora_matmul(h, dp["W"], dp["A"], dp["B"], dp["s"])

    ms_unsloth = triton.testing.do_bench(unsloth_fn, warmup=WARMUP, rep=REP)
    ms_triton = triton.testing.do_bench(triton_fn, warmup=WARMUP, rep=REP)

    speedup = ms_unsloth / ms_triton if ms_triton > 0 else float("inf")
    return {
        "mode": "mlp",
        "batch": batch, "seq_len": seq_len,
        "hidden": hidden, "intermediate": intermediate,
        "rank": rank, "dtype": str(dtype),
        "M": M,
        "unsloth_ms": round(ms_unsloth, 4),
        "triton_v1_ms": round(ms_triton, 4),
        "speedup": round(speedup, 3),
    }


def run_projection_sweep():
    """Per-projection benchmark sweep."""
    results = []
    configs = [
        # (M=batch*seq, N=out_dim, K=in_dim)
        # LLaMA-8B gate/up projection
        {"M": 2048, "N": 14336, "K": 4096},
        {"M": 4096, "N": 14336, "K": 4096},
        {"M": 8192, "N": 14336, "K": 4096},
        # LLaMA-8B down projection
        {"M": 2048, "N": 4096, "K": 14336},
        {"M": 4096, "N": 4096, "K": 14336},
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_projection(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                print(
                    f"  proj M={cfg['M']:5d} N={cfg['N']:5d} K={cfg['K']:5d} "
                    f"r={rank:2d} | unsloth={result['unsloth_ms']:.3f}ms "
                    f"triton={result['triton_v1_ms']:.3f}ms "
                    f"speedup={result['speedup']:.2f}x"
                )
    return results


def run_mlp_sweep():
    """Full MLP benchmark sweep."""
    results = []
    configs = [
        # LLaMA-8B
        {"batch": 1, "seq_len": 2048, "hidden": 4096, "intermediate": 14336},
        {"batch": 2, "seq_len": 1024, "hidden": 4096, "intermediate": 14336},
        {"batch": 4, "seq_len": 512, "hidden": 4096, "intermediate": 14336},
        {"batch": 4, "seq_len": 2048, "hidden": 4096, "intermediate": 14336},
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_mlp(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                M = cfg["batch"] * cfg["seq_len"]
                print(
                    f"  mlp  b={cfg['batch']} s={cfg['seq_len']:4d} "
                    f"M={M:5d} r={rank:2d} | unsloth={result['unsloth_ms']:.3f}ms "
                    f"triton={result['triton_v1_ms']:.3f}ms "
                    f"speedup={result['speedup']:.2f}x"
                )
    return results


def save_results(results, path):
    """Save benchmark results to CSV."""
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    keys = sorted(all_keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA MLP kernel benchmarks")
    parser.add_argument("--mode", choices=["projection", "mlp", "all"], default="all")
    parser.add_argument("--save", type=str, default=None, help="Directory to save CSV results")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    # Single config mode
    if args.hidden and args.intermediate:
        batch = args.batch or 4
        seq = args.seq or 2048
        rank = args.rank or 16
        print(f"\nSingle config: batch={batch} seq={seq} hidden={args.hidden} "
              f"intermediate={args.intermediate} rank={rank}\n")

        print("Per-projection:")
        r1 = bench_projection(batch * seq, args.intermediate, args.hidden, rank, torch.bfloat16)
        print(f"  gate/up: unsloth={r1['unsloth_ms']:.3f}ms triton={r1['triton_v1_ms']:.3f}ms speedup={r1['speedup']:.2f}x")
        r2 = bench_projection(batch * seq, args.hidden, args.intermediate, rank, torch.bfloat16)
        print(f"  down:    unsloth={r2['unsloth_ms']:.3f}ms triton={r2['triton_v1_ms']:.3f}ms speedup={r2['speedup']:.2f}x")

        print("\nFull MLP (Unsloth LoRA_MLP vs Triton v1):")
        r3 = bench_mlp(batch, seq, args.hidden, args.intermediate, rank, torch.bfloat16)
        print(f"  unsloth={r3['unsloth_ms']:.3f}ms triton={r3['triton_v1_ms']:.3f}ms speedup={r3['speedup']:.2f}x")
        return

    # Sweep mode
    all_results = []

    if args.mode in ("projection", "all"):
        print("\n=== Per-Projection Benchmarks ===\n")
        all_results.extend(run_projection_sweep())

    if args.mode in ("mlp", "all"):
        print("\n=== Full MLP Benchmarks ===\n")
        all_results.extend(run_mlp_sweep())

    if args.save and all_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_results(all_results, os.path.join(args.save, f"v1_{timestamp}.csv"))


if __name__ == "__main__":
    main()

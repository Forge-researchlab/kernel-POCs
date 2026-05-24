"""
Benchmark harness for LoRA QKV kernel experiments.

Compares:
  - PyTorch reference (matmul_lora / lora_qkv_forward) — mirrors Unsloth's approach
  - Triton kernel experiments (v1, v2, ...) as they are implemented

Two benchmark modes:
  - projection: single matmul_lora call (per-projection)
  - qkv: full QKV forward (all three projections)

Supports GQA configurations where num_kv_heads < num_heads.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.lora_qkv_pytorch import (
    matmul_lora,
    lora_qkv_forward,
    make_lora_qkv_params,
)


DEVICE = "cuda"
WARMUP = 10
REP = 50


# ---------------------------------------------------------------------------
# Per-projection benchmark
# ---------------------------------------------------------------------------

def bench_projection(
    M: int, N: int, K: int, rank: int, dtype: torch.dtype,
    lora_scale: float = 1.0,
) -> dict:
    """Benchmark a single LoRA projection: Y = X @ W^T + s * (X @ A^T) @ B^T."""
    X = torch.randn(M, K, device=DEVICE, dtype=dtype)
    W = torch.randn(N, K, device=DEVICE, dtype=dtype) * 0.02
    A = torch.randn(rank, K, device=DEVICE, dtype=dtype) * 0.02
    B = torch.randn(N, rank, device=DEVICE, dtype=dtype) * 0.02

    def pytorch_fn():
        return matmul_lora(X, W, A, B, lora_scale)

    ms_pytorch = triton.testing.do_bench(pytorch_fn, warmup=WARMUP, rep=REP)

    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    pytorch_fn()
    torch.cuda.synchronize()
    mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    result = {
        "mode": "projection",
        "M": M, "N": N, "K": K, "rank": rank, "dtype": str(dtype),
        "pytorch_ms": round(ms_pytorch, 4),
        "memory_mb": round(mem_mb, 1),
    }

    # Add kernel versions as they are implemented
    # Example for v1:
    # try:
    #     from experiments.v1.lora_qkv_kernel_v1 import fused_lora_matmul
    #     def triton_v1_fn():
    #         return fused_lora_matmul(X, W, A, B, lora_scale)
    #     ms_v1 = triton.testing.do_bench(triton_v1_fn, warmup=WARMUP, rep=REP)
    #     result["triton_v1_ms"] = round(ms_v1, 4)
    #     result["v1_speedup"] = round(ms_pytorch / ms_v1, 3)
    # except ImportError:
    #     pass

    return result


# ---------------------------------------------------------------------------
# Full QKV benchmark
# ---------------------------------------------------------------------------

def bench_qkv(
    batch: int, seq_len: int, hidden: int,
    num_heads: int, num_kv_heads: int, head_dim: int,
    rank: int, dtype: torch.dtype,
    lora_scale: float = 1.0,
) -> dict:
    """Benchmark full QKV forward: Q, K, V projections with LoRA."""
    M = batch * seq_len
    H_q = num_heads * head_dim
    H_kv = num_kv_heads * head_dim

    params = make_lora_qkv_params(
        hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
        requires_grad=False,
    )
    X = torch.randn(M, hidden, device=DEVICE, dtype=dtype)

    def pytorch_fn():
        return lora_qkv_forward(X, **params)

    ms_pytorch = triton.testing.do_bench(pytorch_fn, warmup=WARMUP, rep=REP)

    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    pytorch_fn()
    torch.cuda.synchronize()
    mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Compute FLOPs
    flops_base = 3 * 2 * M * hidden  # three matmuls (Q, K, V)
    flops_base_q = 2 * M * H_q * hidden
    flops_base_kv = 2 * (2 * M * H_kv * hidden)
    flops_lora = 3 * (2 * M * rank * hidden + 2 * M * rank)  # A and B matmuls
    flops_lora_q = 2 * M * rank * hidden + 2 * M * H_q * rank
    flops_lora_kv = 2 * (2 * M * rank * hidden + 2 * M * H_kv * rank)
    total_flops = flops_base_q + flops_base_kv + flops_lora_q + flops_lora_kv
    tflops = total_flops / (ms_pytorch * 1e-3) / 1e12

    result = {
        "mode": "qkv",
        "batch": batch, "seq_len": seq_len, "hidden": hidden,
        "num_heads": num_heads, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "H_q": H_q, "H_kv": H_kv,
        "M": M, "rank": rank, "dtype": str(dtype),
        "pytorch_ms": round(ms_pytorch, 4),
        "tflops": round(tflops, 2),
        "memory_mb": round(mem_mb, 1),
    }

    return result


# ---------------------------------------------------------------------------
# Backward benchmark
# ---------------------------------------------------------------------------

def bench_backward(
    batch: int, seq_len: int, hidden: int,
    num_heads: int, num_kv_heads: int, head_dim: int,
    rank: int, dtype: torch.dtype,
) -> dict:
    """Benchmark QKV backward pass through LoRAQKV autograd.Function."""
    from reference.lora_qkv_pytorch import LoRAQKV

    M = batch * seq_len
    params = make_lora_qkv_params(
        hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
        requires_grad=True,
    )

    X = torch.randn(M, hidden, device=DEVICE, dtype=dtype, requires_grad=True)

    def fwd_bwd():
        Q, K, V = LoRAQKV.apply(
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        loss = Q.sum() + K.sum() + V.sum()
        loss.backward()

    ms_total = triton.testing.do_bench(fwd_bwd, warmup=WARMUP, rep=REP)

    return {
        "mode": "backward",
        "batch": batch, "seq_len": seq_len, "hidden": hidden,
        "num_heads": num_heads, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "M": M, "rank": rank, "dtype": str(dtype),
        "pytorch_fwd_bwd_ms": round(ms_total, 4),
    }


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

def run_projection_sweep():
    """Per-projection benchmark sweep."""
    results = []
    configs = [
        # Q projection (M=batch*seq, N=H_q, K=H)
        {"M": 2048, "N": 4096, "K": 4096},    # LLaMA-8B Q (small)
        {"M": 4096, "N": 4096, "K": 4096},    # LLaMA-8B Q (medium)
        {"M": 8192, "N": 4096, "K": 4096},    # LLaMA-8B Q (large)
        # K/V projection with GQA
        {"M": 2048, "N": 1024, "K": 4096},    # LLaMA-8B K/V (GQA, small)
        {"M": 8192, "N": 1024, "K": 4096},    # LLaMA-8B K/V (GQA, large)
        # LLaMA-70B Q
        {"M": 4096, "N": 8192, "K": 8192},    # LLaMA-70B Q
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_projection(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                print(
                    f"  proj M={cfg['M']:5d} N={cfg['N']:5d} K={cfg['K']:5d} "
                    f"r={rank:2d} | pytorch={result['pytorch_ms']:.3f}ms"
                )
    return results


def run_qkv_sweep():
    """Full QKV benchmark sweep."""
    results = []
    configs = [
        # LLaMA-3 8B with GQA
        {"batch": 1, "seq_len": 2048, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"batch": 4, "seq_len": 512, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"batch": 4, "seq_len": 2048, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        # LLaMA-3 8B with MHA (all 32 kv heads)
        {"batch": 4, "seq_len": 2048, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},
        # Smaller model
        {"batch": 4, "seq_len": 2048, "hidden": 2048,
         "num_heads": 16, "num_kv_heads": 4, "head_dim": 128},
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_qkv(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                M = cfg["batch"] * cfg["seq_len"]
                gqa_str = "MHA" if cfg["num_heads"] == cfg["num_kv_heads"] else f"GQA({cfg['num_kv_heads']})"
                print(
                    f"  qkv  b={cfg['batch']} s={cfg['seq_len']:4d} "
                    f"M={M:5d} h={cfg['hidden']} {gqa_str:8s} "
                    f"r={rank:2d} | pytorch={result['pytorch_ms']:.3f}ms "
                    f"TFLOPS={result['tflops']:.1f}"
                )
    return results


def run_backward_sweep():
    """Backward pass benchmark sweep."""
    results = []
    configs = [
        {"batch": 4, "seq_len": 2048, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"batch": 4, "seq_len": 2048, "hidden": 4096,
         "num_heads": 32, "num_kv_heads": 32, "head_dim": 128},
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_backward(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                M = cfg["batch"] * cfg["seq_len"]
                print(
                    f"  bwd  b={cfg['batch']} s={cfg['seq_len']:4d} M={M:5d} "
                    f"r={rank:2d} | fwd+bwd={result['pytorch_fwd_bwd_ms']:.3f}ms"
                )
    return results


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA QKV kernel benchmarks")
    parser.add_argument(
        "--mode", choices=["projection", "qkv", "backward", "forward", "all"],
        default="all",
        help="Benchmark mode: projection (single matmul), qkv (all Q/K/V), "
             "backward (fwd+bwd), forward (alias for qkv), all",
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save CSV results")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep across shapes and ranks")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp32"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Single config mode
    if args.hidden and args.num_heads:
        batch = args.batch or 4
        seq = args.seq or 2048
        rank = args.rank or 16
        num_kv_heads = args.num_kv_heads or args.num_heads
        head_dim = args.head_dim

        H_q = args.num_heads * head_dim
        H_kv = num_kv_heads * head_dim
        gqa_str = "MHA" if num_kv_heads == args.num_heads else f"GQA({num_kv_heads})"

        print(f"\nSingle config: batch={batch} seq={seq} hidden={args.hidden} "
              f"heads={args.num_heads} kv_heads={num_kv_heads} head_dim={head_dim} "
              f"rank={rank} {gqa_str}\n")

        M = batch * seq

        if args.mode in ("projection", "all"):
            print("Per-projection:")
            r_q = bench_projection(M, H_q, args.hidden, rank, dtype)
            print(f"  Q proj: pytorch={r_q['pytorch_ms']:.3f}ms")
            r_kv = bench_projection(M, H_kv, args.hidden, rank, dtype)
            print(f"  K/V proj: pytorch={r_kv['pytorch_ms']:.3f}ms")

        if args.mode in ("qkv", "forward", "all"):
            print("\nFull QKV forward:")
            r_qkv = bench_qkv(
                batch, seq, args.hidden, args.num_heads, num_kv_heads,
                head_dim, rank, dtype,
            )
            print(f"  pytorch={r_qkv['pytorch_ms']:.3f}ms "
                  f"TFLOPS={r_qkv['tflops']:.1f}")

        if args.mode in ("backward", "all"):
            print("\nForward + Backward:")
            r_bwd = bench_backward(
                batch, seq, args.hidden, args.num_heads, num_kv_heads,
                head_dim, rank, dtype,
            )
            print(f"  pytorch fwd+bwd={r_bwd['pytorch_fwd_bwd_ms']:.3f}ms")
        return

    # Sweep mode
    all_results = []

    if args.mode in ("projection", "all"):
        print("\n=== Per-Projection Benchmarks ===\n")
        all_results.extend(run_projection_sweep())

    if args.mode in ("qkv", "forward", "all"):
        print("\n=== Full QKV Benchmarks ===\n")
        all_results.extend(run_qkv_sweep())

    if args.mode in ("backward", "all"):
        print("\n=== Backward Benchmarks ===\n")
        all_results.extend(run_backward_sweep())

    if args.save and all_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version = "baseline"
        save_results(
            all_results,
            os.path.join(args.save, f"{version}_{timestamp}.csv"),
        )


if __name__ == "__main__":
    main()

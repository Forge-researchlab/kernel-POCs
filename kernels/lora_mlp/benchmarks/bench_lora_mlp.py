"""
Benchmark harness for LoRA MLP kernel experiments.

Compares (full MLP forward at LLaMA scale):
  - Unsloth's `apply_lora_mlp_swiglu` (10-launch baseline)
  - Triton v1 fused per-projection LoRA matmul
  - v3 cuBLAS + Triton fused LoRA-SwiGLU epilogue (8 launches)
  - v5 packed cuBLAS + Triton epilogue, training path (4 launches)
  - v5_upgrade_1 padded gate+up mega + v3-style down (5 launches)
  - v5 inference path with pre-merged weights (4 launches; cuBLAS-only when
    cublasLt SWISH is available, otherwise still 4 launches with 1 Triton op)

Two benchmark modes:
  - projection: single matmul_lora call (v1 vs Unsloth per-projection)
  - mlp:        full MLP forward, including v3 and v5 paths
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
from experiments.v3.lora_mlp_kernel_v3 import lora_mlp_v3
from experiments.v5.lora_mlp_kernel_v5 import (
    lora_mlp_v5,
    lora_mlp_v5_inference,
    pack_gate_up_weights,
    pack_down_weights,
    prepare_inference_weights,
)
from experiments.v5.lora_mlp_kernel_v5_upgrade_1 import (
    lora_mlp_v5_upgrade_1,
    pack_gate_up_weights_padded,
)


DEVICE = "cuda"
WARMUP = 10
REP = 50


# ---------------------------------------------------------------------------
# Per-projection benchmark (v1 vs Unsloth)
# ---------------------------------------------------------------------------

def bench_projection(M, N, K, rank, dtype, lora_scale=1.0):
    """Benchmark a single LoRA projection: X @ W + s * (X @ A) @ B."""
    X = torch.randn(M, K, device=DEVICE, dtype=dtype)
    W = torch.randn(N, K, device=DEVICE, dtype=dtype) * 0.02
    A = torch.randn(rank, K, device=DEVICE, dtype=dtype) * 0.02
    B = torch.randn(N, rank, device=DEVICE, dtype=dtype) * 0.02

    def unsloth_fn():
        return unsloth_matmul_lora(X, W, None, A, B, lora_scale)

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


# ---------------------------------------------------------------------------
# Full MLP benchmark (Unsloth, v3, v5 train, v5 inference)
# ---------------------------------------------------------------------------

def bench_mlp(batch, seq_len, hidden, intermediate, rank, dtype, lora_scale=1.0):
    """Benchmark full MLP forward: gate + up + SwiGLU + down across implementations."""
    M = batch * seq_len
    params = unsloth_make_params(
        hidden, intermediate, rank, dtype=dtype, device=DEVICE, requires_grad=False
    )
    X = torch.randn(batch, seq_len, hidden, dtype=dtype, device=DEVICE)

    gp, up, dp = params["gate_proj"], params["up_proj"], params["down_proj"]

    # ── Unsloth baseline ──
    def unsloth_fn():
        return unsloth_lora_mlp(X, **params)

    # ── v3 (cuBLAS + Triton epilogue) ──
    def v3_fn():
        return lora_mlp_v3(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )

    # ── v5 training (packed mega-GEMM + Triton epilogue) ──
    # Pack weights once outside the timed loop to mirror real training (where
    # the packed buffers are created once per parameter update, not per fwd).
    W_mega = pack_gate_up_weights(gp["W"], up["W"], gp["A"], up["A"])
    W_down_packed = pack_down_weights(dp["W"], dp["A"])

    def v5_train_fn():
        return lora_mlp_v5(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
            W_mega=W_mega,
            W_down_packed=W_down_packed,
        )

    # ── v5_upgrade_1 training (padded gate+up mega + v3-style down) ──
    # Pack the padded mega-matrix once outside the timed loop, just like v5.
    W_mega_padded, _pad_rows = pack_gate_up_weights_padded(
        gp["W"], up["W"], gp["A"], up["A"]
    )

    def v5_upgrade_1_train_fn():
        return lora_mlp_v5_upgrade_1(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
            W_mega_padded=W_mega_padded,
        )

    # ── v5 inference (pre-merged + transposed once) ──
    W_gate_eff_T, W_up_eff_T, W_down_eff_T = prepare_inference_weights(
        gp["W"], gp["A"], gp["B"], gp["s"],
        up["W"], up["A"], up["B"], up["s"],
        dp["W"], dp["A"], dp["B"], dp["s"],
    )

    def v5_inf_fn():
        return lora_mlp_v5_inference(X, W_gate_eff_T, W_up_eff_T, W_down_eff_T)

    ms_unsloth = triton.testing.do_bench(unsloth_fn, warmup=WARMUP, rep=REP)
    ms_v3 = triton.testing.do_bench(v3_fn, warmup=WARMUP, rep=REP)
    ms_v5_train = triton.testing.do_bench(v5_train_fn, warmup=WARMUP, rep=REP)
    ms_v5_up1_train = triton.testing.do_bench(v5_upgrade_1_train_fn, warmup=WARMUP, rep=REP)
    ms_v5_inf = triton.testing.do_bench(v5_inf_fn, warmup=WARMUP, rep=REP)

    def safe_div(a, b):
        return round(a / b, 3) if b > 0 else float("inf")

    return {
        "mode": "mlp",
        "batch": batch, "seq_len": seq_len,
        "hidden": hidden, "intermediate": intermediate,
        "rank": rank, "dtype": str(dtype),
        "M": M,
        "unsloth_ms": round(ms_unsloth, 4),
        "v3_ms": round(ms_v3, 4),
        "v5_train_ms": round(ms_v5_train, 4),
        "v5_up1_train_ms": round(ms_v5_up1_train, 4),
        "v5_inf_ms": round(ms_v5_inf, 4),
        "v5_train_vs_unsloth": safe_div(ms_unsloth, ms_v5_train),
        "v5_train_vs_v3": safe_div(ms_v3, ms_v5_train),
        "v5_up1_train_vs_unsloth": safe_div(ms_unsloth, ms_v5_up1_train),
        "v5_up1_train_vs_v3": safe_div(ms_v3, ms_v5_up1_train),
        "v5_up1_train_vs_v5": safe_div(ms_v5_train, ms_v5_up1_train),
        "v5_inf_vs_unsloth": safe_div(ms_unsloth, ms_v5_inf),
        "v5_inf_vs_v3": safe_div(ms_v3, ms_v5_inf),
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_projection_sweep():
    """Per-projection benchmark sweep."""
    results = []
    configs = [
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
    """Full MLP benchmark sweep across Unsloth/v3/v5_train/v5_up1_train/v5_inf."""
    results = []
    configs = [
        # LLaMA-8B
        {"batch": 1, "seq_len": 2048, "hidden": 4096, "intermediate": 14336},
        {"batch": 2, "seq_len": 1024, "hidden": 4096, "intermediate": 14336},
        {"batch": 4, "seq_len": 512, "hidden": 4096, "intermediate": 14336},
        {"batch": 4, "seq_len": 2048, "hidden": 4096, "intermediate": 14336},
        # LLaMA-13B-ish
        {"batch": 1, "seq_len": 2048, "hidden": 5120, "intermediate": 17920},
    ]
    for cfg in configs:
        for rank in [8, 16, 32, 64]:
            for dtype in [torch.bfloat16]:
                result = bench_mlp(**cfg, rank=rank, dtype=dtype)
                results.append(result)
                M = cfg["batch"] * cfg["seq_len"]
                print(
                    f"  mlp  b={cfg['batch']} s={cfg['seq_len']:4d} "
                    f"H={cfg['hidden']} I={cfg['intermediate']} "
                    f"M={M:5d} r={rank:2d} | "
                    f"unsl={result['unsloth_ms']:.2f} "
                    f"v3={result['v3_ms']:.2f} "
                    f"v5_tr={result['v5_train_ms']:.2f} "
                    f"v5up1_tr={result['v5_up1_train_ms']:.2f} "
                    f"v5_in={result['v5_inf_ms']:.2f} | "
                    f"v5_tr/v3={result['v5_train_vs_v3']:.2f}x "
                    f"v5up1/v3={result['v5_up1_train_vs_v3']:.2f}x "
                    f"v5_in/v3={result['v5_inf_vs_v3']:.2f}x"
                )
    return results


# ---------------------------------------------------------------------------
# IO
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
# CLI
# ---------------------------------------------------------------------------

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

        if args.mode in ("projection", "all"):
            print("Per-projection:")
            r1 = bench_projection(batch * seq, args.intermediate, args.hidden, rank, torch.bfloat16)
            print(f"  gate/up: unsloth={r1['unsloth_ms']:.3f}ms triton={r1['triton_v1_ms']:.3f}ms speedup={r1['speedup']:.2f}x")
            r2 = bench_projection(batch * seq, args.hidden, args.intermediate, rank, torch.bfloat16)
            print(f"  down:    unsloth={r2['unsloth_ms']:.3f}ms triton={r2['triton_v1_ms']:.3f}ms speedup={r2['speedup']:.2f}x")

        if args.mode in ("mlp", "all"):
            print("\nFull MLP forward (Unsloth vs v3 vs v5_train vs v5_up1_train vs v5_inf):")
            r3 = bench_mlp(batch, seq, args.hidden, args.intermediate, rank, torch.bfloat16)
            print(
                f"  unsloth={r3['unsloth_ms']:.3f}ms "
                f"v3={r3['v3_ms']:.3f}ms "
                f"v5_train={r3['v5_train_ms']:.3f}ms "
                f"v5_up1_train={r3['v5_up1_train_ms']:.3f}ms "
                f"v5_inf={r3['v5_inf_ms']:.3f}ms"
            )
            print(
                f"  v5_train:     {r3['v5_train_vs_unsloth']:.2f}x vs Unsloth, {r3['v5_train_vs_v3']:.2f}x vs v3"
            )
            print(
                f"  v5_up1_train: {r3['v5_up1_train_vs_unsloth']:.2f}x vs Unsloth, {r3['v5_up1_train_vs_v3']:.2f}x vs v3, {r3['v5_up1_train_vs_v5']:.2f}x vs v5"
            )
            print(
                f"  v5_inf:       {r3['v5_inf_vs_unsloth']:.2f}x vs Unsloth, {r3['v5_inf_vs_v3']:.2f}x vs v3"
            )

        if args.save:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results = []
            if args.mode in ("mlp", "all"):
                results.append(r3)
            save_results(results, os.path.join(args.save, f"v5_upgrade_1_{timestamp}.csv"))
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
        save_results(all_results, os.path.join(args.save, f"v5_upgrade_1_{timestamp}.csv"))


if __name__ == "__main__":
    main()

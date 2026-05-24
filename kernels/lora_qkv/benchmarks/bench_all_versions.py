"""
Comprehensive benchmark: all LoRA QKV kernel versions.

Compares PyTorch, Unsloth, v1, v2, v2_2, v2_3, v3 on:
  - Full QKV forward (batch=4, seq=2048, hidden=4096, GQA 32/8, bf16)
  - Ranks: 8, 16, 32, 64
  - Peak memory at rank=16
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.lora_qkv_pytorch import lora_qkv_forward, make_lora_qkv_params, LoRAQKV
from reference.unsloth_baseline import qkv_lora_unsloth, LoRAQKVUnsloth
from experiments.v1.lora_qkv_kernel_v1 import lora_qkv_v1
from experiments.v2.lora_qkv_kernel_v2 import lora_qkv_v2, pack_weights as pack_weights_v2
from experiments.v2.lora_qkv_kernel_v2_2 import lora_qkv_v2_2, pack_weights as pack_weights_v2_2
from experiments.v2.lora_qkv_kernel_v2_3 import lora_qkv_v2_3, pack_weights_all
from experiments.v3.lora_qkv_kernel_v3 import lora_qkv_v3

DEVICE = "cuda"
WARMUP = 10
REP = 50


def measure_memory(fn, clear_first=True):
    """Measure peak memory delta for a single call."""
    if clear_first:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()
    fn()
    torch.cuda.synchronize()
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1024**2


def bench_full_qkv():
    """Benchmark all versions at LLaMA-3 8B GQA scale."""
    batch, seq, hidden = 4, 2048, 4096
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dtype = torch.bfloat16
    M = batch * seq

    ranks = [8, 16, 32, 64]
    results = []

    for rank in ranks:
        print(f"\n--- Rank={rank} ---")
        torch.manual_seed(42)

        params = make_lora_qkv_params(
            hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
            requires_grad=False,
        )
        X = torch.randn(M, hidden, device=DEVICE, dtype=dtype)

        p = params  # shorthand
        W_q, A_q, B_q, s_q = p['W_q'], p['A_q'], p['B_q'], p['s_q']
        W_k, A_k, B_k, s_k = p['W_k'], p['A_k'], p['B_k'], p['s_k']
        W_v, A_v, B_v, s_v = p['W_v'], p['A_v'], p['B_v'], p['s_v']

        # Pre-pack weights
        Wp_q = pack_weights_v2(W_q, A_q)
        Wp_k = pack_weights_v2(W_k, A_k)
        Wp_v = pack_weights_v2(W_v, A_v)
        W_all = pack_weights_all(W_q, A_q, W_k, A_k, W_v, A_v)

        # --- PyTorch reference ---
        def fn_pytorch():
            return lora_qkv_forward(X, **{
                'W_q': W_q, 'W_k': W_k, 'W_v': W_v,
                'A_q': A_q, 'B_q': B_q, 's_q': s_q,
                'A_k': A_k, 'B_k': B_k, 's_k': s_k,
                'A_v': A_v, 'B_v': B_v, 's_v': s_v,
            })
        ms_pytorch = triton.testing.do_bench(fn_pytorch, warmup=WARMUP, rep=REP)

        # --- Unsloth ---
        def fn_unsloth():
            return qkv_lora_unsloth(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v)
        ms_unsloth = triton.testing.do_bench(fn_unsloth, warmup=WARMUP, rep=REP)

        # --- v1 ---
        def fn_v1():
            return lora_qkv_v1(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v)
        ms_v1 = triton.testing.do_bench(fn_v1, warmup=WARMUP, rep=REP)

        # --- v2 ---
        def fn_v2():
            return lora_qkv_v2(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v,
                              Wp_q, Wp_k, Wp_v)
        ms_v2 = triton.testing.do_bench(fn_v2, warmup=WARMUP, rep=REP)

        # --- v2_2 ---
        def fn_v2_2():
            return lora_qkv_v2_2(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v,
                                Wp_q, Wp_k, Wp_v)
        ms_v2_2 = triton.testing.do_bench(fn_v2_2, warmup=WARMUP, rep=REP)

        # --- v2_3 ---
        def fn_v2_3():
            return lora_qkv_v2_3(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v,
                                W_all=W_all)
        ms_v2_3 = triton.testing.do_bench(fn_v2_3, warmup=WARMUP, rep=REP)

        # --- v3 (forward only, no grad) ---
        def fn_v3():
            return lora_qkv_v3(X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v,
                              W_all=W_all)
        ms_v3 = triton.testing.do_bench(fn_v3, warmup=WARMUP, rep=REP)

        row = {
            "rank": rank,
            "pytorch_ms": round(ms_pytorch, 3),
            "unsloth_ms": round(ms_unsloth, 3),
            "v1_ms": round(ms_v1, 3),
            "v2_ms": round(ms_v2, 3),
            "v2_2_ms": round(ms_v2_2, 3),
            "v2_3_ms": round(ms_v2_3, 3),
            "v3_ms": round(ms_v3, 3),
            "v2_vs_unsloth": round(ms_unsloth / ms_v2, 2),
            "v2_2_vs_unsloth": round(ms_unsloth / ms_v2_2, 2),
            "v2_3_vs_unsloth": round(ms_unsloth / ms_v2_3, 2),
            "v3_vs_unsloth": round(ms_unsloth / ms_v3, 2),
        }

        # Memory measurement at rank=16
        if rank == 16:
            mem_pytorch = measure_memory(fn_pytorch)
            mem_unsloth = measure_memory(fn_unsloth)
            mem_v1 = measure_memory(fn_v1)
            mem_v2 = measure_memory(fn_v2)
            mem_v2_2 = measure_memory(fn_v2_2)
            mem_v2_3 = measure_memory(fn_v2_3)
            mem_v3 = measure_memory(fn_v3)
            row.update({
                "mem_pytorch_mb": round(mem_pytorch, 1),
                "mem_unsloth_mb": round(mem_unsloth, 1),
                "mem_v1_mb": round(mem_v1, 1),
                "mem_v2_mb": round(mem_v2, 1),
                "mem_v2_2_mb": round(mem_v2_2, 1),
                "mem_v2_3_mb": round(mem_v2_3, 1),
                "mem_v3_mb": round(mem_v3, 1),
            })
            print(f"  Memory (MB): PyTorch={mem_pytorch:.0f} Unsloth={mem_unsloth:.0f} "
                  f"v1={mem_v1:.0f} v2={mem_v2:.0f} v2_2={mem_v2_2:.0f} "
                  f"v2_3={mem_v2_3:.0f} v3={mem_v3:.0f}")

        results.append(row)

        print(f"  PyTorch: {ms_pytorch:.3f}ms | Unsloth: {ms_unsloth:.3f}ms")
        print(f"  v1: {ms_v1:.3f}ms | v2: {ms_v2:.3f}ms | v2_2: {ms_v2_2:.3f}ms")
        print(f"  v2_3: {ms_v2_3:.3f}ms | v3: {ms_v3:.3f}ms")
        print(f"  Speedup vs Unsloth: v2={row['v2_vs_unsloth']}x "
              f"v2_2={row['v2_2_vs_unsloth']}x "
              f"v2_3={row['v2_3_vs_unsloth']}x "
              f"v3={row['v3_vs_unsloth']}x")

    return results


def bench_v3_backward():
    """Benchmark v3 forward+backward vs Unsloth and PyTorch reference."""
    batch, seq, hidden = 4, 2048, 4096
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dtype = torch.bfloat16
    M = batch * seq

    ranks = [8, 16, 32, 64]
    results = []

    for rank in ranks:
        print(f"\n--- Backward Rank={rank} ---")
        torch.manual_seed(42)

        params = make_lora_qkv_params(
            hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, hidden, device=DEVICE, dtype=dtype, requires_grad=True)
        W_all = pack_weights_all(
            params['W_q'], params['A_q'],
            params['W_k'], params['A_k'],
            params['W_v'], params['A_v'],
        )

        # Unsloth fwd+bwd
        def fn_unsloth_bwd():
            Q, K, V = LoRAQKVUnsloth.apply(
                X,
                params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_unsloth = triton.testing.do_bench(fn_unsloth_bwd, warmup=WARMUP, rep=REP)

        # PyTorch reference fwd+bwd
        def fn_ref_bwd():
            Q, K, V = LoRAQKV.apply(
                X,
                params['W_q'], params['W_k'], params['W_v'],
                params['A_q'], params['B_q'], params['s_q'],
                params['A_k'], params['B_k'], params['s_k'],
                params['A_v'], params['B_v'], params['s_v'],
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_ref = triton.testing.do_bench(fn_ref_bwd, warmup=WARMUP, rep=REP)

        # v3 fwd+bwd
        def fn_v3_bwd():
            Q, K, V = lora_qkv_v3(
                X,
                params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
                W_all=W_all,
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_v3 = triton.testing.do_bench(fn_v3_bwd, warmup=WARMUP, rep=REP)

        row = {
            "rank": rank,
            "mode": "fwd+bwd",
            "ref_ms": round(ms_ref, 3),
            "unsloth_ms": round(ms_unsloth, 3),
            "v3_ms": round(ms_v3, 3),
            "v3_vs_unsloth": round(ms_unsloth / ms_v3, 2),
        }
        results.append(row)
        print(f"  Ref: {ms_ref:.3f}ms | Unsloth: {ms_unsloth:.3f}ms | v3: {ms_v3:.3f}ms")
        print(f"  v3/Unsloth: {row['v3_vs_unsloth']}x")

    return results


def save_csv(results, path):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(results[0].keys())
    for r in results[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Full QKV Forward Benchmark")
    print("Config: batch=4, seq=2048, hidden=4096, GQA 32/8, bf16")
    print("GPU:", torch.cuda.get_device_name())
    print("=" * 60)

    fwd_results = bench_full_qkv()

    print("\n" + "=" * 60)
    print("Forward + Backward Benchmark (v3)")
    print("=" * 60)

    bwd_results = bench_v3_backward()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_csv(
        fwd_results + bwd_results,
        f"benchmarks/results/v2_2_v2_3_v3_{timestamp}.csv",
    )

    # Summary table
    print("\n" + "=" * 60)
    print("FORWARD SUMMARY (Full QKV, bf16, LLaMA-3 8B GQA)")
    print("=" * 60)
    print(f"{'Rank':>4} | {'PyTorch':>8} | {'Unsloth':>8} | {'v1':>8} | {'v2':>8} | {'v2_2':>8} | {'v2_3':>8} | {'v3':>8} | {'v2_3/US':>8}")
    print("-" * 85)
    for r in fwd_results:
        print(f"{r['rank']:>4} | {r['pytorch_ms']:>7.3f}ms | {r['unsloth_ms']:>7.3f}ms | "
              f"{r['v1_ms']:>7.3f}ms | {r['v2_ms']:>7.3f}ms | {r['v2_2_ms']:>7.3f}ms | "
              f"{r['v2_3_ms']:>7.3f}ms | {r['v3_ms']:>7.3f}ms | {r['v2_3_vs_unsloth']:>7.2f}x")

    # Memory row
    for r in fwd_results:
        if "mem_unsloth_mb" in r:
            print(f"\nMemory (rank=16): Unsloth={r['mem_unsloth_mb']:.0f}MB "
                  f"v2={r['mem_v2_mb']:.0f}MB v2_2={r['mem_v2_2_mb']:.0f}MB "
                  f"v2_3={r['mem_v2_3_mb']:.0f}MB v3={r['mem_v3_mb']:.0f}MB")

    print("\nBACKWARD SUMMARY (fwd+bwd)")
    print(f"{'Rank':>4} | {'Ref':>8} | {'Unsloth':>8} | {'v3':>8} | {'v3/US':>8}")
    print("-" * 50)
    for r in bwd_results:
        print(f"{r['rank']:>4} | {r['ref_ms']:>7.3f}ms | {r['unsloth_ms']:>7.3f}ms | "
              f"{r['v3_ms']:>7.3f}ms | {r['v3_vs_unsloth']:>7.2f}x")

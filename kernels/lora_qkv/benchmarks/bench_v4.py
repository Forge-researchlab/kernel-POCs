"""
Benchmark: v4 (packed backward) vs v3 (separate backward) vs Unsloth.

Measures forward+backward time at LLaMA-3 8B GQA scale.
Config: batch=4, seq=2048, hidden=4096, GQA 32/8, bf16, ranks=[8, 16, 32, 64]
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.lora_qkv_pytorch import make_lora_qkv_params, LoRAQKV
from reference.unsloth_baseline import LoRAQKVUnsloth
from experiments.v2.lora_qkv_kernel_v2_3 import pack_weights_all
from experiments.v3.lora_qkv_kernel_v3 import lora_qkv_v3
from experiments.v4.lora_qkv_kernel_v4 import (
    lora_qkv_v4,
    pack_weights_backward,
    pack_lora_a,
)

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


def bench_v4_backward():
    """Benchmark v4 fwd+bwd vs v3 and Unsloth."""
    batch, seq, hidden = 4, 2048, 4096
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dtype = torch.bfloat16
    M = batch * seq

    ranks = [8, 16, 32, 64]
    results = []

    print("=" * 70)
    print("v4 Forward + Backward Benchmark")
    print(f"Config: batch={batch}, seq={seq}, hidden={hidden}, GQA {num_heads}/{num_kv_heads}, bf16")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)

    for rank in ranks:
        print(f"\n--- Rank={rank} ---")
        torch.manual_seed(42)

        params = make_lora_qkv_params(
            hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, hidden, device=DEVICE, dtype=dtype, requires_grad=True)

        # Pre-pack weights
        W_all = pack_weights_all(
            params['W_q'], params['A_q'],
            params['W_k'], params['A_k'],
            params['W_v'], params['A_v'],
        )
        W_dX = pack_weights_backward(params['W_q'], params['W_k'], params['W_v'])
        A_pack = pack_lora_a(params['A_q'], params['A_k'], params['A_v'])

        # --- Unsloth fwd+bwd ---
        def fn_unsloth():
            Q, K, V = LoRAQKVUnsloth.apply(
                X,
                params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_unsloth = triton.testing.do_bench(fn_unsloth, warmup=WARMUP, rep=REP)

        # --- v3 fwd+bwd ---
        def fn_v3():
            Q, K, V = lora_qkv_v3(
                X,
                params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
                W_all=W_all,
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_v3 = triton.testing.do_bench(fn_v3, warmup=WARMUP, rep=REP)

        # --- v4 fwd+bwd ---
        def fn_v4():
            Q, K, V = lora_qkv_v4(
                X,
                params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
                W_all=W_all, W_dX_packed=W_dX, A_packed=A_pack,
            )
            (Q.sum() + K.sum() + V.sum()).backward()
        ms_v4 = triton.testing.do_bench(fn_v4, warmup=WARMUP, rep=REP)

        row = {
            "rank": rank,
            "unsloth_ms": round(ms_unsloth, 3),
            "v3_ms": round(ms_v3, 3),
            "v4_ms": round(ms_v4, 3),
            "v3_vs_unsloth": round(ms_unsloth / ms_v3, 2),
            "v4_vs_unsloth": round(ms_unsloth / ms_v4, 2),
            "v4_vs_v3": round(ms_v3 / ms_v4, 2),
        }

        # Memory at rank=16
        if rank == 16:
            mem_unsloth = measure_memory(fn_unsloth)
            mem_v3 = measure_memory(fn_v3)
            mem_v4 = measure_memory(fn_v4)
            row.update({
                "mem_unsloth_mb": round(mem_unsloth, 1),
                "mem_v3_mb": round(mem_v3, 1),
                "mem_v4_mb": round(mem_v4, 1),
            })
            print(f"  Memory (MB): Unsloth={mem_unsloth:.0f} v3={mem_v3:.0f} v4={mem_v4:.0f}")

        results.append(row)
        print(f"  Unsloth: {ms_unsloth:.3f}ms | v3: {ms_v3:.3f}ms | v4: {ms_v4:.3f}ms")
        print(f"  v3/Unsloth: {row['v3_vs_unsloth']}x | v4/Unsloth: {row['v4_vs_unsloth']}x | v4/v3: {row['v4_vs_v3']}x")

    return results


def bench_v4_forward_only():
    """Verify v4 forward matches v3 speed (should be identical)."""
    batch, seq, hidden = 4, 2048, 4096
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    dtype = torch.bfloat16
    M = batch * seq

    ranks = [8, 16, 32, 64]
    results = []

    print("\n" + "=" * 70)
    print("v4 Forward-Only Benchmark (should match v3)")
    print("=" * 70)

    for rank in ranks:
        torch.manual_seed(42)
        params = make_lora_qkv_params(
            hidden_dim=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
            requires_grad=False,
        )
        X = torch.randn(M, hidden, device=DEVICE, dtype=dtype)
        W_all = pack_weights_all(
            params['W_q'], params['A_q'],
            params['W_k'], params['A_k'],
            params['W_v'], params['A_v'],
        )

        def fn_v3_fwd():
            return lora_qkv_v3(
                X, params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
                W_all=W_all,
            )

        def fn_v4_fwd():
            return lora_qkv_v4(
                X, params['W_q'], params['A_q'], params['B_q'], params['s_q'],
                params['W_k'], params['A_k'], params['B_k'], params['s_k'],
                params['W_v'], params['A_v'], params['B_v'], params['s_v'],
                W_all=W_all,
            )

        ms_v3 = triton.testing.do_bench(fn_v3_fwd, warmup=WARMUP, rep=REP)
        ms_v4 = triton.testing.do_bench(fn_v4_fwd, warmup=WARMUP, rep=REP)

        results.append({"rank": rank, "v3_fwd_ms": round(ms_v3, 3), "v4_fwd_ms": round(ms_v4, 3)})
        print(f"  Rank={rank}: v3={ms_v3:.3f}ms  v4={ms_v4:.3f}ms")

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
    fwd_results = bench_v4_forward_only()
    bwd_results = bench_v4_backward()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Forward + Backward (fwd+bwd, bf16, LLaMA-3 8B GQA)")
    print("=" * 70)
    print(f"{'Rank':>4} | {'Unsloth':>9} | {'v3':>9} | {'v4':>9} | {'v4/Unsloth':>11} | {'v4/v3':>6}")
    print("-" * 60)
    for r in bwd_results:
        print(f"{r['rank']:>4} | {r['unsloth_ms']:>8.3f}ms | {r['v3_ms']:>8.3f}ms | "
              f"{r['v4_ms']:>8.3f}ms | {r['v4_vs_unsloth']:>10.2f}x | {r['v4_vs_v3']:>5.2f}x")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_csv(bwd_results, f"benchmarks/results/v4_{timestamp}.csv")

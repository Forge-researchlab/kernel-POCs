"""
Embedding Kernel Benchmarks
============================

Compare Forge embedding kernel variants against:
  - PyTorch nn.Embedding (baseline)
  - Liger Triton embedding kernel

Usage:
    python benchmarks/bench_embedding.py                   # default sweep
    python benchmarks/bench_embedding.py --vocab 32000     # specific vocab size
    python benchmarks/bench_embedding.py --save results/   # save CSV output
"""

import argparse
import csv
import itertools
import os
import time
from pathlib import Path

import torch
import triton

# ---------------------------------------------------------------------------
# Import kernels — adjust paths as your structure evolves
# ---------------------------------------------------------------------------
import sys

KERNEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v1.embedding_kernel_v1_upgrade_1 import ForgeEmbeddingFunction

try:
    from reference.liger.embedding_kernel import LigerEmbeddingFunction
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def pytorch_embedding_forward_backward(weight, indices):
    emb = torch.nn.functional.embedding(weight, indices)
    loss = emb.sum()
    loss.backward()
    return weight.grad


def forge_embedding_forward_backward(weight, indices):
    out = ForgeEmbeddingFunction.apply(weight, indices)
    loss = out.sum()
    loss.backward()
    return weight.grad


def liger_embedding_forward_backward(weight, indices):
    out = LigerEmbeddingFunction.apply(weight, indices)
    loss = out.sum()
    loss.backward()
    return weight.grad


def bench_fn(fn, weight, indices, warmup=5, rep=20):
    """Time a forward+backward pass in ms using CUDA events."""
    for _ in range(warmup):
        weight.grad = None
        fn(weight, indices)
        torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        weight.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(weight, indices)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    trimmed = times[2:-2] if len(times) > 4 else times
    return sum(trimmed) / len(trimmed)


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
DEFAULT_CONFIGS = {
    "vocab_sizes": [32_000, 64_000, 128_256],
    "embedding_dims": [768, 2048, 4096],
    "seq_lengths": [512, 2048, 8192],
    "dtypes": [torch.float32, torch.bfloat16],
    "duplicate_ratios": [1.0, 0.1],  # 1.0 = all unique, 0.1 = 10% unique
}


def run_sweep(configs, save_dir=None):
    results = []
    header = [
        "vocab_size", "embedding_dim", "seq_len", "dtype",
        "dup_ratio", "n_unique",
        "pytorch_ms", "forge_ms", "liger_ms",
        "forge_speedup", "liger_speedup",
    ]

    for vocab, dim, seq, dtype, dup_ratio in itertools.product(
        configs["vocab_sizes"],
        configs["embedding_dims"],
        configs["seq_lengths"],
        configs["dtypes"],
        configs["duplicate_ratios"],
    ):
        n_unique = max(1, int(seq * dup_ratio))
        unique_ids = torch.randint(0, vocab, (n_unique,), device="cuda")
        if n_unique < seq:
            repeat_ids = unique_ids[torch.randint(0, n_unique, (seq - n_unique,), device="cuda")]
            indices = torch.cat([unique_ids, repeat_ids])[torch.randperm(seq, device="cuda")]
        else:
            indices = unique_ids[:seq]

        weight = torch.randn(vocab, dim, device="cuda", dtype=dtype, requires_grad=True)

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"

        pt_ms = bench_fn(pytorch_embedding_forward_backward, weight, indices)
        forge_ms = bench_fn(forge_embedding_forward_backward, weight, indices)
        liger_ms = bench_fn(liger_embedding_forward_backward, weight, indices) if HAS_LIGER else float("nan")

        forge_speedup = pt_ms / forge_ms
        liger_speedup = pt_ms / liger_ms if HAS_LIGER else float("nan")

        row = {
            "vocab_size": vocab, "embedding_dim": dim, "seq_len": seq,
            "dtype": dtype_str, "dup_ratio": dup_ratio, "n_unique": n_unique,
            "pytorch_ms": f"{pt_ms:.3f}", "forge_ms": f"{forge_ms:.3f}",
            "liger_ms": f"{liger_ms:.3f}",
            "forge_speedup": f"{forge_speedup:.2f}x",
            "liger_speedup": f"{liger_speedup:.2f}x" if HAS_LIGER else "N/A",
        }
        results.append(row)

        print(f"V={vocab:>7} D={dim:>4} S={seq:>5} {dtype_str:>4} dup={dup_ratio:.1f} | "
              f"PT={pt_ms:7.3f}ms  Forge={forge_ms:7.3f}ms ({forge_speedup:.2f}x)"
              + (f"  Liger={liger_ms:7.3f}ms ({liger_speedup:.2f}x)" if HAS_LIGER else ""))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"bench_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(results)
        print(f"\nResults saved to {path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, nargs="+", default=None)
    parser.add_argument("--dim", type=int, nargs="+", default=None)
    parser.add_argument("--seq", type=int, nargs="+", default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    configs = dict(DEFAULT_CONFIGS)
    if args.vocab:
        configs["vocab_sizes"] = args.vocab
    if args.dim:
        configs["embedding_dims"] = args.dim
    if args.seq:
        configs["seq_lengths"] = args.seq

    run_sweep(configs, save_dir=args.save)

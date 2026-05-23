---
name: embedding-kernel-research
description: Guide research and development of Triton embedding kernels. Use when the user asks to create a new kernel version, optimize backward pass, benchmark, debug correctness, or explore new algorithmic approaches for GPU embedding operations.
---

# Embedding Kernel Research

## Goal

Build a Triton embedding kernel that outperforms both PyTorch's `nn.Embedding` and Liger's Triton implementation. The forward pass (table lookup / gather) is straightforward. The core research challenge is the **backward pass** — aggregating gradients for duplicate token indices efficiently on GPU.

## Project Layout

- `reference/liger/` — Liger's kernel (atomic_add baseline) + detailed visual explanation
- `experiments/v{N}/` — Versioned kernel experiments
- `benchmarks/bench_embedding.py` — Benchmark harness; saves CSV to `benchmarks/results/`
- `tests/test_embedding.py` — Correctness tests (forward, backward, gradcheck)
- `docs/` — Design notes and profiling data

## Research Context: What Has Been Done

### v1: Sort + Grouped Reduction
- **Forward**: autotuned 2D tiled gather kernel (functionally identical to Liger)
- **Backward**: sort indices on GPU → `unique_consecutive` to find groups → one Triton program per unique token reduces its group of grad_output rows
- **v1_upgrade_1**: for groups with >32 duplicates, splits the group across multiple sub-programs (cooperative two-phase reduction: phase 1 computes partial sums per chunk, phase 2 reduces partials into grad_weight)
- **Fallback**: for sequences < 256 elements, uses PyTorch `index_add_` (sort overhead not worth it)
- **File**: `experiments/v1/embedding_kernel_v1_upgrade_1.py`

### Known Issues with v1
- `torch.sort` + `unique_consecutive` cause host-device synchronization
- `counts.max().item()` is another sync point
- For mostly-unique workloads, the sort overhead may exceed the atomic contention it avoids
- `triton.next_power_of_2(max_group_size)` can over-allocate registers for the serial loop

## Research Axes to Explore

### Backward Strategy Alternatives
- **Atomic_add with reduced contention**: partition the embedding dim across programs so each handles a wide tile — fewer threads collide on the same row. Simplest approach, may be enough for moderate duplicate counts.
- **Histogram + deterministic scatter**: compute a histogram of indices to pre-allocate output slots, then scatter-add without sorting. Avoids sort overhead entirely.
- **Warp-level reduction**: use warp shuffle intrinsics (`tl.reduce` / warp primitives) to reduce duplicates within a warp before writing. Reduces atomic pressure without sorting.
- **Shared memory accumulation**: for small embedding dims (<=256), accumulate in shared memory (one tile per group), then write once. Trades SRAM for global memory bandwidth.
- **Hybrid**: use different strategies depending on runtime properties (duplicate count, seq length). v1 already does this (index_add_ vs sort-based vs cooperative).

### Forward Optimizations
- The forward kernel is memory-bound and likely already near peak bandwidth
- Possible: fused embedding + dropout, fused embedding + layer norm
- Investigate: does autotuning actually pick good configs, or is manual tuning better?

### Systems-Level
- Eliminate host-device sync in backward (avoid `.item()`, use device-side branching)
- Reduce memory allocation overhead (pre-allocate scratch buffers, use memory pools)
- Profile actual kernel time vs orchestration overhead using `torch.profiler` or Nsight

## Benchmarking

```bash
cd kernels/embedding
python benchmarks/bench_embedding.py --save benchmarks/results/
python benchmarks/bench_embedding.py --vocab 128256 --dim 4096 --seq 2048  # LLaMA-3 scale
```

Always test with both duplicate ratios: 1.0 (all unique) and 0.1 (heavy dups).

## Testing

```bash
pytest tests/test_embedding.py -v
pytest tests/test_embedding.py -v -k "gradcheck"
```

## Performance Analysis

When analyzing results:
1. **Memory-bound or compute-bound?** Embedding is memory-bound — goal is saturating bandwidth
2. **Occupancy**: enough programs to fill the GPU? (n_groups in backward = parallelism)
3. **Time breakdown**: how much is sort/unique vs actual kernel vs alloc?
4. **Effective bandwidth**: compute GB/s transferred vs device peak

## Common Pitfalls

- Accumulating in bf16 instead of fp32 → gradient drift under heavy duplication
- `torch.sort` and `.item()` trigger host-device synchronization
- `triton.next_power_of_2` on large group sizes → excessive register usage / spilling
- Forgetting `stable=True` in sort → nondeterministic reordering of equal elements
- Allocating scratch buffers inside backward on every call → allocation overhead dominates

## Reference

- `reference/liger/explanation.md` — visual walkthrough of embedding forward/backward, atomic_add race conditions, gradient flow
- `reference/liger/embedding_kernel.py` — Liger's kernel (atomic_add backward)

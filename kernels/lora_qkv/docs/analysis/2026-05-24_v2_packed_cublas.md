# Analysis: v2 Packed cuBLAS + Triton Epilogue — Beats Unsloth

**Date**: 2026-05-24
**Kernel version**: v2
**GPU**: NVIDIA A100-SXM4-80GB
**Compared against**: cuBLAS bare, PyTorch naive, Unsloth (addmm\_), Triton v1

## Summary

v2 **beats Unsloth by 4-11%** on full QKV and **beats PyTorch by 16-23%**. The packed W+A approach (single cuBLAS call per projection that reads X once) combined with a Triton LoRA epilogue eliminates the redundant X read and the XA intermediate HBM round-trip. v2 is also **rank-independent** — performance barely changes from r=8 to r=64 (1.927ms → 1.924ms).

## Key Innovation

Instead of Unsloth's 3 calls per projection (X@W, X@A, addmm\_), v2 does:
1. Pack W and A: `W_packed = cat([W, A], dim=0)` → `[N+r, K]`
2. One cuBLAS call: `out_packed = X @ W_packed^T` → reads X ONCE
3. Split + Triton epilogue: `Y = base_out + s * XA @ B^T`

This halves the X HBM reads per projection (1x vs 2x) and replaces the addmm\_ with a cheaper Triton epilogue (the tiny `XA @ B` dot happens in registers).

## Performance Data

### Full QKV (batch=4, seq=2048, hidden=4096, GQA 32q/8kv, bf16, A100)

| Rank | PyTorch (9 ops) | Unsloth (9 cuBLAS) | v1 (3 Triton) | **v2 (3 cuBLAS + 3 Triton)** | v2/Unsloth | v2/PyTorch |
|------|----------------|-------------------|--------------|------------------------------|------------|------------|
| 8    | 2.240ms        | 2.014ms           | 2.921ms      | **1.927ms**                  | **1.04x**  | 1.16x      |
| 16   | 2.285ms        | 2.035ms           | 3.624ms      | **1.869ms**                  | **1.09x**  | 1.22x      |
| 32   | 2.282ms        | 2.057ms           | 3.824ms      | **1.856ms**                  | **1.11x**  | 1.23x      |
| 64   | 2.297ms        | 2.044ms           | 4.195ms      | **1.924ms**                  | **1.06x**  | 1.19x      |

### Per-Projection K/V GQA (M=8192, N=1024, K=4096, bf16)

| Rank | cuBLAS bare | Unsloth | v1      | **v2**     | v2/Unsloth |
|------|------------|---------|---------|-----------|------------|
| 8    | 0.329ms    | 0.389ms | 0.483ms | **0.342ms** | **1.14x** |
| 16   | 0.314ms    | 0.388ms | 0.488ms | **0.337ms** | **1.15x** |
| 32   | 0.318ms    | 0.391ms | 0.503ms | **0.336ms** | **1.16x** |
| 64   | 0.317ms    | 0.393ms | 0.603ms | **0.343ms** | **1.15x** |

## Why v2 Wins

1. **X read halved**: cuBLAS packed matmul reads X once per projection (3 total), vs Unsloth's 6 reads
2. **No XA intermediate**: Unsloth writes XA to HBM then reads it back for addmm\_. v2 gets XA from the packed output (already in memory) and processes it in the Triton epilogue
3. **Rank-independent**: v2 performance barely changes with rank (1.856–1.927ms) because the Triton epilogue's tiny `[r, BLOCK_N]` dot is negligible. Unsloth and v1 both degrade at high ranks
4. **cuBLAS at full speed**: The packed matmul is only 0.4% larger (4096+16=4112 output dims), so cuBLAS runs at essentially the same speed as the bare matmul

## Cross-Version Comparison (Full QKV, rank=16, bf16, A100)

| Kernel | Approach | Time (ms) | vs Unsloth | Launches | X reads |
|--------|----------|-----------|------------|----------|---------|
| cuBLAS bare (no LoRA) | 3× torch.matmul | ~1.7ms | — | 3 | 3× |
| **Unsloth** | 3× (matmul + XA + addmm\_) | 2.035ms | 1.00x | 9 | 6× |
| PyTorch naive | 3× (matmul + XA + add) | 2.285ms | 0.89x | 9+ | 6× |
| Triton v1 | 3× fused Triton | 3.624ms | 0.56x | 3 | 6× |
| **v2** | **3× packed cuBLAS + Triton epilogue** | **1.869ms** | **1.09x** | **6** | **3×** |

## Memory Performance

| Kernel | Peak Forward Delta (MB) | vs Unsloth | Notes |
|--------|------------------------|------------|-------|
| PyTorch naive | 192 MB | 0.50x (2x worse) | All intermediates separate |
| **Unsloth** | **96 MB** | **1.00x** | addmm\_ in-place, minimal temps |
| v1 | 96 MB | 1.00x | Output only, LoRA in registers |
| **v2** | **128 MB** | **0.75x (33% more)** | Packed output + final output coexist |

### Why v2 Uses More Memory

v2's packed matmul creates `[M, N+r]` output (e.g., `[8192, 4112]` = 64.5 MB for Q). The Triton epilogue then writes the final `[M, N]` output (64 MB). During the epilogue, BOTH tensors exist in GPU memory simultaneously — an extra ~32 MB total across all 3 projections.

Unsloth's `addmm_` avoids this because it writes the LoRA result in-place into the base output — no extra allocation needed.

### Tradeoff Assessment

| Metric | v2 vs Unsloth | Verdict |
|--------|---------------|---------|
| Time | **1.09x faster** (9% gain) | Win |
| Memory | **0.75x** (33% more) | Acceptable at LLaMA-8B scale |

At 128 MB vs 96 MB, the memory increase is 32 MB — small relative to the total model memory (~5-16 GB for 7B-13B models). This is an acceptable tradeoff for 9% speed improvement.

For memory-constrained scenarios, v2_2 could use an in-place epilogue (write directly into the packed output's first N columns, avoiding the extra allocation).

## Next Steps

1. **v2_2**: In-place epilogue to eliminate the extra 32 MB allocation — match Unsloth's memory while keeping the speed gain
2. **v2_3**: Fully-packed QKV (1 cuBLAS for all projections, X read once total) — could push to 1.15-1.20x Unsloth
3. **v3**: Wrap in `autograd.Function` with backward pass for training
4. **v3 backward**: The backward has similar structure — packed W+A could help there too

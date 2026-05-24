# Performance Analysis — 2026-05-23 (Updated)

## Bottleneck: 64% from base Triton matmul being 0.73x cuBLAS, 36% from LoRA K-loop overhead

---

## Step 7: Cross-Version Comparison

### Per-projection (M=8192, N=14336, K=4096, r=16, bf16)

| Kernel | Time (ms) | vs cuBLAS | vs Unsloth | Launches |
|--------|-----------|-----------|------------|----------|
| cuBLAS bare | 4.236 | 1.00x | — | 1 |
| Unsloth matmul_lora | 4.352 | 0.97x | 1.00x | 3 |
| **v1** fused LoRA | 6.763 | 0.63x | 0.64x | 1 |
| **v1** base only (no LoRA) | 5.779 | 0.73x | — | 1 |

### Per-projection — down (M=8192, N=4096, K=14336, r=16, bf16)

| Kernel | Time (ms) | vs cuBLAS | vs Unsloth | LoRA OH |
|--------|-----------|-----------|------------|---------|
| cuBLAS bare | 3.842 | 1.00x | — | — |
| Unsloth matmul_lora | 4.014 | 0.96x | 1.00x | — |
| **v1** base only | 5.218 | 0.73x | — | — |
| **v1** fused LoRA | 6.605 | 0.58x | 0.61x | 27% |

### Full MLP (batch=4, seq=2048, H=4096, I=14336, r=16, bf16)

| Kernel | Time (ms) | vs Unsloth | Launches |
|--------|-----------|------------|----------|
| Unsloth LoRA_MLP | 12.714 | 1.00x | 10 |
| **v1** (3× fused + SwiGLU) | 20.439 | 0.62x | 4 |

### Rank Scaling (gate/up, M=8192, N=14336, K=4096, bf16)

| Rank | Unsloth (ms) | v1 (ms) | Speedup | LoRA overhead |
|------|-------------|---------|---------|---------------|
| 8 | 4.255 | 6.750 | 0.63x | 17% |
| 16 | 4.197 | 6.784 | 0.62x | 17% |
| 32 | 4.203 | 7.526 | 0.56x | 30% |
| 64 | 4.283 | 9.542 | 0.45x | 65% |

---

## Step 5: Gap Decomposition

```
Total gap (gate/up, r=16): 2.412ms
├── Base matmul gap: 1.543ms (64%)   ← Triton matmul 0.73x cuBLAS
└── LoRA overhead gap: 0.869ms (36%) ← extra tl.dot + A loads in K-loop
```

### Unsloth Operation Breakdown

| cuBLAS Op | Time (ms) | Notes |
|-----------|-----------|-------|
| X @ W^T | 4.236 | Main matmul, highly tuned |
| X @ A^T | 0.069 | Skinny GEMM, specialized kernel |
| addmm_(XA, B^T) | 0.335 | Skinny GEMM + fused add |
| Sum of parts | 4.640 | — |
| Actual Unsloth | 4.352 | Saves 0.29ms from GPU pipelining overlap |

---

## Root Cause 1: Base Matmul is 0.73x cuBLAS (64% of gap)

Our v1 Triton tiled matmul achieves only 73% of cuBLAS throughput even WITHOUT LoRA.

**Why cuBLAS is faster:**

1. **No L2 swizzle**: Our `pid_m = pid // num_n_blocks` is a naive linear mapping. Adjacent thread blocks access unrelated W rows, causing L2 cache thrashing. cuBLAS groups blocks to share W data in L2. Expected fix: +10-15%.

2. **Weak autotune**: Only 6 tile configs tried. Missing the likely sweet spot (BLOCK_M=128, BLOCK_N=256, BLOCK_K=64 for this shape). cuBLAS tests hundreds of precomputed configs. Expected fix: +5-10%.

3. **Shallow pipeline**: max `num_stages=4`. cuBLAS uses 4-5 stages with architecture-specific tuning.

## Root Cause 2: LoRA K-loop Overhead (36% of gap)

Adding `tl.dot(x_tile, A_tile^T)` inside every K iteration costs 17% at r=16, scaling to 65% at r=64.

**Why it's expensive:**

1. **A tile loaded every K step**: `[BLOCK_R, BLOCK_K]` = 1 KB per step × 128 steps = 128 KB total. Same A data needed by all N-blocks = massive L2 contention.

2. **Register pressure**: `xa` accumulator = `[BLOCK_M, BLOCK_R]` fp32 = 8 KB (r=16) or 32 KB (r=64). At r=64 this forces register spills.

3. **Tensor core competition**: Extra `tl.dot` per K step consumes pipeline slots even though the output is tiny (16 cols vs 64-128 for W).

**Contrast**: Unsloth's separate `X @ A^T` cuBLAS call takes 0.069ms total using a specialized skinny-GEMM kernel.

---

## Recommended Next Steps (priority order)

### 1. v1_upgrade_1: L2 Swizzle + Expanded Autotune

Add GROUP_SIZE_M swizzle and 4+ new tile configs. Expected: base matmul 0.73x -> ~0.85-0.90x cuBLAS. This benefits all future versions.

### 2. v2: Gate + Up + SwiGLU Fusion

Even at 0.73x cuBLAS per-tile, v2 eliminates structural overhead that cuBLAS cannot:
- X read once instead of 4× (saves ~2 × 8192 × 4096 × 2 = 128 MB HBM reads)
- `e` and `g` never materialized (saves ~2 × 8192 × 14336 × 2 = 448 MB HBM writes + reads)
- 7 launches -> 1 launch

Estimated HBM savings at LLaMA-8B: **~700 MB per MLP forward**. At 2 TB/s bandwidth that's ~0.35ms saved, potentially closing the per-tile gap.

### 3. For r >= 32: Consider hybrid approach

Use cuBLAS for the base matmul + Unsloth's addmm_ for the skinny LoRA, but fuse at the MLP level (gate+up share X read). Best of both worlds for high-rank configs.

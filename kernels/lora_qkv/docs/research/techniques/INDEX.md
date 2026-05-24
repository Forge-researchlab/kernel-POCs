# Optimization Techniques Catalog

> Catalog of optimization techniques discovered through research.
> Status tracks whether we've tried each technique in our kernel.

| Technique | Source | Status | Expected Benefit | Notes |
|-----------|--------|--------|------------------|-------|
| All-Triton fused LoRA matmul | v1 design | **tried → rejected** | 1 launch per proj, X read 2× | 0.6x cuBLAS — Triton matmul can't compete |
| Packed W+A cuBLAS + epilogue | v2 design | **adopted** | 1 cuBLAS + 1 Triton per proj, X read 1× | **1.09x Unsloth** — current best |
| L2 cache swizzle (GROUP_SIZE_M) | Triton tutorial | **tried** (v1) | +10-15% matmul throughput | Helped but not enough to beat cuBLAS |
| In-place epilogue | analysis | untried | Fix 33% memory overhead in v2 | Low effort, high impact on memory |
| Fuse 3 epilogues into 1 | analysis | untried | -1 launch, ~0.05ms | Medium effort |
| Fully packed QKV (1 cuBLAS) | qkv_fusion, vLLM | untried | X read 1× total → ~1.2x Unsloth | Medium-high effort, needs shape validation |
| autograd.Function wrapper | Unsloth pattern | untried | Enable training (backward) | Required for production |
| Packed backward (W+A in bwd) | analysis | untried | Same trick for backward pass | Depends on v3 |
| addmm_ avoidance | PyTorch #141210 | **adopted** (implicitly) | Avoids suboptimal cuBLASLt selection | Validated by research |
| GEMM-epilogue pipelining | CODA paper | untried | Epilogue overlaps with GEMM mainloop | Requires Hopper/Blackwell, CUTLASS |
| Triton grouped GEMM | Triton tutorial | untried | Handle Q/K/V different N in 1 launch | Unlikely to beat cuBLAS on A100 |
| Persistent matmul | Triton tutorial | untried | Better SM utilization for large M | Complex, defer |

## Tried & Adopted

1. **Packed W+A cuBLAS + Triton epilogue** (v2) — Packs weight W and LoRA A into one matrix, does single cuBLAS call per projection. Triton epilogue adds the LoRA B term. Achieves 1.09x Unsloth. Rank-independent.
2. **addmm_ avoidance** — Our Triton epilogue replaces addmm_ with a custom kernel. Research confirms addmm_ is not always optimal.

## Tried & Rejected

1. **All-Triton fused LoRA matmul** (v1) — Triton's tiled matmul achieves only 0.6x cuBLAS on A100. The base matmul penalty dominates any fusion savings. Lesson: use cuBLAS for matmuls, Triton only for what cuBLAS can't do.

## Untried — High Priority

1. **In-place epilogue** (v2_2) — Write LoRA result into packed output buffer in-place. Eliminates 32 MB extra memory. Low effort, pure win.
2. **Fully packed QKV** (v2_3) — One cuBLAS call for all projections. X read once. Expected ~1.2x Unsloth.
3. **autograd.Function** (v3) — Required for training support.

## Untried — Low Priority

1. **Fuse 3 epilogues into 1** — Minor launch overhead savings (~0.05ms)
2. **GEMM-epilogue pipelining** (CODA) — Requires Hopper/CUTLASS, future work
3. **Persistent matmul** — Complex, marginal benefit over cuBLAS
4. **Triton grouped GEMM** — Unlikely to beat cuBLAS on A100

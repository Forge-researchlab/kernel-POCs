# Optimization Techniques Catalog

> Catalog of optimization techniques discovered through research.
> Status tracks whether we've tried each technique in our kernel.

| Technique | Source | Status | Expected Benefit | Notes |
|-----------|--------|--------|------------------|-------|
| Fuse LoRA into base matmul | Project design (Axis A) | untried | Eliminate 2 launches + 1 HBM round-trip per projection | Core v1 approach |
| Fuse Q+K+V projections | Project design (Axis B) | untried | Reduce X reads from 6x to 1x | Core v2 approach |
| cuBLAS + Triton epilogue | lora_mlp lessons | untried | Use cuBLAS for base matmul, Triton only for LoRA add | Avoids competing with cuBLAS at matmul |
| Register-level LoRA (r≤16) | Project design | untried | Zero SRAM overhead for LoRA path | Sweet spot for most configs |
| SRAM ping-pong (r=32-64) | Project design | untried | Keep LoRA intermediates in shared memory | For higher-rank configs |
| In-place addmm_ | Unsloth pattern | untried | Fused add+GEMM in single cuBLAS call | Baseline optimization |

## Tried & Adopted

{No techniques adopted yet — kernel development has not started}

## Tried & Rejected

{No techniques rejected yet}

## Untried — High Priority

1. **Fuse LoRA into base matmul** — core v1 approach, eliminates intermediate writes
2. **Fuse Q+K+V projections** — core v2 approach, eliminates redundant X reads
3. **cuBLAS + Triton epilogue** — proven pattern from lora_mlp project

## Untried — Low Priority

1. **Software pipelining** (num_stages > 1) — helps if memory-bound
2. **L2 cache swizzle** — helps with large grid dimensions
3. **Warp specialization** — advanced Triton pattern, may not be supported yet

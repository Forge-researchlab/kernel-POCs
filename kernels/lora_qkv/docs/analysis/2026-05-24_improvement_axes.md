# Improvement Analysis: Where We Are and What's Possible

**Date**: 2026-05-24
**Current best**: v2 (1.869ms, 1.09x Unsloth, 128 MB memory)
**GPU**: NVIDIA A100-SXM4-80GB

## Current State Summary

| Metric | v2 | vs Unsloth | Verdict |
|--------|-----|-----------|---------|
| Time (rank=16) | 1.869ms | **1.09x faster** | Good — already beating SOTA |
| Memory (delta) | 128 MB | **0.75x (33% worse)** | Fixable — see improvement #1 |
| Launches | 6 (3 cuBLAS + 3 Triton) | vs 9 | Reducible to 2 — see #3 |
| X HBM reads | 3× | vs 6× | Reducible to 1× — see #3 |
| Rank sensitivity | None (1.856–1.927ms) | vs Unsloth degrades at r=64 | Excellent |

---

## Improvement #1: In-Place Epilogue (v2_2)

**Impact**: Fix the 33% memory overhead. Match Unsloth's 96 MB while keeping 1.09x speed.
**Effort**: Low (1-2 hour change)
**Risk**: None

**Problem**: v2 allocates a new output tensor `Y` in the Triton epilogue while the packed output `[M, N+r]` still exists. Both coexist → 32 MB extra.

**Solution**: Write the LoRA result in-place into `packed_output[:, :N]`. The epilogue reads `packed_output[:, N:]` (the XA columns) and `B`, computes `XA @ B^T`, adds it to `packed_output[:, :N]` in place. The last `r` columns become garbage (fine — we don't need them).

```python
# Current v2 (allocates new Y):
Y = lora_epilogue(base_out, xa, B, lora_scale)

# v2_2 (in-place, no extra allocation):
lora_epilogue_inplace(packed_output, N, R, B, lora_scale)  # writes into packed_output[:, :N]
Y = packed_output[:, :N]  # view, no copy
```

**Expected result**: Same speed (1.09x Unsloth), memory drops from 128 MB to ~96 MB.

---

## Improvement #2: Fuse 3 Epilogues into 1 Kernel (v2_2 or v2_3)

**Impact**: Reduce Triton launches from 3 to 1. Minor latency savings (~0.05ms).
**Effort**: Medium
**Risk**: Low

**Problem**: Currently 3 separate Triton epilogue launches (one per Q, K, V). Each launch has Python dispatch overhead.

**Solution**: One Triton kernel that processes all 3 projections. Since Q has a different output dim than K/V (GQA), the kernel needs to handle different N per projection. Use a loop over 3 groups or a grouped-GEMM-style tile dispatch.

**Expected result**: 5 total launches (3 cuBLAS + 1 Triton) instead of 6.

---

## Improvement #3: Fully Packed QKV — Single cuBLAS Call (v2_3)

**Impact**: Reduce X reads from 3× to **1×**. Potentially 1.15–1.20x Unsloth.
**Effort**: Medium-High
**Risk**: Medium (cuBLAS may select different algorithm for very wide output)

**Problem**: v2 still does 3 separate cuBLAS calls, each reading X once (3 total). X is 64 MB — reading it 3 times = 192 MB bandwidth.

**Solution**: Pack ALL projection weights into one matrix:
```
W_all = cat([W_q, A_q, W_k, A_k, W_v, A_v], dim=0)
# Shape: [H_q + r + H_kv + r + H_kv + r, K] = [6192, 4096] for LLaMA-8B
```

One cuBLAS call: `out_all = X @ W_all^T` → `[M, 6192]`. Then one Triton epilogue splits and applies LoRA to all 3 projections.

**Validation**: The `qkv_fusion` project (GitHub: hilaryKChen/qkv_fusion) achieves 2x speedup over 3× nn.Linear using packed QKV without LoRA. vLLM does 6-way packed projections for Qwen3.5 (non-LoRA path).

**Concern from research**: One MoE developer noted that "concatenating gate+up into one larger N GEMM was actually slower because cuBLAS selects a different autotune config for the wider output." Need to benchmark to verify this doesn't happen at our shapes.

**Expected result**: 2 total launches (1 cuBLAS + 1 Triton), X read once. If cuBLAS shape penalty < 5%, total time ~1.7ms → 1.20x Unsloth.

---

## Improvement #4: `addmm_` is Not Always Faster (Research Insight)

**Impact**: Validates that v2's approach (separate matmul + Triton epilogue) is correct.
**Source**: PyTorch issue #141210

**Key finding**: For large shapes like `[32768, 12288] @ [12288, 36864]`, PyTorch's `addmm` is **1.5x slower** than separate `mm + add` because cuBLASLt selects a suboptimal kernel for the fused operation.

**Implication**: Unsloth's `addmm_` pattern may not be optimal for all shapes. Our v2 approach (cuBLAS matmul + Triton epilogue) avoids this pitfall because:
1. The cuBLAS matmul is a standard GEMM (not addmm), so cuBLAS picks the best algorithm
2. The Triton epilogue does the add+LoRA as a separate bandwidth-bound pass

---

## Improvement #5: CODA-Style GEMM-Epilogue Programs (Research Insight)

**Impact**: Theoretical framework for what we're doing. Validates our architecture.
**Source**: CODA paper (arxiv 2605.19269, May 2026)

**Key insight**: "Many Transformer computations can be algebraically reparameterized as GEMM-plus-epilogue programs. The epilogue operates on data already produced by the GEMM tile, avoiding additional global-memory round trips."

This is exactly what v2 does. CODA formalizes it and shows the epilogue can be placed "in the shadow of other tiles' mainloops" on Hopper (pipelined execution). Future work: integrate with CUTLASS/CODA's epilogue framework for even better pipelining on H100.

---

## Improvement #6: autograd.Function Wrapper (v3)

**Impact**: Enable training (backward pass). Required for production use.
**Effort**: High (backward pass is complex — 12+ gradient computations)
**Risk**: Medium (backward math must be exact)

**What's needed**:
- Forward: v2 (or v2_2/v2_3) kernel
- Backward: compute dX (accumulated from all 3 projections) + 6 LoRA gradients (dA_q, dB_q, etc.)
- Backward can also use the packed W+A trick for the dX computation

**Pattern from Unsloth's backward** (to match/beat):
- Pre-allocate gradient tensors with `torch.empty_like`
- Use `addmm_` with `alpha=s, beta=0` for LoRA gradients
- Accumulate dX in-place: `dX = dQ @ W_q; dX.addmm_(dK, W_k); dX.addmm_(dV, W_v)`
- 12+ cuBLAS calls in backward

---

## Improvement #7: Triton Grouped GEMM for Asymmetric QKV (Stretch)

**Impact**: Handle GQA in a single kernel dispatch.
**Source**: Triton tutorial 08-grouped-gemm

**Idea**: Instead of 3 separate cuBLAS calls (Q: N=4096, K: N=1024, V: N=1024), use Triton's grouped GEMM to handle all 3 in one launch with different N per group. This would give us 1 Triton launch for all 3 base matmuls.

**BUT**: Triton grouped GEMM is less optimized than cuBLAS for individual GEMMs. From our v1 experience, Triton matmul is ~0.6x cuBLAS. Grouped GEMM would need to be at least 0.33x cuBLAS (since it does 3 GEMMs in 1 launch) to break even with 3× cuBLAS calls.

**Verdict**: Unlikely to help on A100. May become viable on Hopper/Blackwell with TMA support.

---

## Priority Ranking

| # | Improvement | Expected Gain | Memory | Effort | Priority |
|---|------------|---------------|--------|--------|----------|
| 1 | In-place epilogue (v2_2) | Same speed | **-32 MB** (match Unsloth) | Low | **Do first** |
| 2 | Fuse 3 epilogues (v2_2) | -0.05ms | Neutral | Medium | **Do with #1** |
| 3 | Fully packed QKV (v2_3) | +5-15% speed | Neutral | Medium-High | **Try next** |
| 6 | autograd.Function (v3) | Enables training | N/A | High | **Required** |
| 4 | (insight) addmm_ not always faster | Validates v2 | — | None | Documented |
| 5 | (insight) CODA framework | Validates architecture | — | None | Documented |
| 7 | Grouped GEMM | Unlikely gain | Neutral | High | **Defer** |

---

## Current vs Theoretical Best

| Metric | v2 | After #1+#2 | After #3 | Theoretical limit |
|--------|-----|------------|----------|-------------------|
| Time (r=16) | 1.869ms | ~1.85ms | ~1.65ms | ~1.55ms (cuBLAS bare + epilogue) |
| Memory | 128 MB | ~96 MB | ~96 MB | 96 MB |
| Launches | 6 | 4 | 2 | 2 |
| X reads | 3× | 3× | 1× | 1× |
| vs Unsloth | 1.09x | ~1.10x | ~1.23x | ~1.31x |

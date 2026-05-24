# v4: Packed Backward Pass Analysis

**Date:** 2026-05-24  
**GPU:** NVIDIA A100-SXM4-80GB  
**Config:** batch=4, seq=2048, hidden=4096, GQA 32/8, bf16

## What was done

1. Created `experiments/v4/lora_qkv_kernel_v4.py` with packed backward pass
2. Implemented `_lora_dx_epilogue_kernel` Triton kernel for fused dX LoRA contributions
3. Ran full test suite (18 tests: gradcheck, forward, backward, packing helpers)
4. Benchmarked v4 vs v3 vs Unsloth at ranks 8, 16, 32, 64

## What was learned

### Operation Count Reduction

| Operation | v3/Unsloth | v4 | Savings |
|-----------|-----------|-----|---------|
| dX_base (dY @ W) | 3 cuBLAS | 1 packed cuBLAS | -2 launches |
| XA (X @ A^T) | 3 cuBLAS | 1 packed cuBLAS | -2 launches |
| dY @ B (skinny) | 3 cuBLAS | 3 cuBLAS | 0 (can't pack: different N) |
| dA (dY_B^T @ X) | 3 cuBLAS | 1 packed cuBLAS | -2 launches |
| dB (dY^T @ XA) | 3 cuBLAS | 3 cuBLAS | 0 (can't pack: different N) |
| dX LoRA additions | 3 addmm_ | 1 Triton | -2 launches |
| **Total** | **18+ ops** | **10 ops** | **-8 launches** |

### Why packing works for some ops but not others

- **Can pack:** Operations where all 3 projections share the same "other" dimension.
  - `dX_base`: All 3 results sum into the same [M, K] output → pack dQKV column-wise
  - `XA`: All 3 use the same X [M, K] → pack A matrices row-wise
  - `dA`: All 3 use the same X [M, K] → pack dY_B column-wise

- **Cannot pack:** Operations where output dimensions differ.
  - `dY @ B`: dQ is [M, H_q], dK/dV are [M, H_kv] — different source widths
  - `dB`: Results are [H_q, r] vs [H_kv, r] — different output heights

### Triton epilogue design

The backward epilogue is structurally different from the forward epilogue:
- **Forward:** Reads base_out + XA, computes 1× `XA @ B^T`, adds
- **Backward:** Reads dX_base + 3× (dY_B, A), computes 3× `dY_B @ A`, adds all

Each `dY_B @ A` is [BLOCK_M, R] @ [R, BLOCK_K] — same tiny matmul pattern, just 3 of them in one kernel. This avoids 3 separate dX reads + 3 writes (saves 6× [M, K] HBM transfers).

## What changed

### Performance (fwd+bwd, bf16, LLaMA-3 8B GQA)

| Rank | Unsloth | v3 | v4 | v4/Unsloth | v4/v3 |
|------|---------|-----|-----|------------|-------|
| 8 | 5.905ms | 4.641ms | 4.323ms | **1.37x** | 1.07x |
| 16 | 5.058ms | 4.696ms | 4.227ms | **1.20x** | 1.11x |
| 32 | 5.011ms | 4.649ms | 4.283ms | **1.17x** | 1.09x |
| 64 | 5.071ms | 4.690ms | 4.438ms | **1.14x** | 1.06x |

### Performance breakdown

- **v4 backward improvement over v3:** 6-11% (from packing 8 launches into 3)
- **v4 total vs Unsloth:** 14-37% faster (forward packing + backward packing)
- **Rank sensitivity:** Improvement decreases with rank because the Triton epilogue's tiny matmuls grow in register cost at higher R

### Correctness

- fp64 gradcheck: PASSED (MHA, GQA, 3D inputs, ranks 4/8/16)
- Forward: exact match with v2_3 (bit-identical)
- dA/dB: exact match with reference (same operations)
- dX: 1.56e-02 max diff at bf16 (due to operation reordering; correct per gradcheck)

## What's next

1. **Autotune the backward epilogue**: current BLOCK_M=64, BLOCK_K=64 are hardcoded; autotuning may find better configs for different ranks
2. **Explore packing dB**: Could pad K/V results to H_q and pack all 3 dB computations (trade memory for launches)
3. **Profile with ncu**: Identify if the packed cuBLAS calls hit different algorithm selection compared to separate calls
4. **Memory optimization**: Investigate whether W_dX_packed can share storage with W_all to reduce memory overhead

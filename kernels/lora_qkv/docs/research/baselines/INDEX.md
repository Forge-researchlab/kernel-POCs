# Known Baselines

> Catalog of known baseline implementations and their characteristics.

| Baseline | Source | Approach | Launches (fwd) | X HBM Reads | Strengths | Weaknesses |
|----------|--------|----------|-----------------|-------------|-----------|------------|
| PyTorch naive | `reference/lora_qkv_pytorch.py` | Separate matmuls + manual LoRA via `out + s*(XA @ B.t())` | 9+ | 6 | Simple, correct | Slow: no fusion, extra alloc for LoRA intermediates |
| Unsloth matmul_lora × 3 | `reference/unsloth_baseline.py` | Per-projection cuBLAS: `X@W`, `X@A`, `addmm_(XA, B)` | 9 | 6 | Fast cuBLAS matmuls, in-place addmm_ | No cross-projection fusion, X read 6x |
| Packed QKV (no LoRA) | N/A | Single `X @ W_qkv.t()` then split | 1 | 1 | Fewest launches, cuBLAS lower bound | Incompatible with per-projection LoRA |

## Baseline Details

### PyTorch Naive

The simplest correct implementation. Each projection is a separate `X @ W.t()` followed
by `+ s * (X @ A.t()) @ B.t()`. This allocates temporary tensors for the LoRA intermediate
`X @ A.t()` and the LoRA output `(X@A) @ B.t()`. Represents the worst case for launch
overhead and memory traffic.

### Unsloth matmul_lora × 3

The current open-source SOTA. Uses `addmm_` for the fused add+GEMM step, which is faster
than allocating a temporary for the LoRA output and adding it separately. Still makes 3
cuBLAS calls per projection (9 total) and reads X from HBM 6 times.

### Packed QKV (no LoRA)

Theoretical lower bound for the base matmul. Packs W_q, W_k, W_v into a single
`[H_q + 2*H_kv, H]` weight and does one cuBLAS call. Not compatible with per-projection
LoRA (each projection has separate A/B matrices), but useful as a reference for how fast
the base matmul could be.

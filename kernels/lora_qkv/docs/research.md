# LoRA QKV Fused Kernel — Research Notes

## Background

### What is LoRA?

Low-Rank Adaptation (LoRA) adds trainable low-rank matrices `A` (r×d) and `B` (out×r) alongside a frozen weight `W` (out×d). The forward pass computes `y = x @ W^T + s · (x @ A^T) @ B^T`. Because r << min(d, out), the LoRA path has far fewer FLOPs than the base matmul, but in an unfused implementation it still reads/writes intermediate activations from HBM.

### What is QKV in Attention?

Multi-head attention projects the input `x` into queries (Q), keys (K), and values (V):

```
x: [B*S, H]   (batch*seq_len × hidden_dim)

Q = x @ W_q^T   → [B*S, H]  (reshaped to [B, num_heads, S, head_dim])
K = x @ W_k^T   → [B*S, H]  (reshaped similarly)
V = x @ W_v^T   → [B*S, H]  (reshaped similarly)
```

Where `H = num_heads × head_dim`. For LLaMA-3 8B: H=4096, num_heads=32, head_dim=128.

With LoRA applied to all three projections:

```
Q = x @ W_q^T + s_q · (x @ A_q^T) @ B_q^T
K = x @ W_k^T + s_k · (x @ A_k^T) @ B_k^T
V = x @ W_v^T + s_v · (x @ A_v^T) @ B_v^T
```

Where:
- `W_q, W_k, W_v`: frozen base weights, each [H, H] (or [H_kv, H] for GQA)
- `A_q, A_k, A_v`: LoRA down-projections, each [r, H]
- `B_q, B_k, B_v`: LoRA up-projections, each [H, r] (or [H_kv, r])
- `s_q, s_k, s_v`: LoRA scaling factors (alpha/r)

### Multi-Head vs Grouped-Query Attention (GQA)

In standard MHA, Q/K/V all have the same output dimension H.
In GQA (LLaMA-3), K and V have fewer heads:
- Q: [B*S, H] (num_heads × head_dim = 32 × 128 = 4096)
- K: [B*S, H_kv] (num_kv_heads × head_dim = 8 × 128 = 1024)
- V: [B*S, H_kv] (same as K)

This means W_k, W_v are [H_kv, H] — smaller than W_q.

### Why Fuse?

1. **Memory**: LoRA intermediates `x @ A` (shape [B*S, r]) are small and fit in SRAM for typical ranks. Fusing avoids materializing them to HBM.
2. **Kernel launch overhead**: With LoRA, the naive approach is 9 kernel launches (3 base matmuls + 6 LoRA matmuls). Fusing reduces this.
3. **Input reuse**: All three projections (Q, K, V) read the **same input `x`**. Fusing allows loading x once from HBM and computing all three projections + LoRA terms.
4. **HBM bandwidth**: at production scales, the input tensor `x` is large ([8192, 4096] = 64 MB in bf16). Reading it 6 times (3 W + 3 A) vs 1 time is significant.

---

## Primary Baseline: Unsloth

Unsloth is the current state-of-the-art open-source implementation for LoRA attention projections.

**Source**: https://github.com/unslothai/unsloth

### What Unsloth Does for QKV + LoRA

Unsloth uses its `matmul_lora()` function for each projection independently:

```python
Q = matmul_lora(X, W_q, W_q_quant, A_q, B_q, s_q)
K = matmul_lora(X, W_k, W_k_quant, A_k, B_k, s_k)
V = matmul_lora(X, W_v, W_v_quant, A_v, B_v, s_v)
```

Each `matmul_lora()` call internally does:
```python
out = torch.matmul(X, W.t())       # cuBLAS call 1
XA = torch.matmul(X, A.t())        # cuBLAS call 2 (X read again)
out.addmm_(XA, B.t(), alpha=s)     # cuBLAS call 3
```

### Unsloth's QKV Forward: 9 Kernel Launches

```
matmul_lora() for Q:
  1. torch.matmul(X, W_q.t())      ← cuBLAS (X read from HBM)
  2. torch.matmul(X, A_q.t())      ← cuBLAS (X read again)
  3. out.addmm_(XA, B_q.t())       ← cuBLAS

matmul_lora() for K:
  4. torch.matmul(X, W_k.t())      ← cuBLAS (X read again)
  5. torch.matmul(X, A_k.t())      ← cuBLAS (X read again)
  6. out.addmm_(XA, B_k.t())       ← cuBLAS

matmul_lora() for V:
  7. torch.matmul(X, W_v.t())      ← cuBLAS (X read again)
  8. torch.matmul(X, A_v.t())      ← cuBLAS (X read again)
  9. out.addmm_(XA, B_v.t())       ← cuBLAS
```

### Unsloth's Bottlenecks

| Bottleneck | Impact |
|-----------|--------|
| `X` read from HBM 6 times (Q W, Q A, K W, K A, V W, V A) | Bandwidth-bound on large sequences |
| `X @ A` intermediates ([B*S, r]) materialized to HBM 3 times | Unnecessary — they're tiny and fit in SRAM |
| 9 kernel launches | Launch overhead dominates at small batch/seq |
| No cross-projection fusion — each cuBLAS call has its own tiling | Missed opportunity to share X tiles across Q/K/V |
| Each projection output written separately | 3 writes of [B*S, H] to HBM |

### What Unsloth Does Well (Keep)

- Autograd-level fusion: single `autograd.Function` for attention forward+backward
- Quantization support (bitsandbytes 4-bit, FP8)
- In-place gradient accumulation
- Handles GQA (different K/V dimensions)

---

## Primary Baseline: Liger Kernel

### What Liger Does for Attention

Liger Kernel does **NOT** provide fused QKV projections. Their attention-related work focuses on:
- FlashAttention-compatible wrappers
- RoPE (rotary positional embedding) Triton kernels
- Cross-entropy loss fusion

Liger does **NOT** handle:
- QKV projection fusion
- LoRA computation at any level
- Attention projection matmuls

### Our Opportunity vs Liger

Liger leaves the entire QKV+LoRA computation untouched. This is a greenfield opportunity — no existing Triton kernel to compare against for this specific operation.

---

## Other Related Work

### PyTorch Native

PyTorch's `nn.Linear` performs each projection independently. With LoRA adapters (from PEFT/HuggingFace), this becomes 9+ separate operations per forward pass.

### Packed QKV

Some implementations pack W_q, W_k, W_v into a single `[3H, H]` weight and compute a single matmul: `QKV = x @ W_qkv^T`, then split. This reduces launches from 3→1 for base weights but doesn't help with LoRA (each projection has separate A/B).

### CUTLASS / cuBLAS

- Grouped GEMM API can batch multiple matmuls of same shape
- Stream-K decomposition handles workload imbalance
- But neither handles the LoRA addition inside the matmul

---

## Improvement Axes

### Axis A: Fuse LoRA into Base QKV Matmul → v1

**Goal**: compute `Y = X @ W^T + s * (X @ A^T) @ B^T` in a single Triton tiled matmul, applied independently to each of Q, K, V.

```
Unsloth: 3 cuBLAS calls per projection, X read twice, X@A materialized to HBM
Ours:    1 Triton kernel per projection, X read once, X@A stays in registers/SRAM
```

**Kernel design** (output-stationary tiled matmul):
1. Standard K-loop: accumulate `X_tile @ W_tile` in fp32 registers
2. After K-loop, compute LoRA for the same output tile:
   - Load full A column slice (only r columns — fits in registers for r ≤ 64)
   - Compute `X_tile @ A_slice` → shape [BLOCK_M, r] in registers
   - Load B row slice for this output tile
   - Compute `(X_tile @ A_slice) @ B_slice` → shape [BLOCK_M, BLOCK_N]
   - Add to accumulator: `acc += s * lora_result`
3. Store final result

**Register budget**: for r=16, BLOCK_M=128: 128×16 = 2048 fp32 values = 8 KB.

**Benchmark target**: replace Unsloth's `matmul_lora()` — 1 launch vs 3, X read once vs twice.

### Axis B: Fuse Q+K+V Projections → v2

**Goal**: load `X` once, compute all three projections (Q, K, V) with LoRA, write Q/K/V to HBM.

```
Unsloth: 9 cuBLAS = 9 launches, X read 6x from HBM
Ours:    1 Triton kernel, X read 1x from HBM, Q/K/V written out
```

**Kernel design**:
- Each thread block computes tiles of Q, K, AND V for the same input rows
- For each BLOCK_M × BLOCK_N output tile:
  - K-loop for Q: `acc_q += X_tile @ W_q_tile`
  - K-loop for K: `acc_k += X_tile @ W_k_tile` (X_tile already in SRAM)
  - K-loop for V: `acc_v += X_tile @ W_v_tile` (X_tile already in SRAM)
  - Add LoRA terms to each (Axis A technique)
  - Write Q, K, V tiles to HBM

**Challenge**: 3× the weight data must be loaded. The K-loop for each projection reads different weight tiles. Register pressure from 3 accumulators.

**Alternative approach**: fuse just 2 projections (Q+K or K+V) if 3 is too much register pressure.

### Axis C: Full Attention Projection Forward in autograd.Function → v3

**Goal**: wrap everything in a `torch.autograd.Function` with custom backward for training.

```
Unsloth: autograd.Function with 9 cuBLAS calls in forward
Ours:    autograd.Function with 1-3 Triton kernel launches in forward
```

This is the packaging step — combine the best kernel(s) from v1/v2 into a training-compatible wrapper with proper gradient computation.

### Axis D: Fused Backward (stretch goal)

The backward pass for QKV+LoRA computes:
- dX through all 3 projections (accumulated)
- d_A_q, d_B_q, d_A_k, d_B_k, d_A_v, d_B_v (6 LoRA gradients)

Unsloth's backward is 12+ kernel launches. A fused backward could reduce this significantly.

---

## Version Progression

| Version | Approach | Launches (fwd) | Baseline Comparison |
|---------|----------|----------------|---------------------|
| **v1** | Per-projection fused LoRA matmul | 3 (one per Q/K/V) | vs Unsloth `matmul_lora()` × 3 |
| **v2** | Q+K+V projection fusion (load X once) | 1 | vs Unsloth's 9 cuBLAS |
| **v3** | Full forward in autograd.Function | 1–3 | vs Unsloth attention autograd |
| **v4** | Fused backward (stretch) | TBD | vs Unsloth attention backward |

## Rank-Dependent Strategy

| Rank | `X@A` tile size (BLOCK_M=128) | Storage | Strategy |
|------|-------------------------------|---------|----------|
| r ≤ 16 | 128×16 = 8 KB | Registers | Full register-level fusion |
| r = 32–64 | 128×64 = 32 KB | SRAM | Shared-memory ping-pong |
| r ≥ 128 | 128×128 = 64 KB | Spills | Separate GEMM likely faster |

Start with r ≤ 64 (covers the vast majority of practical LoRA configs).

---

## Key Differences from LoRA MLP Fusion

The QKV case differs from MLP in important ways:

| Aspect | LoRA MLP | LoRA QKV |
|--------|----------|----------|
| Projections | 3 (gate, up, down) | 3 (Q, K, V) |
| Non-linearity | SwiGLU between projections | None — projections are independent |
| Output coupling | gate×up elementwise | Q, K, V are independent outputs |
| GQA | N/A | K, V may have different dimensions |
| Fusion opportunity | Can fuse activation with matmul | Can fuse all 3 matmuls (no activation barrier) |

The key insight: QKV projections are **completely independent** with **no non-linearity** between them. This makes cross-projection fusion simpler than MLP (no SwiGLU complication), and potentially more rewarding (we can fuse all 3 without an activation barrier).

---

## Open Questions

1. Can we beat cuBLAS with Triton for these matmul shapes? (Lesson from lora_mlp: probably not — use cuBLAS + Triton epilogue instead)
2. For GQA, should K/V fusion be separate from Q? (different output dimensions)
3. What's the optimal split: all-Triton vs cuBLAS+Triton-epilogue (like lora_mlp v3)?
4. Is packed QKV (single [3H, H] weight) compatible with per-projection LoRA?
5. At what rank does fused LoRA stop being beneficial vs separate cuBLAS?
6. How does the optimal approach change between A100 and H100?

## Lessons from LoRA MLP (Apply Here)

1. **Don't try to beat cuBLAS at matmul** — use cuBLAS for the base matmul, Triton for custom fusion only
2. **Epilogue fusion wins** — cuBLAS for W matmul, then Triton kernel that reads base output + does LoRA + writes final result
3. **Keep it rank-independent** — design so r=8 and r=64 run at similar speed
4. **X reuse is the main win** — eliminating redundant HBM reads of the input tensor

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 (tiling strategy reference)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
- [Unsloth](https://github.com/unslothai/unsloth) — Daniel Han-Chen, 2023 (Apache-2.0)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn, 2024 (BSD-2-Clause)
- Detailed code analysis: `docs/artifacts/ANALYSIS.md`
- LoRA MLP sister project: `../lora_mlp/` (lessons learned apply)

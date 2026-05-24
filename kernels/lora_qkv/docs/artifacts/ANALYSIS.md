# LoRA QKV Kernel Analysis: Unsloth vs Liger Kernel

> Retrieved 2026-05-24 from GitHub.
> This document analyzes how two open-source projects handle the attention QKV +
> LoRA computation in LLaMA-style transformers, what they fuse, and what they don't.

---

## Table of Contents

1. [Background: The QKV + LoRA Problem](#background)
2. [Unsloth Approach](#unsloth)
3. [Liger Kernel Approach](#liger-kernel)
4. [Side-by-Side Comparison](#comparison)
5. [What Neither Project Does (Our Opportunity)](#opportunity)
6. [File Index](#file-index)
7. [Code Artifacts](#code-artifacts)

---

## 1. Background: The QKV + LoRA Problem <a name="background"></a>

A LLaMA-style multi-head attention layer projects the input `x` into queries, keys, and values:

```
Q = x @ W_q^T                    # [B*S, H] -> [B*S, H_q]
K = x @ W_k^T                    # [B*S, H] -> [B*S, H_kv]
V = x @ W_v^T                    # [B*S, H] -> [B*S, H_kv]
```

Where `H_q = num_heads × head_dim` and `H_kv = num_kv_heads × head_dim`. For LLaMA-3 8B with GQA: H_q=4096, H_kv=1024 (8 KV heads × 128 head_dim).

With LoRA applied to all three projections:

```
Q = x @ W_q^T + s_q * (x @ A_q^T) @ B_q^T    # A_q: [r, H], B_q: [H_q, r]
K = x @ W_k^T + s_k * (x @ A_k^T) @ B_k^T    # A_k: [r, H], B_k: [H_kv, r]
V = x @ W_v^T + s_v * (x @ A_v^T) @ B_v^T    # A_v: [r, H], B_v: [H_kv, r]
```

This naively requires:
- **9 matmuls**: 3 base (W) + 6 LoRA (3× A, 3× B)
- **9 kernel launches** minimum
- **X read from HBM 6 times** (once per W matmul + once per A matmul, for each projection)
- **3 intermediate tensors** `x @ A` (shape `[B*S, r]`) materialized to HBM

The key optimization opportunities are:
1. **LoRA fusion**: fuse `x @ W^T + s * (x @ A^T) @ B^T` into one kernel to avoid intermediate writes
2. **Cross-projection fusion**: load `x` once, compute Q, K, V simultaneously
3. **No non-linearity barrier**: unlike MLP, there is no activation function between projections

---

## 2. Unsloth Approach <a name="unsloth"></a>

**Repository**: [unslothai/unsloth](https://github.com/unslothai/unsloth)
**License**: Apache-2.0

### What It Does for QKV

Unsloth applies LoRA to Q, K, V projections independently using `matmul_lora()`:

```python
Q = matmul_lora(X, W_q, W_q_quant, A_q, B_q, s_q)
K = matmul_lora(X, W_k, W_k_quant, A_k, B_k, s_k)
V = matmul_lora(X, W_v, W_v_quant, A_v, B_v, s_v)
```

Each `matmul_lora()` call internally does:

```python
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    W = fast_dequantize(W, W_quant)   # handle 4-bit weights
    out = torch.matmul(X, W.t())      # cuBLAS call 1
    if A is not None:
        XA = torch.matmul(X, A.t())   # cuBLAS call 2 (X read again)
        out.addmm_(XA, B.t(), alpha=s) # cuBLAS call 3
    return out
```

### Unsloth's QKV Forward: 9 Kernel Launches

```
matmul_lora() for Q:
  1. torch.matmul(X, W_q.t())        ← cuBLAS (X read from HBM)
  2. torch.matmul(X, A_q.t())        ← cuBLAS (X read again)
  3. out_q.addmm_(XA_q, B_q.t())     ← cuBLAS

matmul_lora() for K:
  4. torch.matmul(X, W_k.t())        ← cuBLAS (X read again)
  5. torch.matmul(X, A_k.t())        ← cuBLAS (X read again)
  6. out_k.addmm_(XA_k, B_k.t())     ← cuBLAS

matmul_lora() for V:
  7. torch.matmul(X, W_v.t())        ← cuBLAS (X read again)
  8. torch.matmul(X, A_v.t())        ← cuBLAS (X read again)
  9. out_v.addmm_(XA_v, B_v.t())     ← cuBLAS
```

### HBM Traffic Analysis

At LLaMA-3 8B scale (batch=4, seq=2048, H=4096, bf16):

| Tensor | Shape | Size (bf16) | Times Read | Total HBM |
|--------|-------|-------------|------------|-----------|
| X | [8192, 4096] | 64 MB | 6 | 384 MB |
| W_q | [4096, 4096] | 32 MB | 1 | 32 MB |
| W_k | [1024, 4096] | 8 MB | 1 | 8 MB |
| W_v | [1024, 4096] | 8 MB | 1 | 8 MB |
| A_q, A_k, A_v | [r, 4096] × 3 | ~0.4 MB | 1 each | ~0.4 MB |
| B_q, B_k, B_v | [N, r] × 3 | ~0.4 MB | 1 each | ~0.4 MB |
| X@A intermediates | [8192, r] × 3 | ~0.8 MB | written + read | ~1.6 MB |
| **Total** | | | | **~435 MB** |

With perfect fusion (X read once): ~113 MB saved, a **26% reduction** in HBM traffic.

### What Unsloth Does Well

- **Autograd-level fusion**: wraps attention forward+backward in a custom `autograd.Function`
- **Quantization support**: handles bitsandbytes 4-bit and FP8 base weights
- **In-place gradient accumulation**: avoids extra allocations in backward
- **GQA handling**: correctly uses different-sized W_k/W_v for GQA models

### Key Design Decisions

1. **No kernel-level fusion**: matmuls are separate cuBLAS calls — optimized for the common case where cuBLAS is fastest
2. **Per-projection independence**: Q, K, V handled identically via the same `matmul_lora()` function
3. **No cross-projection sharing**: each projection reads X independently from HBM

---

## 3. Liger Kernel Approach <a name="liger-kernel"></a>

**Repository**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
**License**: BSD-2-Clause

### What It Does for Attention

Liger Kernel does **NOT** provide fused QKV projections or LoRA handling. Their attention-related work is limited to:

- **RoPE**: Triton kernel for rotary positional embeddings
- **Cross-entropy**: fused cross-entropy loss for memory savings
- **SwiGLU activation**: fused SiLU(a) * b for MLP layers (not attention)
- **FlashAttention wrappers**: compatibility layers for flash attention

### What It Does NOT Do

- No QKV projection fusion
- No LoRA computation at any level (not in attention, not in projections)
- No attention projection matmuls
- No cross-projection optimization

### Relevance to Our Project

Liger is **not a baseline** for this project. The QKV + LoRA fusion is a greenfield opportunity — no existing Triton kernel implements it. Liger's Triton kernel design patterns (row-parallel grid, fp32 accumulation, in-place backward) are useful as reference for code structure only.

---

## 4. Side-by-Side Comparison <a name="comparison"></a>

| Aspect | Unsloth | Liger Kernel |
|--------|---------|-------------|
| **QKV projection** | matmul_lora() × 3 (cuBLAS) | Not handled (uses nn.Linear) |
| **LoRA handling** | Full forward + backward + gradients | Not handled |
| **Kernel-level fusion** | None (cuBLAS calls) | None for QKV |
| **Cross-projection fusion** | None | None |
| **GQA support** | Yes (different K/V dims) | N/A |
| **Quantization** | bitsandbytes 4-bit, FP8 | Not handled |
| **Forward launches** | 9 (3 per projection) | 3+ (just base matmuls) |
| **X HBM reads** | 6 | 3 (no LoRA) |
| **Triton kernels** | SwiGLU only (for MLP) | RoPE, SwiGLU, cross-entropy |

### Forward Pass Kernel Launches

| Operation | Unsloth (with LoRA) | PyTorch naive | Theoretical best |
|-----------|---------------------|---------------|-----------------|
| Q projection + LoRA | 3 cuBLAS | 3 separate ops | 1 fused |
| K projection + LoRA | 3 cuBLAS | 3 separate ops | 1 fused |
| V projection + LoRA | 3 cuBLAS | 3 separate ops | 1 fused |
| **Total** | **9 launches** | **9 launches** | **1–3 launches** |

---

## 5. What Neither Project Does (Our Opportunity) <a name="opportunity"></a>

Neither Unsloth nor Liger fuses the LoRA computation at the **kernel level** for attention projections. Both leave the matmuls as separate cuBLAS calls. The key opportunities for our fused kernel are:

### 5.1 Fuse LoRA into the Base Matmul

Instead of 3 separate cuBLAS calls per projection:
```
out = X @ W.t()            # cuBLAS call 1
XA = X @ A.t()             # cuBLAS call 2 (X read again)
out += s * XA @ B.t()      # cuBLAS call 3
```

Fuse into a single kernel:
```
# Inside one tiled matmul kernel:
for each output tile:
    acc = 0
    for k in range(0, K, BLOCK_K):
        acc += X_tile @ W_tile
    lora_acc = (X_tile @ A_tile) @ B_tile
    acc += s * lora_acc
    store(acc)
```

This saves 2 kernel launches per projection and avoids materializing `X @ A` to HBM.

### 5.2 Fuse Q + K + V Projections

All three projections read the same input `X`. A fused kernel could:
- Load `X` tiles once from HBM
- Compute `X @ W_q`, `X @ W_k`, `X @ W_v` output tiles
- Add LoRA terms to each
- Write Q, K, V tiles to HBM

This reduces X reads from 6× to 1×. Unlike the MLP case, there is **no activation barrier** between Q, K, V — they are completely independent, making full fusion possible.

### 5.3 GQA-Aware Tiling

For GQA models where K/V output dimension is 4× smaller than Q:
- K/V matmuls (`[M, H] @ [H, H_kv]`) are 4× less compute than Q
- Separate tiling strategies for Q (large output) vs K/V (small output)
- K and V could share the same tile iteration since they have identical shapes

### 5.4 Register-Level LoRA for Small Ranks

For `r ≤ 16`, the LoRA intermediate `X_tile @ A` (shape `[BLOCK_M, r]`) fits in registers. This means:
- No shared memory needed for the LoRA path
- The LoRA addition is essentially free on top of the base matmul
- This is the sweet spot for our fused kernel

### 5.5 Lessons from LoRA MLP (Apply Here)

From the sister project `../lora_mlp/`:

1. **Don't try to beat cuBLAS at matmul** — Triton matmul achieved only ~0.73× of cuBLAS throughput
2. **Epilogue fusion wins** — cuBLAS for W matmul, then Triton for the LoRA epilogue
3. **Keep it rank-independent** — design so r=8 and r=64 run at similar speed
4. **X reuse is the main win** — eliminating redundant HBM reads of the input tensor

The QKV case is **more favorable** than MLP because:
- No activation function barrier (can fuse all 3 projections freely)
- All projections read the same input X (6× redundant reads in baseline)
- Simpler output structure (3 independent tensors vs gate×up coupling)

### 5.6 Summary of Fusion Levels

```
Level 0: Naive PyTorch       → 9+ kernel launches, X read 6x from HBM
Level 1: Unsloth (cuBLAS)    → 9 launches, X read 6x, X@A intermediates in HBM
Level 2: Our v1 (per-proj)   → 3 launches, X read 3x, X@A stays in registers
Level 3: Our v2 (fused QKV)  → 1 launch, X read 1x, all LoRA in registers
```

---

## 6. File Index <a name="file-index"></a>

### Unsloth

| File | Description | Source URL |
|------|-------------|-----------|
| — | `matmul_lora()` used for Q, K, V projections | [GitHub (fast_lora.py)](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) |
| — | Attention autograd.Function | [GitHub (llama.py)](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py) |

### Liger Kernel

| File | Description | Source URL |
|------|-------------|-----------|
| — | No QKV-specific kernels exist | [GitHub](https://github.com/linkedin/Liger-Kernel) |
| — | RoPE kernel (reference for Triton patterns) | [GitHub (rope.py)](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py) |

### Project References

| File | Description |
|------|-------------|
| `reference/lora_qkv_pytorch.py` | PyTorch reference with forward + backward |
| `reference/unsloth_baseline.py` | Unsloth-style baseline with addmm_ (primary perf target) |
| `docs/research.md` | Research notes, improvement axes, GQA analysis |
| `docs/benchmarks.md` | Benchmark methodology and results |
| `../lora_mlp/` | Sister project — lessons learned apply here |

---

## 7. Code Artifacts <a name="code-artifacts"></a>

Extracted source code from baseline projects, annotated with explanations of how each
part works and how it relates to our fused kernel project.

### Unsloth Artifacts

| File | Source URL | Description |
|------|-----------|-------------|
| `docs/artifacts/unsloth/matmul_lora.py` | [fast_lora.py](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) + [utils.py](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py) | Core `matmul_lora()` function and `LoRA_QKV` autograd.Function. Shows the 3-cuBLAS-call pattern per projection with addmm_ for fused add+GEMM. Includes full backward pass. |
| `docs/artifacts/unsloth/swiglu_triton.py` | [swiglu.py](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py) | SwiGLU Triton kernels (forward + backward). Reference for Triton patterns: flat 1D grid, fp32 sigmoid, in-place backward, int64 long indexing guard. |

### Liger Kernel Artifacts

| File | Source URL | Description |
|------|-----------|-------------|
| `docs/artifacts/liger_kernel/swiglu_ops.py` | [ops/swiglu.py](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py) | SwiGLU Triton kernels. Row-parallel grid pattern, gate_multiplier support, LigerSiLUMulFunction autograd.Function with DTensor support. |
| `docs/artifacts/liger_kernel/swiglu_mlp.py` | [transformers/swiglu.py](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/swiglu.py) | MLP wrappers for HuggingFace models (LLaMA, Mixtral, Phi-3, FalconH1). Shows the pattern for integrating Triton kernels into HF model architecture. |

### Baseline Reference

| File | Description |
|------|-------------|
| `reference/unsloth_baseline.py` | Unsloth-style LoRA QKV using plain PyTorch with addmm_. Implements `matmul_lora_unsloth()`, `qkv_lora_unsloth()`, and `LoRAQKVUnsloth` autograd.Function. Primary performance baseline. |

### Key Observations from Code Extraction

1. **Unsloth's `matmul_lora()` is simple but effective**: 3 cuBLAS calls with addmm_ for the LoRA path. The addmm_ avoids allocating a temporary tensor and fuses scalar multiply + matmul + addition.

2. **Unsloth's `LoRA_QKV` backward uses 12+ cuBLAS calls**: 6 addmm_ calls for LoRA gradients (dA, dB for each of Q/K/V) plus 6 calls for dX accumulation (3 base weight contributions + 3 LoRA contributions).

3. **Both Unsloth and Liger use in-place operations extensively**: addmm_ in Unsloth, tl.store to input buffers in Liger's backward. Memory efficiency is a first-class concern.

4. **Liger's row-parallel pattern vs Unsloth's flat pattern**: For elementwise ops, Liger assigns one program per row (better locality when row fits in BLOCK_SIZE), Unsloth assigns programs to contiguous blocks (simpler, handles arbitrary shapes).

5. **Neither project fuses at the matmul level for LoRA**: Both keep the base matmul (X @ W.t()) as a separate cuBLAS call. Our kernel's primary value proposition is fusing the LoRA computation *into* the matmul tile loop.

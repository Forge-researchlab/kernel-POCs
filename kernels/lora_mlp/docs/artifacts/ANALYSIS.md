# LoRA MLP Kernel Analysis: Liger Kernel vs Unsloth

> Retrieved 2026-05-23 from GitHub.
> This document analyzes how two open-source projects handle the MLP + LoRA
> computation in LLaMA-style transformers, what they fuse, and what they don't.

---

## Table of Contents

1. [Background: The MLP + LoRA Problem](#background)
2. [Liger Kernel Approach](#liger-kernel)
3. [Unsloth Approach](#unsloth)
4. [Side-by-Side Comparison](#comparison)
5. [What Neither Project Does (Our Opportunity)](#opportunity)
6. [File Index](#file-index)

---

## 1. Background: The MLP + LoRA Problem <a name="background"></a>

A LLaMA-style MLP with LoRA on all three projections involves:

```
gate = x @ (W_gate + s_g * A_g @ B_g)    # [B*S, H] -> [B*S, I]
up   = x @ (W_up   + s_u * A_u @ B_u)    # [B*S, H] -> [B*S, I]
act  = SiLU(gate) * up                    # SwiGLU activation
down = act @ (W_down + s_d * A_d @ B_d)   # [B*S, I] -> [B*S, H]
```

This naively requires:
- **9 matmuls**: 3 base (W) + 6 LoRA (3x A, 3x B)
- **9 kernel launches** minimum
- **Intermediate tensors** stored in HBM: `x @ A_g`, `x @ A_u`, `act @ A_d`, etc.

The two key optimization opportunities are:
1. **Activation fusion**: fuse `SiLU(gate) * up` to avoid materializing the full intermediate
2. **LoRA fusion**: fuse `x @ W + s * (x @ A) @ B` to reuse `x` from SRAM/registers

---

## 2. Liger Kernel Approach <a name="liger-kernel"></a>

**Repository**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
**License**: BSD-2-Clause
**Files**: `artifacts/liger_kernel/swiglu_ops.py`, `artifacts/liger_kernel/swiglu_mlp.py`

### What It Fuses

Liger fuses **only the SwiGLU pointwise activation**: `c = SiLU(a) * b`.

It does **NOT** fuse:
- The matmuls (gate_proj, up_proj, down_proj remain separate cuBLAS calls)
- Any LoRA computation (LoRA terms aren't handled at all)

### Architecture

```
┌─────────────────────────────────────────────────┐
│ LigerSwiGLUMLP.forward(x)                      │
│                                                 │
│  gate = self.gate_proj(x)   ← cuBLAS matmul    │
│  up   = self.up_proj(x)     ← cuBLAS matmul    │
│  fused = LigerSiLUMulFunction(gate, up)         │
│          ↑                                      │
│          │ Triton kernel: SiLU(gate) * up       │
│          │ One kernel, one row per thread block  │
│  out  = self.down_proj(fused) ← cuBLAS matmul  │
└─────────────────────────────────────────────────┘
```

### How the Triton Kernel Works

**Forward** (`_swiglu_forward_kernel`):
- Grid: one program per row (each row = one token's intermediate dimension)
- Each program loads one row of `gate` and `up` (both shape `[I]`)
- Computes `SiLU(gate) * up` elementwise in fp32 (for sigmoid stability)
- Casts result back to input dtype, stores to output

```python
a_row = tl.load(a_ptr + col_offsets, mask=mask).to(tl.float32)
b_row = tl.load(b_ptr + col_offsets, mask=mask)
c_row = silu(a_row).cast(b_row.dtype) * b_row
tl.store(c_ptr + col_offsets, c_row, mask=mask)
```

**Backward** (`_swiglu_backward_kernel`):
- Recomputes `sigmoid(gate)` and `SiLU(gate)` from saved `gate` values (saves memory vs. saving the activation)
- Computes gradients for both `gate` and `up` projections
- Writes gradients **in-place** into the saved `gate` and `up` buffers

The derivative of SiLU is: `d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`

### Key Design Decisions

1. **Row-parallel grid**: one thread block per token row — simple but limits parallelism to `B*S`
2. **fp32 accumulation**: sigmoid always computed in fp32 for stability
3. **In-place backward**: gradients overwrite the saved tensors (`a` and `b`), cutting memory by 2x
4. **Recomputation**: forward recomputes `sigmoid` in backward rather than saving it (classic memory/compute trade-off)
5. **No matmul fusion**: the intermediate dimension `I` (e.g., 14336 for LLaMA-8B) is too large for SRAM, so fusing the matmul would require tiled output accumulation — Liger chose simplicity

### Memory Savings

Without Liger: must store `gate_output`, `silu_gate`, `up_output` (3 tensors of shape `[B*S, I]`)
With Liger: stores `gate_output` and `up_output` only, recomputes `SiLU` in backward (2 tensors)

This gives ~1.5x peak memory reduction on the activation, per their README.

---

## 3. Unsloth Approach <a name="unsloth"></a>

**Repository**: [unslothai/unsloth](https://github.com/unslothai/unsloth)
**License**: Apache-2.0
**Files**: `artifacts/unsloth/fast_lora_mlp.py`, `artifacts/unsloth/swiglu_triton.py`

### What It Fuses

Unsloth fuses the **entire MLP autograd graph** including LoRA, but at the **PyTorch level**, not at the Triton/CUDA level. The matmuls themselves are still separate cuBLAS/bitsandbytes calls.

Specifically:
1. **Autograd fusion**: the entire MLP (3 base matmuls + 6 LoRA matmuls + SwiGLU + all gradients) is a single `torch.autograd.Function`
2. **SwiGLU Triton kernels**: pointwise activation fused via Triton (similar to Liger but with in-place buffer reuse)
3. **LoRA matmuls**: done via `matmul_lora()` which calls `torch.matmul` + `addmm_` — standard cuBLAS, not fused

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ LoRA_MLP.forward(X, gateW, gateA, gateB, ..., downB, downS) │
│                                                              │
│  e = matmul_lora(X, gateW, gateA, gateB, gateS)             │
│      ↑ cuBLAS: X @ W_gate + s * (X @ A_g) @ B_g             │
│                                                              │
│  g = matmul_lora(X, upW, upA, upB, upS)                     │
│      ↑ cuBLAS: X @ W_up + s * (X @ A_u) @ B_u               │
│                                                              │
│  h = swiglu_fg_kernel(e, g)                                  │
│      ↑ Triton: SiLU(e) * g                                  │
│                                                              │
│  i = matmul_lora(h, downW, downA, downB, downS)              │
│      ↑ cuBLAS: h @ W_down + s * (h @ A_d) @ B_d             │
│                                                              │
│  save_for_backward: X, e, g, all LoRA A/B matrices          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LoRA_MLP.backward(dY)                                        │
│                                                              │
│  Step 1: DW = dY @ W_down^T (+ LoRA)                        │
│                                                              │
│  Step 2: Triton backward kernel (in-place!)                  │
│          DW buffer → h = f * g                               │
│          e  buffer → df = DW * f                             │
│          g  buffer → de = DW * g * σ(e) * (1 + e(1-σ(e)))   │
│                                                              │
│  Step 3: LoRA grads via addmm_ (6 matmuls)                  │
│          d_downA, d_downB, d_upA, d_upB, d_gateA, d_gateB   │
│                                                              │
│  Step 4: Input gradient dX via dequantized W + LoRA terms    │
│          dX  = df @ W_up^T + df @ B_u^T @ A_u^T             │
│          dX += de @ W_gate^T + de @ B_g^T @ A_g^T           │
└──────────────────────────────────────────────────────────────┘
```

### How the SwiGLU Triton Kernels Work

**Forward** (`_fg_kernel`):
- Flat grid over all elements (not row-based like Liger)
- Block size = 1024 (fixed, not autotuned)
- Handles int32 overflow for large tensors (> 2^31 elements) with `LONG_INDEXING`

```python
e_row = tl.load(e + offsets, mask=mask).to(tl.float32)
g_row = tl.load(g + offsets, mask=mask)
f_row = e_row * tl.sigmoid(e_row)  # SiLU in fp32
f_row = f_row.to(g_row.dtype)      # cast back
h_row = f_row * g_row
tl.store(h + offsets, h_row, mask=mask)
```

**Backward** (`_DWf_DW_dfg_kernel`) — the clever part:
- Takes 3 input buffers (DW, e, g) and **overwrites all 3 with different results**
- `DW` (which held `dY @ W_down^T`) is overwritten with `h = f * g` (needed for down LoRA grads)
- `e` (which held gate pre-activations) is overwritten with `df = DW * f` (up-projection gradient)
- `g` (which held up-projection output) is overwritten with `de` (gate-projection gradient)
- This reuses 3 buffers that would otherwise be dead, saving 3 tensor allocations

### How matmul_lora Works

```python
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    # Step 1: Base matmul (handles quantized weights)
    W = fast_dequantize(W, W_quant)  # 4-bit -> fp16/bf16 on-the-fly
    out = torch.matmul(X, W.t())

    # Step 2: LoRA additive term (if A is not None)
    XA = torch.matmul(X, A.t().to(dtype))
    out.addmm_(XA, B.t().to(dtype), alpha=s)  # out += s * XA @ B^T

    return out
```

This is 3 operations (dequantize + base matmul + LoRA addmm) but they are NOT fused into a single kernel.

### Key Design Decisions

1. **Autograd-level fusion**: by writing a custom `autograd.Function`, Unsloth controls the entire forward+backward graph, allowing buffer reuse and specific gradient ordering
2. **In-place backward**: the backward Triton kernel overwrites 3 input buffers, saving ~3× intermediate memory
3. **In-place dX**: input gradient is accumulated in-place into `X` when `inplace=True`, saving one allocation
4. **addmm_ for LoRA grads**: uses `addmm_` with `alpha=s, beta=0` which is a single cuBLAS call per LoRA gradient
5. **Quantization-aware**: `matmul_lora` handles bitsandbytes 4-bit weights natively
6. **Fixed block size**: 1024 elements per block, no autotuning (simpler, possibly suboptimal for some shapes)

### Memory Savings

The main memory wins come from:
- Not storing `SiLU(gate)` separately (recomputed in backward)
- Buffer reuse in backward (3 tensors rewritten in-place)
- In-place `dX` accumulation (saves one `[B*S, H]` tensor)
- Custom autograd avoids PyTorch's default retention of all intermediates

---

## 4. Side-by-Side Comparison <a name="comparison"></a>

| Aspect | Liger Kernel | Unsloth |
|--------|-------------|---------|
| **Fusion scope** | SwiGLU activation only | Full MLP autograd (PyTorch-level) |
| **Triton kernels** | SwiGLU forward + backward | SwiGLU forward + backward (similar) |
| **Matmul fusion** | None (cuBLAS) | None (cuBLAS / bitsandbytes) |
| **LoRA handling** | Not handled | Full forward + backward + gradients |
| **Quantization** | Not handled | bitsandbytes 4-bit, FP8 support |
| **Memory technique** | Recompute activation, in-place grads | Recompute activation, in-place buffer reuse (3 buffers), in-place dX |
| **Grid strategy** | Row-parallel (1 block/row) | Flat (1024 elements/block) |
| **Autotuning** | Via `calculate_settings` (power-of-2 blocks) | Fixed BLOCK_SIZE=1024 |
| **Backward memory** | Saves `a`, `b` tensors | Saves `X`, `e`, `g` + LoRA matrices, overwrites in backward |
| **DTensor support** | Yes (distributed) | No |
| **LoRA grad computation** | N/A | `addmm_` with alpha scaling |

### Forward Pass Kernel Launches

| Operation | Liger (no LoRA) | Unsloth (with LoRA) |
|-----------|----------------|---------------------|
| gate projection | 1 cuBLAS | 1 cuBLAS + 1 addmm |
| up projection | 1 cuBLAS | 1 cuBLAS + 1 addmm |
| SwiGLU activation | 1 Triton | 1 Triton |
| down projection | 1 cuBLAS | 1 cuBLAS + 1 addmm |
| **Total** | **4 launches** | **7 launches** |

### Memory Footprint (Training, per MLP layer)

Assuming shapes: `[B*S, H]` input, `[B*S, I]` intermediate, rank `r`:

| Saved Tensor | Liger | Unsloth |
|-------------|-------|---------|
| Input `X` | No (PyTorch saves it) | Yes (`[B*S, H]`) |
| Gate pre-act `e` | Yes (`[B*S, I]`) | Yes (`[B*S, I]`) |
| Up output `g` | Yes (`[B*S, I]`) | Yes (`[B*S, I]`) |
| LoRA A, B matrices | N/A | Yes (6 small matrices) |
| SiLU(gate) | No (recomputed) | No (recomputed) |

---

## 5. What Neither Project Does (Our Opportunity) <a name="opportunity"></a>

Neither Liger nor Unsloth fuses the LoRA computation at the **Triton kernel level**. Both leave the matmuls as separate cuBLAS calls. The key opportunities for our fused kernel are:

### 5.1 Fuse LoRA into the Base Matmul

Instead of:
```
out = X @ W            # cuBLAS call 1
XA = X @ A             # cuBLAS call 2
out += s * XA @ B      # cuBLAS call 3
```

Fuse into a single Triton kernel:
```
# Inside the same tiled matmul kernel:
for each output tile:
    acc = 0
    for k in range(0, K, BLOCK_K):
        acc += X_tile @ W_tile       # base matmul
    # LoRA fits in SRAM for small r:
    lora_acc = (X_tile @ A_tile) @ B_tile
    acc += s * lora_acc
    store(acc)
```

This saves 2 kernel launches per projection and avoids materializing `X @ A` to HBM.

### 5.2 Fuse Gate + Up Projections

Both gate and up projections read the same input `X`. A fused kernel could:
- Load `X` tiles once from HBM
- Compute both `X @ W_gate` and `X @ W_up` output tiles
- Apply SwiGLU in registers before writing to HBM

This halves the HBM reads of `X` and fuses the activation with the matmul.

### 5.3 Register-Level LoRA for Small Ranks

For `r <= 16`, the LoRA intermediate `X_tile @ A` (shape `[BLOCK_M, r]`) fits in registers. This means:
- No shared memory needed for the LoRA path
- The LoRA addition is essentially free on top of the base matmul
- This is the sweet spot for our fused kernel

### 5.4 Summary of Fusion Levels

```
Level 0: Naive PyTorch       → 9+ kernel launches, all intermediates in HBM
Level 1: Liger (activation)  → 4 launches, SwiGLU fused, no LoRA
Level 2: Unsloth (autograd)  → 7 launches, custom backward, buffer reuse
Level 3: Our target (Triton) → 1-3 launches, LoRA fused into matmul tiles
```

---

## 6. File Index <a name="file-index"></a>

### Liger Kernel

| File | Description | Source URL |
|------|-------------|-----------|
| `liger_kernel/swiglu_ops.py` | Triton SwiGLU kernels + autograd Function | [GitHub](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py) |
| `liger_kernel/swiglu_mlp.py` | nn.Module wrapper for HuggingFace models | [GitHub](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/swiglu.py) |

### Unsloth

| File | Description | Source URL |
|------|-------------|-----------|
| `unsloth/fast_lora_mlp.py` | LoRA_MLP autograd.Function + apply helper | [GitHub](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) |
| `unsloth/swiglu_triton.py` | SwiGLU Triton kernels (forward + backward) | [GitHub](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py) |

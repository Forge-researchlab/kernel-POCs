# LoRA MLP Fused Kernel — Research Notes

## Background

### What is LoRA?

Low-Rank Adaptation (LoRA) adds trainable low-rank matrices `A` (d×r) and `B` (r×k) alongside frozen weight `W` (d×k). The forward pass computes `y = x @ W + α · (x @ A) @ B`. Because r << min(d, k), the LoRA path is a rank bottleneck with far fewer FLOPs than the base matmul, but in an unfused implementation it still reads/writes intermediate activations from HBM.

### MLP Structure (LLaMA-style)

```
gate = x @ W_gate        # [B*S, H] → [B*S, I]
up   = x @ W_up          # [B*S, H] → [B*S, I]
act  = SiLU(gate) * up   # SwiGLU activation
down = act @ W_down       # [B*S, I] → [B*S, H]
```

With LoRA on all three projections, this becomes 9 matmuls (3 base + 6 LoRA) plus the SwiGLU.

### Why Fuse?

1. **Memory**: LoRA intermediates `x @ A` (shape [B*S, r]) are small and fit in SRAM for typical ranks. Fusing avoids materializing them to HBM.
2. **Kernel launch overhead**: 9 separate matmuls = 9 kernel launches. Fusing reduces to 1–3 launches.
3. **Memory bandwidth**: the base matmul and LoRA addition read the same input `x` — fusing reuses it from registers/SRAM.

## Primary Baseline: Unsloth

Unsloth is the current state-of-the-art open-source LoRA MLP implementation. Our kernel targets improving upon it at the GPU kernel level.

**Source**: https://github.com/unslothai/unsloth — code analysis in `docs/artifacts/unsloth/`

### What Unsloth Does

- Custom `torch.autograd.Function` (`LoRA_MLP`) handles all 3 projections + LoRA + SwiGLU in a single autograd node
- Matmuls themselves are still **separate cuBLAS calls** via `matmul_lora()`
- `matmul_lora()` computes `X @ W + s * (X @ A) @ B` as 2–3 cuBLAS calls per projection
- SwiGLU forward/backward done via Triton pointwise kernels
- Backward kernel overwrites 3 input buffers in-place (DW→h, e→df, g→de)
- Supports bitsandbytes 4-bit and FP8 quantized base weights

### Unsloth's Forward: 10 Kernel Launches

```
matmul_lora() for gate:
  1. torch.matmul(X, W_gate.t())       ← cuBLAS
  2. torch.matmul(X, A_gate.t())       ← cuBLAS  (X read again)
  3. out.addmm_(XA, B_gate.t())        ← cuBLAS

matmul_lora() for up:
  4. torch.matmul(X, W_up.t())         ← cuBLAS  (X read again)
  5. torch.matmul(X, A_up.t())         ← cuBLAS  (X read again)
  6. out.addmm_(XA, B_up.t())          ← cuBLAS

swiglu_fg_kernel(e, g):
  7. Triton pointwise SiLU(e) * g       ← reads e,g from HBM, writes h

matmul_lora() for down:
  8.  torch.matmul(h, W_down.t())       ← cuBLAS
  9.  torch.matmul(h, A_down.t())       ← cuBLAS
  10. out.addmm_(XA, B_down.t())        ← cuBLAS
```

### Unsloth's Bottlenecks

| Bottleneck | Impact |
|-----------|--------|
| `X` read from HBM 4 times (gate W, gate A, up W, up A) | Bandwidth-bound on large sequences |
| `e` and `g` (`[B*S, I]`) written to HBM, read back for SwiGLU | 2 round-trips of the largest intermediates |
| `X @ A` intermediates (`[B*S, r]`) materialized to HBM | Unnecessary — they're tiny and fit in SRAM |
| 10 kernel launches | Launch overhead dominates at small batch/seq |
| No matmul fusion — each cuBLAS call has its own tiling | Missed opportunity to share X tiles across ops |

### What Unsloth Does Well (Keep)

- Autograd-level fusion: single `autograd.Function` for the full MLP
- In-place backward buffer reuse (3 tensors overwritten by Triton kernel)
- In-place `dX` accumulation to avoid extra allocations
- `addmm_` with `alpha=s, beta=0` for LoRA gradient computation

## Other Related Work

### Liger Kernel
- Fuses **only** the SwiGLU activation (`SiLU(gate) * up`) via Triton — no LoRA handling
- Not a relevant baseline for LoRA fusion, but their activation kernel design is a reference
- Code analysis: `docs/artifacts/liger_kernel/`

### CUTLASS / cuBLAS
- Split-K and stream-K decompositions handle skinny matmuls (like LoRA) better
- Grouped GEMM API can batch multiple small matmuls

### Triton Constraints
- `tl.dot` requires both operands to be at least 16×16 for good performance
- Power-of-2 block sizes required
- No native grouped GEMM primitive (must be hand-tiled)
- Shared memory management is implicit (via `num_warps`, `num_stages`)

---

## Improvement Axes

### Axis A: Fuse LoRA into Base Matmul → v1

**Goal**: compute `Y = X @ W + s * (X @ A) @ B` in a single Triton tiled matmul.

```
Unsloth: 3 cuBLAS calls per projection, X read twice, X@A materialized to HBM
Ours:    1 Triton kernel per projection, X read once, X@A stays in registers/SRAM
```

**Kernel design** (output-stationary tiled matmul):
1. Standard K-loop: accumulate `X_tile @ W_tile` in fp32 registers
2. After the K-loop, compute the LoRA term for the same output tile:
   - Load the full `A` column slice (only `r` columns — fits in registers for r ≤ 64)
   - Compute `X_tile @ A_slice` → shape `[BLOCK_M, r]` in registers
   - Load the `B` row slice for this output tile
   - Compute `(X_tile @ A_slice) @ B_slice` → shape `[BLOCK_M, BLOCK_N]`
   - Add to accumulator: `acc += s * lora_result`
3. Store final result

**Register budget**: for r=16 and BLOCK_M=128, the LoRA intermediate is 128×16 = 2048 fp32 values = 8 KB. Well within register/SRAM budget.

**Benchmark target**: replace Unsloth's `matmul_lora()` — 1 launch vs 3, X read once vs twice.

### Axis B: Fuse Gate + Up + SwiGLU → v2

**Goal**: load `X` once, compute both gate and up outputs with LoRA, apply SwiGLU in registers, write only the fused result.

```
Unsloth: 6 cuBLAS + 1 Triton = 7 launches, X read 4x, e+g round-tripped through HBM
Ours:    1 Triton kernel, X read 1x, SwiGLU applied in registers, only h written
```

**Kernel design**:
- Each thread block computes a tile of BOTH gate and up for the same input rows
- For each BLOCK_M × BLOCK_N output tile:
  - K-loop for gate: `acc_gate += X_tile @ W_gate_tile`
  - K-loop for up: `acc_up += X_tile @ W_up_tile` (X_tile already in SRAM)
  - Add LoRA terms to both (Axis A technique)
  - Apply `SiLU(acc_gate) * acc_up` in registers
  - Write only `h` to HBM

**Memory savings**: eliminates `e` and `g` (both `[B*S, I]`), the two largest intermediate tensors.

### Axis C: Full Forward MLP → v3

**Goal**: combine v2 (gate+up+SwiGLU) with v1 (down+LoRA) for the complete forward pass.

```
Unsloth: 10 kernel launches
Ours:    2 kernel launches (gate+up+SwiGLU fused, down+LoRA fused)
```

The down projection can't fuse with gate+up because it operates on a different input (`h` vs `X`). Wrap in `torch.autograd.Function` for training.

### Axis D: Fused Backward (stretch goal)

Unsloth's backward has 20+ kernel launches (matmul_lora for backprop, SwiGLU backward Triton kernel, 6 addmm_ for LoRA grads, 2 base weight backprop matmuls + 4 LoRA addmm_ for dX). A fused backward is the hardest piece but could bring similar launch-count and HBM savings.

---

## Version Progression

| Version | Approach | Launches (fwd) | Baseline Comparison |
|---------|----------|----------------|---------------------|
| **v1** | Single-projection fused LoRA matmul | 3 (one per projection) + 1 SwiGLU = 4 | vs Unsloth `matmul_lora()` |
| **v2** | Gate+Up+SwiGLU fusion | 1 (gate+up+SwiGLU) + 1 (down) = 2 | vs Unsloth gate+up+SwiGLU path |
| **v3** | Full forward MLP in autograd.Function | 2 | vs Unsloth `LoRA_MLP.forward()` |
| **v4** | Fused backward (stretch) | TBD | vs Unsloth `LoRA_MLP.backward()` |

## Rank-Dependent Strategy

Different LoRA ranks require different fusion strategies:

| Rank | `X@A` tile size (BLOCK_M=128) | Storage | Strategy |
|------|-------------------------------|---------|----------|
| r ≤ 16 | 128×16 = 8 KB | Registers | Full register-level fusion |
| r = 32–64 | 128×64 = 32 KB | SRAM | Shared-memory ping-pong |
| r ≥ 128 | 128×128 = 64 KB | Spills | Separate GEMM likely faster |

Start with r ≤ 64 (covers the vast majority of practical LoRA configs).

## Resolved Questions

| Question | Answer |
|----------|--------|
| Does Unsloth fuse LoRA at the kernel level? | No — PyTorch autograd-level only. Matmuls are separate cuBLAS calls. |
| Does Liger handle LoRA? | No — activation fusion only, no LoRA awareness. |
| Can gate+up share input reads? | Yes — same input `X`, different weight matrices. Perfect for tile reuse. |
| Is SwiGLU worth fusing into the matmul? | Yes — it's elementwise on the output tile, essentially free in registers. |

## Open Questions

1. At what LoRA rank does Triton fusion stop beating separate cuBLAS calls? (needs benchmarking)
2. How does the optimal tiling change across GPU architectures (A100 vs H100)?
3. What's the best backward strategy — save `X@A` intermediates or recompute?
4. Can we handle non-power-of-2 LoRA ranks efficiently (e.g., r=24)?

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 (tiling strategy reference)
- [Unsloth](https://github.com/unslothai/unsloth) — Daniel Han-Chen, 2023 (Apache-2.0)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn, 2024 (BSD-2-Clause)
- Detailed code analysis: `docs/artifacts/ANALYSIS.md`

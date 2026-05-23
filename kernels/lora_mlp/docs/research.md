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

## Related Work

### Liger Kernel
- Fuses **only the SwiGLU pointwise activation** (`SiLU(gate) * up`) via Triton
- Does NOT fuse any matmul (gate, up, down remain separate cuBLAS calls)
- Does NOT handle LoRA at all — it's purely an activation fusion
- Grid strategy: one thread block per row (token), processes the full intermediate dim
- Memory savings: ~1.5x by recomputing SiLU in backward instead of saving it
- Backward writes gradients in-place into saved `gate` and `up` buffers
- With LoRA this gives 4 kernel launches (3 cuBLAS + 1 Triton activation)
- Reference: https://github.com/linkedin/Liger-Kernel
- Code analysis: `docs/artifacts/liger_kernel/`

### Unsloth
- Fuses the **entire MLP autograd graph** including LoRA at the **PyTorch level** (not Triton/CUDA)
- Custom `torch.autograd.Function` (`LoRA_MLP`) handles all 3 projections + LoRA + SwiGLU
- Matmuls themselves are still separate cuBLAS/bitsandbytes calls via `matmul_lora()`
- `matmul_lora()` computes: `X @ W + s * (X @ A) @ B` as 2-3 separate cuBLAS calls
- SwiGLU forward/backward done via Triton kernels (similar to Liger)
- Backward kernel is clever: overwrites 3 input buffers in-place (DW→h, e→df, g→de)
- Supports bitsandbytes 4-bit quantized base weights and FP8
- Total: 7 kernel launches with LoRA (3 base + 3 LoRA addmm + 1 Triton activation)
- Key insight: for small ranks, the LoRA matmul is memory-bound, not compute-bound
- Reference: https://github.com/unslothai/unsloth
- Code analysis: `docs/artifacts/unsloth/`

### CUTLASS / cuBLAS
- Split-K and stream-K decompositions handle skinny matmuls (like LoRA) better
- Grouped GEMM API can batch multiple small matmuls

### Triton Limitations
- No native grouped GEMM primitive (must be hand-tiled)
- `tl.dot` requires power-of-2 dimensions for good performance
- Shared memory management is implicit (via `num_warps`, `num_stages`)

## Research Axes

### Axis 1: Fusion Scope

What operations to fuse together:

| Strategy | Fuses | Launches | Complexity |
|----------|-------|----------|------------|
| **No fusion** | Nothing | 9+ | Trivial |
| **LoRA-only fusion** | `x@W + (x@A)@B` per projection | 3 | Medium |
| **Gate+Up fusion** | Both projections sharing input `x` | 2 | Medium |
| **Full MLP fusion** | Gate + Up + SwiGLU + Down | 1 | High |

Start with LoRA-only fusion (one projection at a time), then expand.

### Axis 2: LoRA Computation Strategy

How to compute the LoRA term `(x @ A) @ B` within the fused kernel:

- **Serial two-step**: compute `x @ A` into registers, then `(x@A) @ B`, add to base result. Simple but may waste registers.
- **Outer-product accumulation**: for each tile of the output, accumulate the rank-r outer product directly. Avoids materializing `x @ A` entirely.
- **Shared-memory ping-pong**: compute `x @ A` tiles into shared memory, then multiply by `B` tiles. Good for larger ranks.

### Axis 3: Tiling Strategy

- **Output-stationary**: each thread block computes a tile of the output. Natural for matmul, but LoRA addition requires both `W` and `B` tiles aligned.
- **Input-stationary**: each block processes a tile of input rows across all output columns. Better for reusing `x` and `x @ A` across gate/up projections.
- **Split-K**: partition the reduction dimension across blocks, then reduce. Helps when K is large relative to the number of SMs.

### Axis 4: Rank-Dependent Dispatch

Different ranks have different optimal strategies:
- **r <= 16**: LoRA intermediate fits in registers. Full register-level fusion viable.
- **r = 32–64**: Spills to shared memory. SRAM-based approach needed.
- **r >= 128**: LoRA is no longer "small" — separate GEMM may be faster than fusion overhead.

### Axis 5: Backward Pass

The backward pass is more complex than forward:
- Gradients w.r.t. LoRA A and B require the intermediate `x @ A`
- If not saved in forward, must recompute (activation checkpointing trade-off)
- Gradient w.r.t. `x` flows through both base `W` and LoRA `A @ B`

Options:
- **Save intermediates**: straightforward but uses memory
- **Recompute in backward**: saves memory, costs FLOPs
- **Fused backward**: single kernel computes all gradients (hardest to implement)

## Open Questions

1. At what LoRA rank does fusion stop being profitable vs. separate cuBLAS calls?
2. Can we fuse gate + up projections (they share the same input `x`) into a single kernel?
3. Is the SwiGLU activation worth fusing into the matmul kernel, or is it better as a separate pointwise kernel?
4. How does the optimal tiling change across GPU architectures (A100 vs H100)?
5. What's the memory-compute trade-off for saving vs. recomputing LoRA intermediates in backward?

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 (tiling strategy reference)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn, 2024 (BSD-2-Clause)
- [Unsloth](https://github.com/unslothai/unsloth) — Daniel Han-Chen, 2023 (Apache-2.0)
- Detailed code analysis: `docs/artifacts/ANALYSIS.md`

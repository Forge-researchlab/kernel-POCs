# Changelog

All notable changes to the LoRA MLP kernel experiments are documented here.

Format: each version entry records the algorithmic approach, key results, and what motivated the next version. Upgrades within a version are listed as sub-entries.

---

## [Unreleased]

### 2026-05-23 — Research & Baseline Analysis

#### Research: Unsloth Code Analysis

Analyzed Unsloth's `LoRA_MLP` autograd.Function and `matmul_lora()` from GitHub.
Key findings:

- Unsloth fuses the MLP at the **PyTorch autograd level**, not at the GPU kernel level
- Each `matmul_lora()` call makes 3 cuBLAS launches: `X@W`, `X@A`, `addmm_(XA, B)`
- Full forward is **10 kernel launches** (3 per projection × 3 projections + 1 Triton SwiGLU)
- Input `X` is read from HBM **4 times** (gate W, gate A, up W, up A)
- `X@A` intermediates (tiny, shape `[B*S, r]`) are materialized to HBM unnecessarily
- SwiGLU intermediates `e`, `g` (large, shape `[B*S, I]`) round-trip through HBM
- Backward uses clever in-place buffer overwriting via Triton kernel (3 tensors reused)

Code saved to `docs/artifacts/unsloth/` with annotated analysis in `docs/artifacts/ANALYSIS.md`.

#### Improvement Direction Defined

Three axes of improvement over Unsloth, each mapping to a kernel version:

| Axis | Version | What | Launches |
|------|---------|------|----------|
| A: Fuse LoRA into matmul | v1 | `X@W + s*(X@A)@B` in single Triton kernel | 3+1=4 |
| B: Gate+Up+SwiGLU fusion | v2 | Load X once, both projections + activation | 1+1=2 |
| C: Full MLP forward | v3 | v2 + v1(down) in autograd.Function | 2 |

#### Reference Implementation Created

- `reference/lora_mlp_pytorch.py`: clean PyTorch reference with 3 levels:
  1. `matmul_lora()` — single projection (for v1 testing)
  2. `lora_swiglu_mlp()` — full MLP forward (for v2/v3 testing)
  3. `LoRAMLP` — autograd.Function with forward + backward
- Backward passes `torch.autograd.gradcheck` in fp64
- No external dependencies (no bitsandbytes, no Triton)

#### Docs Updated

- `docs/research.md` — reframed around Unsloth as primary baseline, 3 concrete improvement axes
- `docs/benchmarks.md` — Unsloth `matmul_lora` and `LoRA_MLP` added as explicit baselines

---

## [v1] — 2026-05-23

### Approach
Single-projection fused LoRA matmul: `Y = X @ W^T + s * (X @ A^T) @ B^T` in one Triton kernel.
Fused K-loop reads X once from HBM, simultaneously accumulating both the base matmul and `X @ A^T`.

### Changes
- Implemented `fused_lora_matmul()` Triton kernel with output-stationary tiling
- Fused K-loop: X loaded once, used for both W and A products
- fp32 accumulation with `input_precision="ieee"` for fp32 inputs, native bf16 tensor cores for bf16
- Autotune over 6 tile configurations
- 28 correctness tests passing (fp32, bf16, multiple ranks/shapes, LLaMA-scale)
- Full benchmark sweep saved to `benchmarks/results/`

### Results (bf16, LLaMA-8B scale)

Per-projection (M=8192, N=14336, K=4096):

| Rank | PyTorch (ms) | Triton v1 (ms) | Speedup |
|------|-------------|----------------|---------|
| 8 | 4.67 | 6.73 | 0.69x |
| 16 | 4.68 | 6.94 | 0.67x |
| 32 | 4.79 | 7.61 | 0.63x |
| 64 | 4.79 | 9.50 | 0.50x |

Full MLP forward (batch=4, seq=2048):

| Rank | PyTorch (ms) | Triton v1 (ms) | Speedup |
|------|-------------|----------------|---------|
| 8 | 14.22 | 19.91 | 0.71x |
| 16 | 14.18 | 21.60 | 0.66x |
| 64 | 14.30 | 28.17 | 0.51x |

### Limitations
- **Base matmul ~0.7x cuBLAS**: the Triton matmul needs more tuning (tile sizes, L2 swizzle, software pipelining) to match cuBLAS
- **LoRA overhead grows with rank**: the LoRA A dot inside the fused K-loop adds registers/SRAM pressure
- **No kernel launch savings yet**: still 3+1 launches (same as Unsloth's 3 cuBLAS + 1 SwiGLU) since SwiGLU isn't fused
- **Forward only**: no backward pass implemented

### Key Finding
The fused K-loop architecture is correct and reads X only once (vs Unsloth's 2x for base+LoRA). The bottleneck is the base matmul speed vs cuBLAS. v2 should show gains from gate+up fusion (halving X reads and eliminating e/g intermediates) which cuBLAS cannot do.

---

## [v3] — 2026-05-23

### Approach
cuBLAS for all heavy matmuls + Triton kernel that fuses LoRA addition + SwiGLU into a single bandwidth-bound epilogue. No Triton tiled matmul — completely avoids the 0.73x cuBLAS penalty.

### Changes
- `fused_lora_swiglu()`: Triton kernel that reads cuBLAS outputs (e_base, g_base), adds LoRA via tiny `tl.dot` in registers, applies SiLU and multiply, writes h
- `lora_mlp_v3()`: full MLP = cuBLAS base matmuls + cuBLAS skinny LoRA + Triton fused LoRA+SwiGLU + cuBLAS down
- Key insight: use each tool for what it's best at — cuBLAS for matmuls, Triton for custom fusion
- Eliminates Unsloth's `addmm_` on e/g (saves ~900 MB HBM traffic at LLaMA-8B scale)
- Rank-independent performance — r=64 is same speed as r=8 (no register pressure)
- 52 tests passing (45 v1/v2 + 7 v3)

### Results (bf16, LLaMA-8B, batch=4, seq=2048)

Cross-version comparison:

| Kernel | Time (ms) | vs Unsloth | Launches |
|--------|-----------|------------|----------|
| Unsloth LoRA_MLP | 14.10 | 1.00x | 10 |
| v1 (all Triton) | 20.51 | 0.69x | 4 |
| v2 (Triton gate+up + cuBLAS down) | 16.65 | 0.85x | 4 |
| **v3 (cuBLAS + Triton epilogue)** | **12.38** | **1.14x** | **8** |

Rank scaling:

| Rank | Unsloth (ms) | v3 (ms) | Speedup |
|------|-------------|---------|---------|
| 8 | 12.72 | 12.43 | 1.02x |
| 16 | 12.58 | 12.36 | 1.02x |
| 32 | 12.71 | 12.29 | 1.03x |
| 64 | 12.68 | 12.39 | 1.02x |

### Key Finding
The v1/v2 approach of replacing cuBLAS with Triton tiled matmuls was wrong — cuBLAS is too fast to beat. The winning strategy is to keep cuBLAS for all matmuls and use Triton only for custom fusion that cuBLAS can't do (LoRA addition + SwiGLU in one pass). This eliminates the `addmm_` HBM round-trips on e and g without paying any matmul speed penalty.

---

## [v2] — 2026-05-23

### Approach
Hybrid: gate+up+LoRA+SwiGLU fused into a single Triton kernel (1 launch), down projection via cuBLAS (2-3 launches). Loads X once from HBM for both projections, applies SiLU(gate) * up in registers, never materializes e or g.

### Changes
- `fused_gate_up_swiglu()`: single Triton kernel doing both gate and up projections with LoRA + SwiGLU
- `lora_mlp_v2()`: full MLP forward = Triton gate+up+SwiGLU + cuBLAS down+LoRA (inference)
- `LoRAMLPv2`: `torch.autograd.Function` with full backward pass (training)
- Single K-loop loads X once, accumulates gate matmul, up matmul, and both LoRA A products simultaneously
- SiLU and elementwise multiply done in fp32 registers before writing h to HBM
- Training mode: kernel also writes `e` and `g` for the backward pass
- Backward: cuBLAS for all matmuls + SwiGLU backward recomputing sigmoid
- Gradients for all 6 LoRA matrices (dA_gate, dB_gate, dA_up, dB_up, dA_down, dB_down) + dX
- Down projection delegated to cuBLAS (faster than our Triton matmul)
- `gradcheck` passes in fp64 (fp64 uses PyTorch fallback, Triton used for bf16/fp32)
- 45 tests passing (33 v1 + 7 v2 forward + 5 v2 backward)

### Results (bf16, LLaMA-8B, batch=4, seq=2048)

Full MLP:

| Kernel | Time (ms) | vs Unsloth | Launches |
|--------|-----------|------------|----------|
| Unsloth LoRA_MLP | 12.73 | 1.00x | 10 |
| v1 (3x Triton + SwiGLU) | 20.47 | 0.62x | 4 |
| **v2 (Triton gate+up+SwiGLU + cuBLAS)** | **16.44** | **0.77x** | **4** |

Gate+Up+SwiGLU sub-path:

| Kernel | Time (ms) | vs Unsloth | Launches |
|--------|-----------|------------|----------|
| Unsloth (6 cuBLAS + 1 SwiGLU) | 8.55 | 1.00x | 7 |
| v2 (1 fused Triton kernel) | 12.88 | 0.66x | 1 |

Rank scaling (full MLP):

| Rank | Unsloth (ms) | v2 (ms) | Speedup |
|------|-------------|---------|---------|
| 8 | 12.71 | 16.41 | 0.77x |
| 16 | 12.71 | 16.63 | 0.76x |
| 32 | 12.71 | 21.96 | 0.58x |

### Key Findings
- v2 is **24% faster than v1** at the full MLP level (16.4ms vs 20.5ms)
- The gate+up fusion saves ~4ms vs v1 by eliminating X re-reads and e/g intermediates
- But Triton's per-tile matmul is still 0.73x cuBLAS, limiting the overall gain
- The hybrid approach (Triton for structural fusion, cuBLAS for down projection) works well
- r=32+ causes register pressure with 4 accumulators (acc_gate, acc_up, xa_gate, xa_up)

### Limitations
- Still 0.77x Unsloth — the base Triton matmul bottleneck (0.73x cuBLAS) dominates
- r=64 causes extreme register pressure (4 large accumulators) — needs rank-adaptive dispatch
- Forward only, no backward pass
- Gate and up K-loops are sequential (could potentially interleave for better pipelining)

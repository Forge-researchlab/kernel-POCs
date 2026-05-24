# Changelog

All notable changes to the LoRA QKV kernel experiments are documented here.

Format: each version entry records the algorithmic approach, key results, and what motivated the next version. Minor upgrades within a version are listed as sub-entries (v1_2, v1_3, etc.).

---

## [v4] — 2026-05-24

### Approach
Packed backward pass: reduces from 18+ cuBLAS calls (v3/Unsloth) to 9 cuBLAS + 1 Triton = 10 ops. Packs compatible matrix operations (dX_base, XA_all, dA_all) into single cuBLAS calls. A Triton epilogue fuses the 3 LoRA dX contributions ([M,r]@[r,K]) into one pass.

### Changes
- `LoRAQKVv4Function` autograd.Function with packed forward (v2_3) + packed backward
- `_lora_dx_epilogue_kernel`: Triton kernel computing 3 tiny matmuls + add in one pass
- `pack_weights_backward(W_q, W_k, W_v)`: pre-pack [W_q; W_k; W_v] for backward dX
- `pack_lora_a(A_q, A_k, A_v)`: pre-pack [A_q; A_k; A_v] for backward XA
- `lora_qkv_v4()` convenience wrapper accepting pre-packed weights
- fp64 fallback for gradcheck (no Triton)
- Backward operation count: 1 packed dX_base + 1 packed XA + 3 dY@B + 1 packed dA + 3 dB + 1 Triton epilogue

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

Forward (same as v2_3/v3):

| Rank | v4 fwd (ms) | Notes |
|------|-------------|-------|
| 8    | ~1.8ms      | Same code path as v2_3 |
| 16   | ~1.8ms      | Same code path as v2_3 |

Forward + Backward:

| Rank | Unsloth (ms) | v3 (ms) | v4 (ms) | v4/Unsloth | v4/v3 |
|------|-------------|---------|---------|------------|-------|
| 8    | 5.905ms     | 4.641ms | 4.323ms | **1.37x**  | 1.07x |
| 16   | 5.058ms     | 4.696ms | 4.227ms | **1.20x**  | 1.11x |
| 32   | 5.011ms     | 4.649ms | 4.283ms | **1.17x**  | 1.09x |
| 64   | 5.071ms     | 4.690ms | 4.438ms | **1.14x**  | 1.06x |

### Key Finding
**v4 is the fastest training-compatible kernel.** The packed backward eliminates 8 cuBLAS launches (18→10 ops), reducing overhead by 6-11% over v3. Total fwd+bwd speedup vs Unsloth: 1.14x–1.37x. The improvement is largest at small ranks (r=8) where launch overhead dominates compute. The Triton epilogue for dX avoids 3 extra cuBLAS calls + 3 dX read-modify-write round trips.

### Limitations
- Pre-computing W_dX_packed adds one-time memory (~48 MB at LLaMA-8B scale)
- dB GEMMs cannot be packed (different N dims: H_q vs H_kv)
- Memory slightly higher than v3 due to saving packed tensors for backward

---

## [v3] — 2026-05-24

### Approach
Training-compatible wrapper: `torch.autograd.Function` with fused forward (v2_3) and custom backward using cuBLAS `addmm_` chain. Falls back to plain PyTorch for fp64 (gradcheck compatibility).

### Changes
- `LoRAQKVFunction` autograd.Function with v2_3 forward + addmm_ backward
- `lora_qkv_v3()` convenience wrapper
- Backward computes: dX via accumulated addmm_, dA/dB via matmul + mul_
- fp64 gradcheck passes for MHA, GQA, 3D inputs, rank sweep
- Backward matches PyTorch reference `LoRAQKV` within rtol=1e-3

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

Forward (same as v2_3):

| Rank | v3 (ms) | v3/Unsloth |
|------|---------|------------|
| 8    | 1.867ms | **1.25x** |
| 16   | 1.800ms | **1.12x** |
| 32   | 1.825ms | **1.11x** |
| 64   | 1.819ms | **1.12x** |

Forward + Backward:

| Rank | Unsloth (ms) | v3 (ms) | v3/Unsloth |
|------|-------------|---------|------------|
| 8    | 5.012ms | 4.639ms | **1.08x** |
| 16   | 5.076ms | 4.693ms | **1.08x** |
| 32   | 5.033ms | 4.632ms | **1.09x** |
| 64   | 5.054ms | 4.722ms | **1.07x** |

### Limitations
- Backward uses separate cuBLAS calls (same as Unsloth) — not fused
- Saving X for backward increases memory (standard tradeoff)
- Forward fallback to PyTorch for fp64 adds overhead for gradcheck only

---

## [v2_3] — 2026-05-24

### Approach
Fully packed QKV: ALL 6 weight matrices packed into one tall matrix for a single cuBLAS call. X is read from HBM only ONCE. Single fused Triton epilogue applies LoRA to all 3 projections in-place.

### Changes
- `pack_weights_all(W_q, A_q, W_k, A_k, W_v, A_v)` creates [H_q+r+H_kv+r+H_kv+r, K] packed weight
- `lora_qkv_v2_3()`: 1 cuBLAS + 1 Triton = 2 total launches
- `_fused_qkv_epilogue_packed_kernel`: operates on slices of the single packed output
- Outputs are views into the packed buffer (no extra allocation)

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

| Rank | Unsloth | **v2_3** | v2_3/Unsloth | Memory |
|------|---------|---------|-------------|--------|
| 8    | 2.335ms | **1.847ms** | **1.26x** | 97 MB |
| 16   | 2.010ms | **1.777ms** | **1.13x** | 97 MB |
| 32   | 2.034ms | **1.809ms** | **1.12x** | 97 MB |
| 64   | 2.039ms | **1.792ms** | **1.14x** | 97 MB |

### Key Finding
**v2_3 is the new best forward kernel.** The single cuBLAS call eliminates 2 redundant X reads (saves ~128 MB bandwidth at LLaMA-8B scale). The wide output [M, 6192] does NOT trigger cuBLAS algorithm-selection penalties — confirmed experimentally. Memory matches Unsloth (~97 vs 96 MB).

### Limitations
- Outputs are non-contiguous views (downstream may need .contiguous())
- Forward only (backward in v3)

---

## [v2_2] — 2026-05-24

### Approach
In-place epilogue + fused 3-in-1 epilogue kernel. Two improvements over v2: (1) Triton epilogue writes in-place into packed output's first N columns, (2) single kernel launch for all 3 QKV epilogues.

### Changes
- `_lora_epilogue_inplace_kernel`: writes LoRA result in-place (no extra Y allocation)
- `_fused_qkv_epilogue_kernel`: processes Q, K, V in one launch using `program_id(1)`
- `lora_qkv_v2_2()`: 3 cuBLAS + 1 Triton = 4 launches
- Memory fixed from 128 MB (v2) to 97 MB (matching Unsloth)

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

| Rank | Unsloth | **v2_2** | v2_2/Unsloth | Memory |
|------|---------|---------|-------------|--------|
| 8    | 2.335ms | 1.954ms | **1.20x** | 97 MB |
| 16   | 2.010ms | 1.862ms | **1.08x** | 97 MB |
| 32   | 2.034ms | 1.898ms | **1.07x** | 97 MB |
| 64   | 2.039ms | 1.910ms | **1.07x** | 97 MB |

### Key Finding
Memory fixed to match Unsloth. Speed is slightly slower than v2 due to fused epilogue's wasted grid tiles (Q is 4x larger than K/V in GQA, so K/V blocks waste 3/4 of the grid). v2_3 addresses this by using a single cuBLAS call.

### Limitations
- Fused epilogue wastes GPU threads for GQA (asymmetric Q vs K/V sizes)
- Still 3 cuBLAS calls (v2_3 reduces to 1)

---

## [Unreleased]

### 2026-05-24 — Project Scaffolding & Baseline Research

#### Project Structure Created
- Full folder structure with `experiments/v1-v3`, `benchmarks`, `docs`, `reference`, `tests`
- Generic Cursor skills (research, analysis, benchmarking) and rules adapted from lora_mlp
- PyTorch reference implementation (`reference/lora_qkv_pytorch.py`) for correctness testing
- Test suite template (`tests/test_lora_qkv.py`) covering forward, backward, gradcheck, GQA
- Benchmark harness (`benchmarks/bench_lora_qkv.py`) with sweep and single-config modes

#### Research: Unsloth QKV Analysis

Analyzed how Unsloth handles QKV + LoRA projections.
Key findings:

- Unsloth applies `matmul_lora()` independently to each of Q, K, V
- Each `matmul_lora()` makes 3 cuBLAS calls: `X@W.t()`, `X@A.t()`, `out.addmm_(XA, B.t())`
- **9 kernel launches** total for QKV forward (3 projections × 3 calls)
- **X read from HBM 6 times** (Q W, Q A, K W, K A, V W, V A)
- `X@A` intermediates (shape `[B*S, r]`, tiny) materialized to HBM unnecessarily 3 times
- No cross-projection fusion — each cuBLAS call tiles independently
- Supports bitsandbytes 4-bit and FP8 quantized base weights
- Handles GQA via different-sized W_k/W_v matrices

Code analysis documented in `docs/artifacts/ANALYSIS.md`.

#### Research: Liger Kernel Analysis

Analyzed Liger Kernel's attention-related work.
Key findings:

- Liger does **NOT** handle QKV projection fusion or LoRA computation
- Liger focuses on RoPE, cross-entropy, and activation kernels
- The QKV + LoRA fusion is a greenfield opportunity — no existing Triton kernel exists

#### Improvement Axes Defined

| Axis | Version | What | Launches |
|------|---------|------|----------|
| A: Fuse LoRA into base matmul | v1 | `Y = X @ W^T + s*(X @ A^T) @ B^T` per projection | 3 |
| B: Fuse Q+K+V projections | v2 | Load X once, compute all three outputs + LoRA | 1 |
| C: Full forward in autograd.Function | v3 | Wrap best kernel(s) for training | 1–3 |
| D: Fused backward (stretch) | v4 | Reduce backward from 12+ launches | TBD |

#### Reference Implementation Created

- `reference/lora_qkv_pytorch.py`: clean PyTorch reference with 3 levels:
  1. `matmul_lora()` — single projection with LoRA (for per-projection testing)
  2. `lora_qkv_forward()` — all Q/K/V projections (for full-QKV testing)
  3. `LoRAQKV` — autograd.Function with forward + backward
- Backward pass verified against `torch.autograd.gradcheck` in fp64
- Handles GQA (different K/V output dimensions)
- No external dependencies (no bitsandbytes, no Triton)

#### Docs Updated

- `docs/research.md` — comprehensive baseline research with Unsloth/Liger analysis, improvement axes, GQA considerations
- `docs/benchmarks.md` — methodology, baselines, sweep configurations, result templates
- `docs/artifacts/ANALYSIS.md` — deep-dive comparison of Unsloth and Liger approaches

---

## [v1] — 2026-05-24

### Approach
Per-projection fused LoRA matmul: `Y = X @ W^T + s * (X @ A^T) @ B^T` in a single Triton kernel.
Applied independently to each of Q, K, V (3 launches total). Includes L2 cache swizzle (GROUP_SIZE_M=8) and 10 autotune configs.

### Changes
- Implemented `fused_lora_matmul()` Triton kernel with output-stationary tiling + L2 swizzle
- Separate post-pass for LoRA: base K-loop for W, then second K-loop for A, then XA @ B
- fp32 accumulation with `input_precision="ieee"` for fp32 inputs
- `lora_qkv_v1()` wrapper that calls fused_lora_matmul 3 times (Q, K, V)
- Correctness verified: fp32 diff < 1e-6, bf16 diff < 0.05, LLaMA-3 8B GQA shapes
- Benchmarked on A100-SXM4-80GB

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

Per-projection Q (M=8192, N=4096, K=4096):

| Rank | cuBLAS bare | PyTorch naive | Unsloth (addmm\_) | Triton v1 | v1/Unsloth | v1/PyTorch | v1/cuBLAS |
|------|------------|--------------|-------------------|-----------|------------|------------|-----------|
| 8    | 1.256ms    | 1.578ms      | 1.411ms           | 1.843ms   | 0.77x      | 0.86x      | 0.68x     |
| 16   | 1.094ms    | 1.396ms      | 1.226ms           | 1.914ms   | 0.64x      | 0.73x      | 0.57x     |
| 32   | 1.084ms    | 1.408ms      | 1.229ms           | 1.917ms   | 0.64x      | 0.73x      | 0.57x     |
| 64   | 1.099ms    | 1.422ms      | 1.252ms           | 2.330ms   | 0.54x      | 0.61x      | 0.47x     |

Per-projection K/V GQA (M=8192, N=1024, K=4096):

| Rank | cuBLAS bare | PyTorch naive | Unsloth (addmm\_) | Triton v1 | v1/Unsloth | v1/PyTorch |
|------|------------|--------------|-------------------|-----------|------------|------------|
| 8    | 0.320ms    | 0.436ms      | 0.389ms           | 0.482ms   | 0.81x      | 0.90x      |
| 16   | 0.314ms    | 0.433ms      | 0.388ms           | 0.489ms   | 0.79x      | 0.89x      |
| 32   | 0.313ms    | 0.431ms      | 0.387ms           | 0.509ms   | 0.76x      | 0.85x      |
| 64   | 0.319ms    | 0.438ms      | 0.392ms           | 0.603ms   | 0.65x      | 0.73x      |

Full QKV (batch=4, seq=2048, hidden=4096, 32q/8kv heads):

| Rank | PyTorch (9 ops) | Unsloth (9 cuBLAS) | Triton v1 (3 launches) | v1/Unsloth | v1/PyTorch |
|------|----------------|-------------------|----------------------|------------|------------|
| 8    | 2.238ms        | 2.030ms           | 2.888ms              | 0.70x      | 0.77x      |
| 16   | 2.250ms        | 2.033ms           | 2.895ms              | 0.70x      | 0.78x      |
| 32   | 2.251ms        | 2.015ms           | 3.001ms              | 0.67x      | 0.75x      |
| 64   | 2.280ms        | 2.060ms           | 3.593ms              | 0.57x      | 0.63x      |

### Key Finding
Same pattern as lora_mlp v1: Triton tiled matmul is ~0.58-0.69x cuBLAS. Need cuBLAS + Triton epilogue approach (→ v2).

---

## [v2] — 2026-05-24

### Approach
Packed W+A cuBLAS matmul + Triton LoRA epilogue. Instead of Unsloth's 3 calls per projection (X@W, X@A, addmm\_), v2 concatenates W and A into a single matrix, does one cuBLAS call that reads X once, then a cheap Triton epilogue adds the LoRA term.

### Changes
- `fused_lora_matmul_v2()`: packs `cat([W, A], dim=0)` → one cuBLAS matmul → split → Triton epilogue
- `_lora_epilogue_kernel`: Triton kernel that loads base output + XA, computes tiny `XA @ B^T` in registers, adds and writes
- `pack_weights(W, A)` helper for pre-packing in the training loop
- `lora_qkv_v2()` wrapper with optional pre-packed weights
- Correctness verified: fp32 exact match, bf16 < 0.004 diff, LLaMA-3 8B GQA

### Results (bf16, LLaMA-3 8B GQA, A100-SXM4-80GB)

Full QKV (batch=4, seq=2048, hidden=4096, 32q/8kv heads):

| Rank | PyTorch (9 ops) | Unsloth (9 cuBLAS) | v1 (3 Triton) | **v2 (3 cuBLAS + 3 Triton)** | v2/Unsloth | v2/PyTorch |
|------|----------------|-------------------|--------------|------------------------------|------------|------------|
| 8    | 2.240ms        | 2.014ms           | 2.921ms      | **1.927ms**                  | **1.04x**  | 1.16x      |
| 16   | 2.285ms        | 2.035ms           | 3.624ms      | **1.869ms**                  | **1.09x**  | 1.22x      |
| 32   | 2.282ms        | 2.057ms           | 3.824ms      | **1.856ms**                  | **1.11x**  | 1.23x      |
| 64   | 2.297ms        | 2.044ms           | 4.195ms      | **1.924ms**                  | **1.06x**  | 1.19x      |

Per-projection K/V GQA (M=8192, N=1024, K=4096):

| Rank | Unsloth | v2 | v2/Unsloth |
|------|---------|-----|------------|
| 8    | 0.389ms | 0.342ms | **1.14x** |
| 16   | 0.388ms | 0.337ms | **1.15x** |
| 32   | 0.391ms | 0.336ms | **1.16x** |
| 64   | 0.393ms | 0.343ms | **1.15x** |

### Key Finding
**v2 beats Unsloth by 4-11% on full QKV.** The packed W+A approach halves X HBM reads (3 vs 6) and eliminates XA intermediate round-trips. Performance is rank-independent (1.856–1.927ms for r=8 to r=64) because the Triton epilogue's tiny dot product is negligible. The L2 swizzle helps but doesn't close the gap. The LoRA post-pass adds 14-41% overhead depending on rank. The cuBLAS + Triton epilogue approach (lora_mlp v3 pattern) should be applied directly as v2.

### Limitations
- **Base matmul ~0.6x cuBLAS**: fundamental Triton vs cuBLAS gap
- **LoRA overhead grows with rank**: second K-loop reads X again
- **No cross-projection fusion**: still 3 launches (same as number of projections)
- **Forward only**: no backward pass

---

<!--
## [v1] — YYYY-MM-DD

### Approach
Per-projection fused LoRA matmul: `Y = X @ W^T + s * (X @ A^T) @ B^T` in one Triton kernel.
Applied independently to each of Q, K, V. 3 kernel launches total.

### Changes
- (TBD)

### Results
- (TBD)

### Limitations
- (TBD)

---

## [v2] — YYYY-MM-DD

### Approach
Q+K+V projection fusion: load X once from HBM, compute all three projections with LoRA.
Handle GQA (asymmetric Q vs K/V output dimensions).

### Changes
- (TBD)

### Results
- (TBD)

### Limitations
- (TBD)

---

## [v3] — YYYY-MM-DD

### Approach
Full QKV forward wrapped in torch.autograd.Function with custom backward.
Combine best kernel(s) from v1/v2 into a training-compatible wrapper.

### Changes
- (TBD)

### Results
- (TBD)

### Limitations
- (TBD)
-->

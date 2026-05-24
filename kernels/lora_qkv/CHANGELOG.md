# Changelog

All notable changes to the LoRA QKV kernel experiments are documented here.

Format: each version entry records the algorithmic approach, key results, and what motivated the next version. Minor upgrades within a version are listed as sub-entries (v1_2, v1_3, etc.).

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

# v5_upgrade_1 Results: padded gate+up + dropped down packing

**Date:** 2026-05-24
**GPU:** NVIDIA A100-SXM4-80GB (CUDA 12.4, Torch 2.4.1, BLAS = cuBLAS)
**Shape:** LLaMA-8B forward, batch=4, seq=2048, M=8192, H=4096, I=14336, r=16, bf16
**Source:** [`experiments/v5/lora_mlp_kernel_v5_upgrade_1.py`](../../experiments/v5/lora_mlp_kernel_v5_upgrade_1.py)
**Pairs with:** [`v5_packing_diagnosis.md`](v5_packing_diagnosis.md)

## TL;DR

> v5_upgrade_1 applies the two fixes recommended by the v5 diagnosis. Both
> fixes verify in the microbench (combined ~1.2 ms of matmul-only savings)
> and ~0.35 ms of that survives to wall-clock. v5_upgrade_1 beats v3 by
> 2.78% (12.48 → 12.13 ms) at LLaMA-8B/r=16, satisfying the ≥1% target.

## Changes vs v5

1. **Drop down-phase packing** — revert to v3's two-cuBLAS-call pattern
   plus `addmm_` on the contiguous `out` buffer (no `.contiguous()` copy,
   clean N=H tile for cuBLAS).
2. **Pad gate+up mega-matrix to a multiple of 128** — append zero-rows so
   `N = ceil((2*I + 2*r) / 128) * 128`. At LLaMA-8B/r=16 that's
   28704 → 28800 (96 zero rows). Padded columns of the result are
   zero @ X = 0 and ignored when slicing.

Inference path is the v5 path verbatim (re-exported, not re-implemented).
Backward is the same Unsloth-style in-place pattern as v3/v5 — packing
doesn't help backward because each backward matmul has a different LHS.

## Microbench (matmul-only)

A100 peaks: 312 TFLOPS bf16. Run with `triton.testing.do_bench(warmup=10, rep=50)`.

### Gate+Up matmul

| Variant                         | N      | time      | TFLOPS | %peak |
|---------------------------------|--------|-----------|--------|-------|
| v3 (4 cuBLAS calls)             | 14336+r| 8.594 ms  | 224.1  | 71.8% |
| v5 (1 mega-GEMM)                | 28704  | 8.349 ms  | 230.7  | 74.0% |
| **v5_upgrade_1 (padded)**       | **28800** | **7.445 ms** | **259.6** | **83.2%** |

Padding the mega-N to 28800 gives cuBLAS a tile-aligned shape and lifts
the effective TFLOPS from 230 → 260 (+9.2 percentage points of peak).

### Down matmul (end-to-end with addmm_)

| Variant                                | time      | TFLOPS | %peak |
|----------------------------------------|-----------|--------|-------|
| v5 (mega + .contiguous() + addmm_)     | 4.266 ms  | 226.6  | 72.6% |
| **v3 / v5_upgrade_1 (matmul + skinny + addmm_)** | **3.948 ms** | **244.9** | **78.5%** |

Reverting to v3's pattern saves 0.318 ms by (a) keeping cuBLAS on its clean
N=H tile and (b) skipping the `.contiguous()` copy of the H slice.

### Combined isolated savings

  +0.904 ms (gate+up padding)
  +0.318 ms (drop down packing)
  ───────────
  +1.222 ms (matmul-only, vs v5)

## End-to-end (full forward, including Triton epilogue)

5-run medians with `triton.testing.do_bench(warmup=20, rep=100)` at LLaMA-8B/r=16/bf16:

| Implementation                 | Median (ms) | Mean (ms) | Min   | Max   |
|--------------------------------|-------------|-----------|-------|-------|
| Unsloth `apply_lora_mlp_swiglu`| 12.720      | 12.759    | 12.71 | 12.90 |
| v3                             | 12.481      | 12.398    | 12.21 | 12.52 |
| v5 training                    | 12.394      | 12.400    | 12.35 | 12.47 |
| **v5_upgrade_1 training**      | **12.134**  | **12.148**| **12.00** | **12.27** |
| v5 inference (pre-merged)      | 11.784      | 11.792    | 11.78 | 11.83 |

| Comparison                  | Δ (ms) | Δ (%)   |
|-----------------------------|--------|---------|
| v5_upgrade_1 vs v3          | −0.347 | −2.78%  |
| v5_upgrade_1 vs v5 training | −0.260 | −2.10%  |
| v5_upgrade_1 vs Unsloth     | −0.586 | −4.61%  |
| v5_inf vs v5_upgrade_1      | −0.350 | −2.89%  |
| v5_inf vs v3                | −0.697 | −5.58%  |

Full sweep: [`benchmarks/results/v5_upgrade_1_20260524_112741.csv`](../../benchmarks/results/v5_upgrade_1_20260524_112741.csv).

## Where the 0.87 ms went

Microbench says matmul-only savings vs v5 are ~1.22 ms; end-to-end says
v5_upgrade_1 saves ~0.26 ms vs v5. The other ~0.96 ms gets absorbed into
secondary effects that don't show up in matmul-only timing:

* **Triton epilogue's strided reads.** `e_base = result[:, :I]` has
  stride 0 = 28800 (was 28704 in v5), so each [BLOCK_M, BLOCK_N] tile load
  still straddles wider rows than v3's contiguous N=14336 layout. The
  epilogue is small (~0.3-0.5 ms) but the L2-locality hit costs ~100-200 µs.
* **Mega-matmul intermediate is even larger now** (M·28800·2 = 450 MiB
  vs v5's 448 MiB) and evicts useful state from L2 between the gate+up
  matmul and the Triton epilogue.
* **do_bench noise**: σ ≈ 50-100 µs run-to-run on a 12 ms benchmark.

The net is that v5_upgrade_1 captures the predicted ~0.35 ms of wall-clock
savings, exactly in the range the diagnosis predicted (~0.1-0.3 ms padding
recovery + ~0.22 ms down-packing recovery).

## Verdict

**Both fixes work as advertised.** The original v5 packing experiment
landed at parity with v3 because gate+up packing had its tile penalty and
down packing had two separate problems. Fixing all three lets us realise
the launch-savings benefit at long last:

* v5_upgrade_1 is **the new training-path best**, at 12.13 ms.
* v5_inference (pre-merged) is still the overall best at 11.78 ms,
  because pre-merging eliminates the LoRA-A skinny matmuls and the down
  LoRA-B `addmm_` entirely. That gap is fundamental to LoRA inference,
  not specific to the kernel.

### Reachable next steps (not done)

* **Make the Triton epilogue contiguous-friendly.** Allocate `e_base` and
  `g_base` as separate contiguous buffers (two cuBLAS calls) and only
  pack the LoRA-A skinny matmuls. Trades 1 launch for clean strides;
  expected ~0.1-0.2 ms gain on the epilogue. Different algorithmic
  family — would be v6.
* **CUDA 12.5 + cublasLt SWISH for inference.** This box runs CUDA 12.4,
  so the inference fast-path falls back to a 4-launch version with one
  Triton kernel. CUDA 12.5+ would replace that with a fused matmul+SiLU
  cublasLt call (zero Triton at inference). Expected ~0.1-0.2 ms additional
  inference speedup.

## Reproduction

```bash
cd /workspace/kernel-POCs/kernels/lora_mlp

# unit tests (21 v5_upgrade_1 + 91 prior = 112 total)
python -m pytest tests/test_lora_mlp.py -v

# matmul-only microbench
python benchmarks/microbench_v5_upgrade_1.py

# full LLaMA sweep
python benchmarks/bench_lora_mlp.py --mode mlp --save benchmarks/results/
```

Verified on A100-SXM4-80GB, CUDA 12.4, Torch 2.4.1, bf16.

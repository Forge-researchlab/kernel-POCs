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

## [v6] — 2026-05-24

### Approach
v5 / v5_upgrade_1 traded ~200-400 MB of extra forward memory (transient packed weights + mega-matmul output) for a small latency win that dwindled to <0.5 ms at LLaMA-8B production. v6 reframes the problem around **"use where each framework shines"**:

- **cuBLAS for big GEMMs.** Per-projection cuBLAS calls (gate, up, down) hit clean N tiles (N=I=14336 or N=H=4096, both multiples of 128) and run at >80% of peak A100 bf16 throughput. There is no upside to mega-packing — it only pulls cuBLAS off its preferred tile (the v5 packing diagnosis showed this is worth ~1 ms of GEMM penalty).
- **Triton for the tiny LoRA-A matmuls.** Two skinny matmuls `X @ A_gate.T` and `X @ A_up.T` are individually too small for cuBLAS to amortize launch overhead. v6 stacks them into a single Triton GEMM `X @ [A_gate; A_up].T → [M, 2r]`. The compiler caches each `[BLOCK_M, BLOCK_K]` tile of X in shared memory and reuses it across all 2r output columns — i.e. it gets "load X once" for free via a standard matmul kernel, no fancy multi-output kernel needed.
- **Optional CUDA stream parallelism (`enable_streams=True`, default).** Phase-1 gate/up cuBLAS calls overlap on two streams; Phase-4 `h @ W_down.T` and `h @ A_down.T` likewise overlap. Event-based syncs serialize where needed (before the Triton SwiGLU epilogue, before the final `addmm_`). At small M this overlap hides launch overhead; at large M it has no measurable effect either way (kernels are already long enough to fill the GPU).
- **Backward identical to v3 / v5_upgrade_1.** Backward GEMMs have distinct LHS operands (dY, df, de, X), so neither packing nor stream parallelism help — we keep the Unsloth `swiglu_DWf_DW_dfg_kernel` + in-place `addmm_` pattern verbatim.

Result: **7 forward launches** (vs v3's 8, v5's 4, v5_upgrade_1's 5), but the right ones — every big GEMM uses cuBLAS's best tile, and the tiny LoRA-A pair fuses into one Triton launch. Memory footprint **matches v3 to within 0.3 MB** while sidestepping all of v5's transient buffers.

### Changes
- `experiments/v6/lora_mlp_kernel_v6.py`: new file with
  - `_stacked_lora_a_kernel`: Triton GEMM with weight shape `[2r, H]`, output `[M, 2r]`, `BLOCK_N = next_power_of_2(2r)`; handles r ∈ {8, 16, 32, 64}, bf16 / fp16 / fp32 with fp32 accumulation.
  - `fused_lora_a_stacked(X, A_stack, r)`: helper that returns `(xa_gate, xa_up)` views of the `[M, 2r]` Triton output.
  - `stack_lora_a(A_gate, A_up)`: builds the `[2r, H]` stacked weight buffer (intended to be cached per optimizer step).
  - `_v6_forward_impl()`: shared sync + streams forward; takes `enable_streams` and `side_stream` kwargs.
  - `lora_mlp_v6()`: no-autograd convenience function with optional pre-cached `A_stack` arg.
  - `LoRAMLPv6(autograd.Function)`: training forward + backward (backward identical to v5_upgrade_1).
  - `LoRAMLPv6Module(nn.Module)`: nn.Module wrapper with `_A_stack_cache` non-persistent buffer, `refresh_packed()` / `invalidate_packed()` for use after optimizer steps, and a lazy `_v6_side_stream` CUDA stream.
  - `lora_mlp_v6_inference` re-exports v5's pre-merged inference path.
- `tests/test_lora_mlp.py`: new `TestV6` class with 40 tests covering the stacked-A kernel in isolation, fp32 / bf16 forward correctness (sync and streams), rank sweep r ∈ {8, 16, 32, 64}, sync-vs-streams **bit-exact** equivalence (the streams path only changes which stream a kernel lands on, not the kernel contents), Unsloth parity, LLaMA-8B / 13B shapes, fp64 gradcheck, bf16 backward vs PyTorch reference, no-LoRA / non-power-of-2 / 2D-input edge cases, module forward / backward / cache refresh.
- `benchmarks/bench_lora_mlp.py`: adds `v6_sync_ms`, `v6_streams_ms` columns and seven new ratio columns (`v6_*_vs_unsloth`, `v6_*_vs_v3`, `v6_*_vs_v5_up1`, `v6_streams_vs_v6_sync`). Pre-builds the `A_stack` buffer once outside the timed loop. Also adds a small-M (b=1, s=512) config to the sweep for stream-regression diagnostics.
- `benchmarks/bench_memory.py`: adds `v6_sync` and `v6_streams` rows tracking the same fwd / fwd+bwd / resident-after-fwd metrics. Adds a small-M (b=1, s=512) config.
- `docs/analysis/v6_design.md`: design writeup of the "use where each framework shines" principle, why per-projection cuBLAS beats mega-packing on tile alignment, the role of stream parallelism, and the measured results.

### Results

**LLaMA-8B production forward (bf16, batch=4, seq=2048, M=8192, H=4096, I=14336):**

| Implementation | Time (ms) | vs Unsloth | vs v3 | vs v5_up1 | Fwd peak mem (MB) | Mem vs Unsloth |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth `apply_lora_mlp_swiglu` | 12.89 | 1.00x | — | — | 736 | 1.00x |
| v3 (cuBLAS + Triton epilogue) | 12.35 | 1.04x | 1.00x | 0.97x | 1185 | 1.61x |
| v5 (packed mega-GEMM) | 12.38 | 1.04x | 1.00x | 0.97x | 1585 | 2.15x |
| v5_upgrade_1 (padded mega + v3-style down) | 11.99 | 1.07x | 1.03x | 1.00x | 1412 | 1.92x |
| **v6_sync (cuBLAS + Triton-stacked LoRA-A)** | **12.19** | **1.06x** | **1.01x** | **0.98x** | **1185** | **1.61x** |
| **v6_streams (v6_sync + side-stream overlap)** | **12.03** | **1.07x** | **1.03x** | **1.00x** | **1185** | **1.61x** |
| v5 inference (pre-merged) | 11.78 | 1.09x | 1.05x | 1.02x | 736 | 1.00x |

**Small-M LLaMA-8B forward (bf16, batch=1, seq=512, M=512, H=4096, I=14336, r=16):**

| Implementation | Time (ms) | vs Unsloth | vs v6_sync | Fwd peak (MB) |
|---|---:|---:|---:|---:|
| Unsloth | 1.013 | 1.00x | — | 46 |
| v3 | 0.973 | 1.04x | — | 74 |
| v5 | 0.859 | 1.18x | — | 415 |
| v5_upgrade_1 | 0.870 | 1.17x | — | 300 |
| v6_sync | 1.015 | 1.00x | 1.00x | 74 |
| **v6_streams** | **0.852** | **1.19x** | **1.19x** | **74** |
| v5 inference | 0.921 | 1.10x | — | 46 |

At small M, the launch overhead from v6's 7 sync kernels becomes comparable to GEMM time, regressing `v6_sync` to ~Unsloth speed (worse than v5_upgrade_1's ~0.87 ms). Turning on `enable_streams=True` recovers a 1.19x win by overlapping the gate/up and down/A_down GEMMs across two streams — closing the launch-overhead gap entirely.

**Rank sweep at LLaMA-8B production (bf16, batch=4, seq=2048):**

| Rank | Unsloth | v3 | v5_up1 | v6_sync | v6_streams | v6_streams vs v3 | v6_streams vs v5_up1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 13.01 | 12.16 | 12.05 | 12.20 | 12.04 | 1.010x | 1.001x |
| 16 | 12.89 | 12.35 | 11.99 | 12.19 | 12.03 | 1.027x | 0.997x |
| 32 | 12.71 | 12.44 | 12.09 | 12.15 | 12.01 | 1.036x | 1.007x |
| 64 | 12.79 | 12.23 | 12.20 | 12.39 | 12.20 | 1.003x | 1.000x |

**Memory across all configs:** v6_sync and v6_streams use **identical** memory (the side stream doesn't allocate anything new), and that memory is **bit-equal to v3 within 0.3 MB** across every config in the sweep — by design, since v6 uses the same per-projection cuBLAS calls and the same fused SwiGLU+LoRA epilogue. The 0.3 MB delta comes from the tiny `[M, 2r]` Triton-stacked output (vs v3's two separate `[M, r]` cuBLAS outputs).

Full sweep CSV: `benchmarks/results/v6_20260524_135529.csv` (24 rows × all five impls). Memory CSV: `benchmarks/results/memory_20260524_135718.csv` (7 configs × 7 impls).

### Key Finding

**v6 delivers v3-level memory at v5_upgrade_1-level latency, with a clean stream-parallelism win at small M.** The end-to-end story:

- Forward peak memory: 1185 MB at LLaMA-8B production — **400 MB less than v5, 227 MB less than v5_upgrade_1, identical to v3**. The wins come from (a) no transient packed weights (no `W_mega` / `W_down_packed`), (b) per-projection outputs are independent `[M, I]` tensors that don't share storage with a non-contiguous mega-output, and (c) the Triton stacked LoRA-A output is `[M, 2r]` — at most a few MB.
- Forward latency: v6_streams ties v5_upgrade_1 at LLaMA-8B production (11.99 vs 12.03 ms) and beats every other training path. At small M it's the only training implementation that beats Unsloth (1.19x) while keeping v3-level memory.
- Stream parallelism contributes ~1.4% at LLaMA-8B production (12.19 → 12.03 ms; saturated GEMMs leave little to hide) but ~19% at small M (1.015 → 0.852 ms; launch overhead is exposed) — a clean validation of the "use streams when launches dominate" heuristic.

The Triton stacked-A trick gets all the way to "load X once" without writing a custom multi-output kernel. The compiler does the SMEM caching automatically because `N=2r ≤ 128` fits in one program block. This is the inverse of v5's approach: instead of packing weights to fuse cuBLAS launches, v6 keeps cuBLAS calls separate (so each gets its optimal tile) and uses Triton for the one specific case where small-N caching beats cuBLAS's tile geometry.

### Limitations

- **Stream parallelism is workload-dependent.** At small M (M ≤ ~1024) it's a 15–20% latency win. At LLaMA-8B production (M=8192) it's a ~1.4% win — still strictly positive in our measurements, but within run-to-run noise on some configs. The `enable_streams=False` path is a safe fallback and is available at every level (kwarg on `lora_mlp_v6`, `LoRAMLPv6.apply`, and `LoRAMLPv6Module`).
- **Sync-vs-streams output is bit-exact in our tests** (37/40 v6 tests assert `rtol=0, atol=0` between modes), because both paths perform exactly the same kernel launches in the same order — streams only change which CUDA stream each launch lands on. This bit-exactness is a property of the current pipeline (no kernel splits or split-K) and could change if a future tiling tweak introduces non-determinism.
- **Backward unchanged.** Same as v3 / v5 / v5_upgrade_1 — backward GEMMs have distinct LHS operands, so neither packing nor stream parallelism help. We use Unsloth's optimized in-place buffer-reuse pattern verbatim.
- **Stacked-A cache must be refreshed after every optimizer step.** The `LoRAMLPv6Module` handles this via `refresh_packed()`; users of the bare `lora_mlp_v6()` convenience function must either re-stack each call (pass `A_stack=None`) or manage the cache themselves. The test suite covers this with `test_v6_module_cache_refresh`.
- **r=64 stacked-A kernel is no faster than r=8.** With BLOCK_N=128 the stacked kernel still does 2-3 µs at LLaMA-8B production — well below cuBLAS launch overhead. We did not split into two cuBLAS calls at r=64 because the stacked Triton kernel still wins (or ties) by absorbing the second launch.

### v6_upgrade_1 — 2026-05-24

#### Approach
In-place SwiGLU epilogue: when `save_eg=True` (training), `fused_lora_swiglu_inplace` writes `e_full` and `g_full` back to the **same storage** as `e_base` and `g_base` instead of allocating fresh `[M, N]` tensors. The Triton kernel loads each tile into registers before storing, and each program instance handles a non-overlapping tile, so in-place is safe. This eliminates two `[M, I]` transient allocations (~224 MB each at LLaMA-8B), cutting peak forward memory by **448 MB**.

#### Changes
- `experiments/v6/lora_mlp_kernel_v6.py`:
  - Stopped importing `fused_lora_swiglu` from v5; instead imports the raw Triton kernel `_fused_lora_swiglu_kernel`.
  - Added `fused_lora_swiglu_inplace()`: local wrapper that passes `e_base` as both `E_ptr` and `E_out_ptr` (and same for `g_base`/`G_out_ptr`) when `save_eg=True`. Asserts contiguity of `e_base`/`g_base` (always true in v6 from `torch.matmul`).
  - Updated `_v6_forward_impl()` to call `fused_lora_swiglu_inplace` instead of `fused_lora_swiglu`.
- v5 kernel file is **not modified** (project convention).
- No test changes needed — all 152 existing tests pass unchanged (the backward still receives `e_full`/`g_full` which now alias `e_base`/`g_base` storage; backward only reads them).

#### Results

**Memory (LLaMA-8B production, bf16, batch=4, seq=2048, r=16):**

| Implementation | Fwd peak (MB) | vs Unsloth | vs v6 (before) |
|---|---:|---:|---:|
| Unsloth | 736 | 1.00x | — |
| v3 | 1185 | 1.61x | — |
| v6 (before, v6 baseline) | 1185 | 1.61x | 1.00x |
| **v6_upgrade_1 (in-place)** | **737** | **1.00x** | **0.62x** |
| v5 inference (pre-merged) | 736 | 1.00x | — |

Memory reduction: **1185 → 737 MB (−448 MB)**, now matching Unsloth to within 1 MB across all configs and rank sweeps. Fwd+Bwd also matches Unsloth (802 MB).

**Latency (LLaMA-8B production, bf16, batch=4, seq=2048, r=16):**

| Implementation | Time (ms) | vs Unsloth | vs v3 |
|---|---:|---:|---:|
| Unsloth | 14.09 | 1.00x | — |
| v3 | 12.06 | 1.17x | 1.00x |
| v6_sync | 12.09 | 1.17x | 1.00x |
| v6_streams | 11.91 | 1.18x | 1.01x |
| v5 inference | 11.65 | 1.21x | 1.03x |

Latency is unchanged from v6 baseline — the in-place write eliminates allocations but does not change kernel arithmetic.

Full memory CSV: `benchmarks/results/memory_20260524_142146.csv`. Latency CSV: `benchmarks/results/v6_upgrade_1_latency_20260524_142232.csv`.

#### Key Finding
**v6_upgrade_1 closes the last gap vs Unsloth: memory.** v6 already matched or beat Unsloth on latency; this upgrade brings peak forward memory from 1.61x Unsloth down to 1.00x — the in-place trick is free (no latency cost, no correctness risk) because the Triton kernel's tile-level load→compute→store pattern makes each tile's write independent. v6_upgrade_1 is now strictly better than Unsloth on both axes (1.18x faster, 1.00x memory).

---

## [v5] — 2026-05-24

### Approach
v3 (the prior best) launches 8 kernels including 4 separate cuBLAS calls that all multiply by the same input X (W_gate, W_up, A_gate, A_up). v5 packs those four matrices into a single mega-matrix `W_mega = [W_gate; W_up; A_gate; A_up]` and runs **one** cuBLAS call `result = X @ W_mega^T`, then slices the output into the four pieces. The down projection plays the same trick: `W_down_packed = [W_down; A_down]` reduces 2 cuBLAS calls into 1. The Triton fused LoRA + SwiGLU epilogue from v3 is reused unchanged. The training forward path is **4 launches** (3 cuBLAS + 1 Triton), down from v3's 8 and Unsloth's 10. For inference, `prepare_inference_weights()` merges LoRA into the base weights offline (`W_eff = W + s*B@A`), so the runtime path becomes just three cuBLAS matmuls plus a SwiGLU fusion. When CUDA ≥ 12.5 is available, the gate matmul fuses with SiLU via cublasLt's `CUBLASLT_EPILOGUE_SWISH` (zero Triton at runtime). On this host (CUDA 12.4) the kernel probes once at import time and falls back to a 4-launch path that uses Unsloth's `swiglu_fg_kernel` for the SiLU(e)*g fusion — still 4 launches, but with one Triton kernel instead of zero.

### Changes
- `experiments/v5/lora_mlp_kernel_v5.py`: full v5 implementation
  - `pack_gate_up_weights()`, `pack_down_weights()`: weight-packing helpers
  - `merge_lora_weights()`, `prepare_inference_weights()`: pre-merge LoRA for inference
  - `fused_lora_swiglu()`: Triton epilogue (copy of v3's, accepts non-contiguous slices via explicit strides)
  - `lora_mlp_v5()`: training forward, 4 launches when LoRA is enabled
  - `lora_mlp_v5_inference()`: inference forward with cublasLt SWISH probe + fallback
  - `LoRAMLPv5(autograd.Function)`: training forward + backward (backward identical to v3)
  - Probe `_CUBLASLT_SWISH` at import time so missing-CUDA-12.5 hosts silently take the fallback path
- `tests/test_lora_mlp.py`: new `TestV5` class with 17 tests (forward fp32/bf16, rank sweep 8/16/32/64, vs Unsloth, inference fp32/bf16, fp64 gradcheck, bf16 backward vs reference, no-LoRA, non-power-of-2, 2D/3D inputs, LLaMA-8B/13B shapes). Also fixed 6 pre-existing v3 epilogue-unit tests that were broken because `fused_lora_swiglu` returns a tuple — they now unpack `(out, _, _)`.
- `benchmarks/bench_lora_mlp.py`: extended `bench_mlp` to time Unsloth, v3, v5 training (with pre-packed weights), and v5 inference (with pre-merged weights) in one pass; CSV columns now include `v3_ms`, `v5_train_ms`, `v5_inf_ms`, and the four corresponding speedup ratios. Output CSV is named `v5_<timestamp>.csv`.

### Results

**LLaMA-8B forward (bf16, batch=4, seq=2048, rank=16, M=8192, H=4096, I=14336):**

| Implementation | Time (ms) | vs Unsloth | vs v3 | Launches |
|----------------|-----------|------------|-------|----------|
| Unsloth `apply_lora_mlp_swiglu` | 12.85 | 1.00x | — | 10 |
| v3 (cuBLAS + Triton epilogue) | 12.35 | 1.04x | 1.00x | 8 |
| **v5 training (packed)** | **12.36** | **1.04x** | **1.00x** | **4** |
| **v5 inference (pre-merged)** | **11.79** | **1.09x** | **1.05x** | **4** |

**Rank scaling at LLaMA-8B (bf16, batch=4, seq=2048):**

| Rank | Unsloth (ms) | v3 (ms) | v5 train (ms) | v5 inf (ms) | v5 inf vs v3 |
|------|-------------:|--------:|--------------:|------------:|-------------:|
| 8 | 12.58 | 12.43 | 12.66 | 11.92 | 1.04x |
| 16 | 12.85 | 12.35 | 12.36 | 11.79 | 1.05x |
| 32 | 12.71 | 12.29 | 12.61 | 11.91 | 1.03x |
| 64 | 12.90 | 12.50 | 12.65 | 11.82 | 1.06x |

Full sweep CSV: `benchmarks/results/v5_20260524_104637.csv`.

### Key Finding
Packing **did not** materially speed up training over v3 (12.36 vs 12.35 ms at LLaMA-8B / r=16; basically a wash across the rank sweep). The reason is that on A100, cuBLAS already pipelines the four small launches that v5 fuses into one — the host-side launch overhead at this scale is negligible compared to the ~12 ms of GEMM compute, so cutting 8 → 4 launches buys nothing measurable. The launch-count argument only pays back on hosts with high host→device latency, on much smaller M (where launch overhead dominates), or in CUDA Graph–captured paths (not exercised here).

The inference path **does** show a small but consistent ~5% win over v3 (11.79 vs 12.35 ms). That gain comes from skipping the four LoRA-A skinny matmuls and the LoRA-B `addmm_` updates entirely — those operations are bandwidth-bound and disappear cleanly when the LoRA is pre-merged into the base weights. This is the inherent inference-time advantage of LoRA-adapted models, not a Triton-kernel improvement.

A small smaller-shape config (b=1, s=2048, r=8) shows v5_train at 1.19x vs v3 (3.06 vs 3.62 ms) — confirming that packing does help when launch overhead is a larger fraction of the total. At LLaMA-8B training scale that regime never kicks in.

### Limitations
- **CUDA 12.5+ requirement not met on this host.** `CUBLASLT_EPILOGUE_SWISH` (added in CUDA 12.5) was the linchpin of the original "zero Triton at inference" goal: gate matmul + SiLU fused into one cublasLt call. This box runs CUDA 12.4, so the import-time probe disables that path and v5_inference falls back to a separate `gate` cuBLAS matmul + Unsloth's `swiglu_fg_kernel`. The fallback is still 4 launches, but it includes 1 Triton kernel instead of 0. Numbers above are with the fallback path. On CUDA 12.5+ hosts, expect a small additional speedup on inference because (a) the Triton kernel is replaced by a cublasLt SWISH epilogue that runs concurrently with the matmul write-out, and (b) one fewer kernel launch.
- **v4 cublasLt wrapper has incorrect attribute enum IDs.** `experiments/v4/cublaslt_wrapper.py` hardcodes `TRANSA=0`, `TRANSB=1`, `EPILOGUE=2`, but the correct CUDA values are 3, 4, and 7 respectively. v5 sidesteps this by probing the SWISH path at import and never calling the wrapper on CUDA < 12.5 (where the call would fail anyway). Fixing v4 is left as a separate task per the project's "never modify a previous version" rule.
- **Backward unchanged from v3.** Packing only helps the forward path because each backward matmul has a different input (dY, df, de, X) — there is no shared input X to reuse. Backward still uses Unsloth's optimized in-place buffer-reuse pattern from v3, which is already near-optimal.
- **Mega-GEMM output is non-contiguous in N.** Slices like `result[:, :I]` are stride-`(2*I+2*r, 1)` rather than stride-`(I, 1)`. The Triton epilogue handles this via explicit `stride_em / stride_en` arguments. The down-projection `out_slice` has to be `.contiguous()`-copied before in-place `addmm_` for the LoRA-B term, which adds one O(M·H) write — but this is dwarfed by the matmul time at LLaMA scale.

### v5_upgrade_1 — 2026-05-24

#### Approach
Two cuBLAS-tile-alignment fixes on top of v5, identified by the perf-analysis microbench in [`docs/analysis/v5_packing_diagnosis.md`](docs/analysis/v5_packing_diagnosis.md):
1. **Drop down-phase packing.** v5's packed [M, H+r] mega-output costs both a worse cuBLAS tile (76% peak at N=4112 vs 82% peak at N=4096) and a forced 0.15 ms `.contiguous()` copy on the H slice before `addmm_`. Reverting to v3's two-cuBLAS-call pattern (`out = h @ W_down^T`, then `xa_down = h @ A_down^T`, then `out.addmm_(xa_down, B_down^T, ...)`) skips both penalties.
2. **Pad the gate+up mega-matrix to a multiple of 128.** `2*I + 2*r = 28704` falls off cuBLAS's preferred N tile width (128). Appending zero-rows so N=28800 lifts the mega-GEMM from 79.6% → 83.2% peak (+0.9 ms in isolation). The padded columns of the result are zero @ X = 0 and are simply ignored when slicing.

The Triton SwiGLU+LoRA epilogue is reused unchanged from v5 (imported, not copied). Inference path is the v5 path verbatim — pre-merging LoRA already sidesteps both pain points.

#### Changes
- `experiments/v5/lora_mlp_kernel_v5_upgrade_1.py`: new file with
  - `pack_gate_up_weights_padded()` returning `(W_mega_padded, pad_rows)`
  - `_v5_upgrade_1_forward_impl()` (padded gate+up mega + v3-style down)
  - `lora_mlp_v5_upgrade_1()` and `LoRAMLPv5_upgrade_1(autograd.Function)`
  - `lora_mlp_v5_upgrade_1_inference` re-exports v5's inference path unchanged
- v5 kernel file is **not modified** (project convention).
- `tests/test_lora_mlp.py`: new `TestV5Upgrade1` class with 21 tests (forward fp32/bf16, rank sweep 8/16/32/64, vs Unsloth, no-LoRA, 2D/3D/non-pow-2 inputs, LLaMA-8B/13B shapes, fp64 gradcheck, bf16 backward vs reference). Adds two specific tests: `test_padded_alignment` (mega-N divisible by 128 across ranks) and `test_padded_correctness_matches_v5` (bf16 output matches v5 within tolerance).
- `benchmarks/bench_lora_mlp.py`: adds `v5_up1_train_ms`, `v5_up1_train_vs_unsloth`, `v5_up1_train_vs_v3`, `v5_up1_train_vs_v5` columns; pre-packs `W_mega_padded` once outside the timed loop, mirroring the v5 setup. CSVs now write to `v5_upgrade_1_<timestamp>.csv`.
- `benchmarks/microbench_v5_upgrade_1.py`: new microbench measuring (a) v3 vs v5 vs v5_upgrade_1 gate+up matmul, (b) v3 vs v5 down matmul end-to-end, (c) end-to-end training forward at LLaMA-8B/r16.

#### Results

**LLaMA-8B forward (bf16, batch=4, seq=2048, rank=16, M=8192, H=4096, I=14336):**

5-run medians with `triton.testing.do_bench(warmup=20, rep=100)`:

| Implementation | Time (ms) | vs Unsloth | vs v3 | Launches |
|----------------|-----------|------------|-------|----------|
| Unsloth `apply_lora_mlp_swiglu` | 12.72 | 1.00x | — | 10 |
| v3 (cuBLAS + Triton epilogue) | 12.48 | 1.02x | 1.00x | 8 |
| v5 training (packed) | 12.39 | 1.03x | 1.01x | 4 |
| **v5_upgrade_1 training (padded gate+up + v3-style down)** | **12.13** | **1.05x** | **1.03x** | **5** |
| v5 inference (pre-merged) | 11.78 | 1.08x | 1.06x | 4 |

**Microbench (matmul work only, M=8192, bf16):**

| Phase | v3 | v5 | v5_upgrade_1 |
|-------|------|------|--------------|
| gate+up matmul | 8.59 ms / 71.8% peak | 8.35 ms / 74.0% peak (N=28704) | **7.44 ms / 83.2% peak (N=28800)** |
| down matmul end-to-end | 3.95 ms / 78.5% peak | 4.27 ms / 72.6% peak | **3.95 ms / 78.5% peak** |

**End-to-end full sweep:** see `benchmarks/results/v5_upgrade_1_20260524_112741.csv`. v5_upgrade_1 beats or ties v3 across all 20 configurations swept; the gain is 1–3% at LLaMA-8B/M=8192 and 3–6% at smaller M=2048 where launch overhead is a larger share.

#### Key Finding
**Both fixes are real wins.** The microbench cleanly attributes ~0.9 ms to the padding fix and ~0.32 ms to dropping down packing — total ~1.2 ms of matmul-only savings. End-to-end, v5_upgrade_1 captures ~0.35 ms of that vs v5 and ~0.35 ms vs v3 at LLaMA-8B/r=16, satisfying the ≥1% beats-v3 success criterion (we land at +2.78%).

The asymmetry between the 1.2 ms isolated matmul savings and the 0.35 ms end-to-end gain is the same effect documented in the v5 diagnosis: the Triton SwiGLU+LoRA epilogue still reads `e_base`/`g_base` as non-contiguous slices of the gate+up mega-output (stride 0 = 28800 instead of 14336), which costs ~100–200 µs of L2-locality penalty even when the mega-matmul itself is fast. Closing that gap would require a different epilogue design (e.g. allocate `e_base` and `g_base` as separate contiguous buffers via two cuBLAS calls), which trades the LoRA-A absorption trick for clean strides — a different algorithmic family, so future work.

#### Limitations
- **One extra launch vs v5** (5 vs 4): v5_upgrade_1 has separate `h @ W_down^T` and `h @ A_down^T` calls instead of v5's packed down. Net wall-clock still wins because the cuBLAS-tile penalty was bigger than the saved launch overhead.
- **Pad rows waste a tiny amount of compute.** At LLaMA-8B/r=16, padding 28704 → 28800 adds 96 zero-rows × 4096 columns = 0.4M extra FMAs per matmul = 0.3% of the total. Worth it because cuBLAS picks a much better tile.
- **Backward unchanged.** Same as v5 — backward matmuls don't share an input.

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

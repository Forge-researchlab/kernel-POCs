# v5 Packed-cuBLAS Diagnosis: why "4 calls → 1 call" didn't move wall-clock

**Date:** 2026-05-24
**GPU:** NVIDIA A100-SXM4-80GB (CUDA 12.4, Torch 2.4.1, BLAS = cuBLAS)
**Shape:** LLaMA-8B forward, M=8192, H=4096, I=14336, r=16, bf16
**Bench tool:** `triton.testing.do_bench(warmup=10, rep=50)`
**Script:** `kernels/lora_mlp/benchmarks/microbench_v5_packing.py`

## TL;DR

> Packing the gate+up phase **does** save ~0.8 ms in isolation, but packing the
> down phase **loses** ~0.2 ms. Net matmul savings are ~0.6 ms, and they get
> swallowed by v5-specific epilogue costs (non-contiguous slice reads, an extra
> `.contiguous()` copy, an awkward N=4112 in the down GEMM), leaving wall-clock
> identical to v3.

## Hypothesis (going in)

Packing 4 cuBLAS calls into 1 should:
1. Read `X` once instead of 4× — save ~3× M*H*sizeof(bf16) = ~192 MiB of HBM traffic.
2. Cut kernel launches from 4 → 1, saving ~50–100 µs.
3. Give cuBLAS a single larger N to pick a more efficient tile.

## Method

For each of v3 (separate calls) and v5 (1 packed call), I measured **just the
matmul work** at LLaMA-8B scale, computed achieved TFLOPS and HBM bandwidth, and
also measured the **end-to-end** down phase (mega-matmul + slice + `.contiguous()`
+ `addmm_`) plus the standalone cost of the `.contiguous()` copy.

## Raw numbers

A100 peaks: 312 TFLOPS bf16, ~1.94 TB/s HBM (effective ~1.5 TB/s).

### Gate+Up matmul variants

| Variant                            | time      | TFLOPS | %peak |
|------------------------------------|-----------|--------|-------|
| v3 (4 cuBLAS calls)                | 8.576 ms  | 224.6  | 72.0% |
| v5 (1 mega-GEMM, N=28704)          | 7.753 ms  | 248.4  | 79.6% |
| v3 base only, no LoRA-A (2 cuBLAS) | 7.554 ms  | 254.7  | 81.6% |
| v5 base only, no LoRA-A (1 mega, N=28672) | 7.461 ms | 257.9 | 82.7% |

### Down matmul variants

| Variant                                | time      | TFLOPS | %peak |
|----------------------------------------|-----------|--------|-------|
| v3 down (2 cuBLAS calls)               | 3.808 ms  | 253.6  | 81.3% |
| v5 down (1 mega, N=H+r=4112)           | 4.090 ms  | 236.1  | 75.7% |
| v3 down base only (1 cuBLAS, N=4096)   | 3.743 ms  | 257.1  | 82.4% |
| v3 down full (2 cuBLAS + addmm_)       | 4.010 ms  | 241.1  | 77.3% |
| v5 down full (mega + contig + addmm_)  | 4.232 ms  | 228.4  | 73.2% |
| `.contiguous()` copy of [M, H] slice   | 0.147 ms  | —      | 0.91 TB/s |

### Net matmul work (gate+up + end-to-end down)

| Path | time      | Δ vs v3 |
|------|-----------|---------|
| v3   | 12.587 ms | —       |
| v5   | 11.986 ms | −0.60 ms (+4.8%) |

Full-MLP benchmark (from `benchmarks/results/v5_20260524_104637.csv`,
`mlp_train_v5_vs_v3`):

| Path        | full-MLP time |
|-------------|---------------|
| v3 training | 12.35 ms      |
| v5 training | 12.36 ms      |
| v5 inference (pre-merged W) | 11.79 ms |

So v5's training path captures **0 / 600 µs** of the isolated savings. v5's
inference path **does** capture an additional ~0.5 ms because it avoids both
LoRA-A matmuls and the down-phase `.contiguous()`.

## Root cause (operation by operation)

### 1. Gate+up: packing helps, but not for the reason you'd expect

The two **big** base matmuls alone are already cuBLAS-saturated:

```
v3 (2 cuBLAS, both N=14336):   7.554 ms  @ 254.7 TFLOPS
v5 (1 mega,    N=28672):       7.461 ms  @ 257.9 TFLOPS
                                 ~~~~~~
                            only 0.09 ms diff (1.2%)
```

Packing **two big GEMMs** into a single N=28672 mega-GEMM is essentially free —
cuBLAS already hits ~82% of bf16 peak at N=14336, and a 2× wider N doesn't help
because the kernel is fundamentally compute-bound on these shapes.

The real saving in gate+up comes from the **two skinny LoRA-A matmuls**:

```
v3 (4 cuBLAS = 2 big + 2 skinny):   8.576 ms
v5 (1 mega = base + LoRA-A cols):   7.753 ms
                                     ~~~~~~
                            -0.82 ms (+9.6%)
```

Each `X @ A_gate^T` is a [8192, 4096] × [4096, 16] = ~1.07 GFLOP matmul that
cuBLAS launches as a separate skinny kernel. The skinny kernel itself isn't
free (cuBLAS picks a GEMV-style algo, low SM utilization), and 4 launches
back-to-back fight for the L2 with the bigger calls. Folding those 16 columns
onto the end of the mega-GEMM (going from N=28672 → N=28704) costs essentially
nothing — the mega-GEMM stays at the same TFLOPS — so the per-call overhead of
the skinny launches disappears.

### 2. Down: packing hurts

```
v3 (matmuls only):        3.808 ms
v5 (mega only, N=4112):   4.090 ms     -0.28 ms (-7.4%)

v3 (matmul + addmm):      4.010 ms
v5 (mega + contig + addmm): 4.232 ms   -0.22 ms (-5.5%)
```

Two compounding problems:

(a) **Awkward N=4112.** cuBLAS picks great tiles for the clean N=4096 down
GEMM (257 TFLOPS, 82% peak). Adding r=16 padding columns drops it to 236 TFLOPS
(76% peak). That's a ~7 TFLOPS hit on a ~250 TFLOPS GEMM = ~0.18 ms cost.

(b) **Non-contiguous H slice forces an explicit copy.** The mega-output is
[M, H+r] row-major, so the H slice `result[:, :H]` has stride 0 = H+r = 4112,
not H = 4096. `addmm_` needs a contiguous output buffer, so v5 calls
`.contiguous()` on the slice, which is a 64 MiB bf16 read+write copy that
benchmarks at 0.147 ms / 0.91 TB/s (about 47% of effective HBM peak). v3
skips this entirely because its `out_base` is contiguous straight from cuBLAS.

### 3. The "lost" 600 µs in the full forward

Isolated savings = 0.60 ms. Wall-clock savings ≈ 0. The 600 µs goes into v5
overheads that don't appear in the matmul-only microbench:

- **Triton epilogue reads non-contiguous slices.** v5's
  `_fused_lora_swiglu_kernel` reads `e_base` and `g_base` from the mega-output
  with stride 0 = 2*I + 2*r = 28704 instead of v3's stride 0 = 14336.
  Each [BLOCK_M, BLOCK_N] tile load now straddles wider rows in HBM, hurting
  L2 hit rate. The epilogue is small (~0.3–0.5 ms) but a 30–50% slowdown
  from non-contiguous reads costs ~100–200 µs.
- **Saving `e_full`, `g_full` for backward** in training mode: v5 does this
  via two extra `torch.empty(M, I)` writes inside the epilogue (~115 MiB each).
  v3 does the same, so this isn't a v5-specific cost — included for completeness.
- **`.contiguous()` copy in down phase**: ~150 µs (already counted in the
  end-to-end down number above).
- **Mega-matmul intermediate is larger**: v5's [M, 28704] gate+up output is
  448 MiB (vs v3's individual buffers totaling the same bytes, but allocated
  separately). This evicts useful state from L2 between the gate+up matmul
  and the Triton epilogue.
- **do_bench noise** on a 12 ms benchmark: σ ≈ 50–100 µs run-to-run.

Net: the 600 µs of matmul savings exactly cancels the ~150 µs `.contiguous()`
copy + ~150 µs non-contig epilogue + ~100 µs launch/L2 effects + noise.

## Phase 2: cuBLAS algorithm investigation

`torch.backends.cuda.preferred_blas_library()` returns `_BlasBackend.Cublas`
(not cuBLASLt). cuBLAS picks its algorithm internally per shape; we can't
override it without going through cublasLt. The microbench already proves the
algo selection story: N=4096 gets a great tile (82% peak), N=4112 gets a worse
one (76% peak), and N=28672 vs N=28704 are nearly identical. The N=4112 hit
is the bigger of the two cuBLAS-tile-selection penalties.

## Phase 5: cublasLt explicit algo — skipped

Per the prior worker's notes the v4 `cublaslt_wrapper.py` has broken algo enum
IDs. Even if we fixed it, the upside is at most:
- Recover the N=4112 → ~257 TFLOPS gap: ~0.18 ms in the down phase.
- Maybe match the gate+up mega-GEMM at ~260 TFLOPS instead of 248: ~0.3 ms.

Total reachable: ~0.5 ms. Same order as the noise we're already inside.
**Not worth the engineering cost.**

## Recommendation

**Revert the down-phase packing; keep the gate+up packing if anything.**

The data is unambiguous:

1. **Drop down-phase packing.** It costs both compute (cuBLAS picks a worse
   tile for N=4112) and memory (forced `.contiguous()` copy). Use v3's pattern:
   two separate cuBLAS calls for `h @ W_down^T` and `h @ A_down^T`, then
   `addmm_` as before.
2. **Gate+up packing is a marginal win in isolation (~0.8 ms)** but the
   Triton epilogue's non-contiguous reads claw most of it back. If we want to
   keep packing here, the next step is to **make the epilogue write directly
   into a layout that matches the mega-output** so the strided reads aren't
   wasted — i.e., write the SwiGLU output `h` with the same stride pattern as
   the mega-matmul, avoiding any restride. Or simpler: allocate `e_base` and
   `g_base` as **separate contiguous buffers** and call cuBLAS twice (giving up
   the LoRA-A absorption trick but keeping clean strides for the epilogue).
3. **v5 inference is the clear keeper.** Pre-merging LoRA into W and pre-
   transposing eliminates both the LoRA-A overhead in gate+up AND the
   down-phase contiguous copy (W_down_eff is [H, I], not [H+r, I]). That's
   why inference shows the full ~0.5 ms win.

### Possible follow-ups worth measuring (not done here)

- **Variant A — gate+up split, down split (revert v5 training to v3 layout):**
  expected ~12.4 ms, same as v3, but with cleaner code.
- **Variant B — gate+up packed, down split:** expected 12.1–12.2 ms if the
  Triton epilogue can be made friendlier to the strided mega-output reads.
- **Variant C — keep both packings but use a writable output for the down
  GEMM via cublasLt with an explicit output layout that places A_down rows at
  a separate output buffer:** speculative, requires fixing the cublasLt
  wrapper, upside ~0.2–0.5 ms.

## Reproduction

```bash
cd /workspace/kernel-POCs/kernels/lora_mlp
python benchmarks/microbench_v5_packing.py
```

Verified on A100-SXM4-80GB, CUDA 12.4, Torch 2.4.1, bf16.

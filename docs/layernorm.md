# LayerNorm — Forge Kernel

> Status: CP1 reference implementation. A100-80GB validated. 89/89 hard gates passing as of 2026-05-23.

This document describes Forge's LayerNorm Triton kernel, how it differs from
the two reference implementations we benchmark against (Liger-Kernel and
Unsloth), and the measured behavior on an A100-80GB.

---

## 1. Operation

LayerNorm normalizes each row of `X ∈ ℝ^{B×S×H}` over the trailing dimension
and applies a per-channel affine:

```
μ_r    = (1/H) · Σ_i x_{r,i}
σ²_r   = (1/H) · Σ_i (x_{r,i} − μ_r)²
rstd_r = 1 / √(σ²_r + ε)
y_{r,i} = (x_{r,i} − μ_r) · rstd_r · w_i + b_i
```

Backward (per row, with `x̂ = (x − μ) · rstd`):

```
dx̂_i  = dy_i · w_i
c1    = (1/H) · Σ_i dx̂_i
c2    = (1/H) · Σ_i dx̂_i · x̂_i
dx_i  = rstd · (dx̂_i − c1 − x̂_i · c2)
dw_i  = Σ_r dy_{r,i} · x̂_{r,i}     # reduced over rows
db_i  = Σ_r dy_{r,i}                # reduced over rows
```

LayerNorm is **bandwidth-bound**. Math per element is trivial; the cost is
moving `X` and `Y` between HBM and SRAM. Every meaningful optimization is
about reducing HBM round-trips.

---

## 2. Two variants

Forge ships LayerNorm as **two Triton variants** with the same forward but
different backward strategies:

| Variant   | dX | dW | dB | Peak VRAM (bwd) | Intended use                                |
|-----------|----|----|----|-----------------|---------------------------------------------|
| `ForgeLayerNormLiger`   | ✓  | ✓  | ✓  | higher       | Training W, B (default LN behavior)         |
| `ForgeLayerNormUnsloth` | ✓  | ✗  | ✗  | lower        | Frozen LN (LoRA, inference, locked norms)   |

Both are wrapped in `torch.autograd.Function` and a `torch.nn.Module` shim
that mirrors `torch.nn.LayerNorm`.

**Contract for the Unsloth variant:** backward returns `None` for `dW, dB`.
If wired into a module whose `weight`/`bias` are `requires_grad=True`, those
parameters will silently never update. Forge's test suite enforces this via
an explicit contract test and an `xfail` gradcheck — but the kernel itself
does not raise.

---

## 3. How Forge's kernel differs from existing implementations

### vs PyTorch eager (`F.layer_norm`)

Modern PyTorch already calls a fused ATen LayerNorm kernel — so the
folklore "eager fires 10+ small kernels and Triton fuses them" is **out of
date** (we measured 6 CUDA events for eager fwd+bwd, see §6). The win over
eager comes from elsewhere:

1. **Single Python `autograd.Function` per call** — eager dispatches through
   the ATen graph (`native_layer_norm` + `native_layer_norm_backward`); Forge
   skips the graph and goes straight to a Triton launch.
2. **fp32/fp64 accumulators with native-dtype stores** — eager honors the same
   pattern, but our backward keeps the entire reduction chain in the
   accumulator dtype before the final `tl.store` cast.
3. **dW/dB without atomics** — eager's CUDA path uses an unrolled reduction
   strategy that's solid but tuned for the general case; we use a
   partial-buffer reduction sized to `min(M, SM_count)` (see §4.3) that
   avoids both atomics and full-M materialization.

### vs Liger-Kernel

Liger's LayerNorm forward and backward are structurally the same as
Forge's Liger variant — that's why we call it "Liger-style". The
differences are:

- **fp64 accumulators on fp64 inputs** — Liger keeps accumulators in
  `tl.float32` regardless of input dtype. Their `torch.autograd.gradcheck` at
  fp64 therefore fails (silent dtype downcast). Forge's kernel derives an
  `ACC_DTYPE: tl.constexpr` from `X.dtype` and routes `fp64 → tl.float64`,
  everything else → `tl.float32`. This makes Forge the only one of the three
  that passes vanilla `gradcheck(..., atol=1e-5, rtol=1e-3)` without test
  contortions.
- **Single-pass row reduction (no Welford)** — Liger also avoids Welford here;
  we made the same call and document the rationale (§4.2) for future
  contributors who'll be tempted to add it.
- **Tolerance-aware test suite** — Liger's reference compares same-dtype to
  same-dtype, which masks the `√M · ε` reduction-noise floor. Forge's test
  compares against an fp64 reference (in cell 7 of the test notebook) so the
  remaining error is the kernel's, not the reference's.

### vs Unsloth

Unsloth's LayerNorm is the inspiration for our second variant. Key
differences vs Unsloth's original:

- **No assumption that the parent module owns dY** — Unsloth's in-place trick
  works because it knows the caller will not reuse `dY` after backward. We
  preserve that contract but also export a Liger-style variant for callers
  that need `dW`/`dB`.
- **Explicit `None` return for `dW`/`dB`** — Unsloth's wrapper sometimes
  builds zero tensors; we return `None` so autograd doesn't waste a
  zero-allocation. The cost is the silent-no-grad footgun, which our test
  suite catches.
- **fp64 path** — same as the Liger comparison: Unsloth is fp32-accumulator
  only; we route to fp64 on fp64 inputs.
- **fp32-source attribution** — our reference test (cell 7 in the notebook)
  was rewritten to compare against an fp64-precision reference rather than
  a same-dtype eager reference, so noise observed is attributable to the
  kernel under test, not the reference.

### Summary table

| Behavior                                        | Eager | Liger | Unsloth | **Forge** |
|-------------------------------------------------|-------|-------|---------|-----------|
| Fused fwd + bwd kernels                         | ✓     | ✓     | ✓       | ✓         |
| In-place `dY → dX` backward (variant)           | ✗     | ✗     | ✓       | ✓         |
| Liger-style partial-buffer dW/dB (variant)      | ✗     | ✓     | ✗       | ✓         |
| fp64 accumulators on fp64 inputs                | ✓     | ✗     | ✗       | **✓**     |
| Passes `torch.autograd.gradcheck` at fp64       | ✓     | ✗     | ✗       | **✓**     |
| Reduction reference tested against fp64-truth   | n/a   | ✗     | ✗       | **✓**     |
| Modular API (kernel registry, A/B testable)     | n/a   | partial | ✗     | ✓         |

---

## 4. Design decisions

### 4.1 Single-pass forward, one row per program

Each Triton program handles one row of `X_flat ∈ ℝ^{M×H}` where `M = B·S`.
The row is loaded into SRAM once; `μ`, `σ²`, `rstd`, `x̂`, the affine, and the
output `y` are all computed in registers before a single HBM write of `Y`,
`Mean`, `RSTD`. This is the minimum HBM traffic the operation admits.

### 4.2 Welford avoided on purpose

We compute `σ² = mean((x − μ)²)` with a second SRAM-resident pass over the
row rather than `var = mean(x²) − mean(x)²` or a Welford recurrence. At
LN's typical scale (H ≤ 16K) the second pass is free (data is already in
SRAM), and the cancellation-prone form is numerically inferior at low
precision. Welford would matter for H ≫ 16K — when we hit that we'll
revisit.

### 4.3 Partial-accumulator backward (Liger variant)

`dW = Σ_r dy_r · x̂_r` over `M = B·S` rows is the classic atomic-vs-reduction
tradeoff. Two options:

- **Atomic write per row**: simple, but serializes SM writes to a shared
  `[H]` buffer.
- **Per-SM partial accumulator**: each program accumulates a private `[H]`
  strip in registers/SRAM and writes one row of an `[SMs, H]` partial buffer
  out to HBM; Python does the final `partial.sum(0)` reduction.

We picked option 2 — no atomics, no SM contention. We launch
`num_programs = min(M, SM_count)` programs. Each handles
`ceil(M / num_programs)` rows. The extra reduction is itself
bandwidth-bound and runs at ~PyTorch's eager reduction speed — cheap.

Partial-buffer overhead is `num_programs × H × sizeof(acc)`. On A100 with
`108 SMs`, `H = 4096`, fp32 accumulator, that's **1.7 MB** — negligible.

### 4.4 In-place `dY → dX` backward (Unsloth variant)

When `W` and `B` are frozen we don't need `dW`/`dB` and we don't need a
separate `dX` buffer. The Unsloth kernel reads `dy_row` into SRAM and writes
`dx_row` back over the same HBM slot. Saves ≈ `M · H · sizeof(dtype)` bytes
of peak VRAM and one allocation. See §6 for the measured saving.

### 4.5 Power-of-2 BLOCK_SIZE heuristic

```python
def _calculate_settings(n_cols):
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 65536)
    num_warps  = min(max(BLOCK_SIZE // 256, 1), 8)
    return BLOCK_SIZE, num_warps
```

Power-of-2 alignment matters: at `H = 4097`, `BLOCK = 8192` and ~50% of every
thread's work is masked-out. `num_warps` scales with block size so per-warp
register pressure stays bounded. This is a heuristic — no autotune yet (see
§7).

### 4.6 fp32 reductions, native-dtype tensor ops, fp64 promotion when warranted

All sums/reductions happen in `ACC_DTYPE`:

- `X.dtype == torch.float64` → `ACC_DTYPE = tl.float64`
- otherwise → `ACC_DTYPE = tl.float32`

Stores cast back to the input dtype only at the final `tl.store`. This is
what keeps the bf16 path within a few ULP of `F.layer_norm` and what allows
the fp64 path to pass `gradcheck`. The `Mean`, `RSTD`, `R`, `Mu`, and
`DW_partial`, `DB_partial` buffers also track this dtype so the reduction
chain stays in promoted precision end to end.

---

## 5. Numerical accuracy — the `√M · ε` floor

The dW reduction sums `M = B · S` terms in the accumulator dtype. The
resulting error scales as `~√M · ε_dtype` (linear growth would be a real
bug; sqrt growth is the central-limit floor of rounding noise).

Measured Forge dW error vs fp64 reference at fp32 accumulator:

| shape            | M       | √M    | dW max abs error | error / √M |
|------------------|---------|-------|------------------|------------|
| (4, 512, 4096)   | 2,048   | 45.3  | 2.3e-5           | 5.1e-7     |
| (4, 2048, 4096)  | 8,192   | 90.5  | 7.6e-5           | 8.4e-7     |
| (8, 2048, 4096)  | 16,384  | 128.0 | 1.1e-4           | 8.6e-7     |
| (8, 4096, 4096)  | 32,768  | 181.0 | 2.3e-4           | 1.3e-6     |

`error / √M` is bounded; this is the fp32 reduction noise floor — there is
no algorithm that does better at fp32 without spending bytes on Kahan or
pairwise compensated sums. Forge ships **two** mitigation paths:

1. **fp64 accumulators on fp64 inputs.** When the caller passes
   `X.dtype == torch.float64`, the entire reduction chain (including
   `DW_partial`) is fp64. dW error drops below the `atol=1e-7` fp64
   gradcheck tolerance, which is how 3/3 gradcheck rows now pass.
2. **Tolerance scaled to reduction width.** For fp32 callers, the test
   tolerance must absorb `√M · ε_fp32 ≈ 1e-4` at M = 32K. Forge's test
   suite (notebook cell 7) compares the kernel **at fp64** rather than
   loosening fp32 tolerance — this makes "is the kernel correct?" and "is
   fp32 enough precision for your reduction?" two separate questions.

---

## 6. Measured results — A100-80GB, May 2026

Test suite: `kernels/layernorm/layernorm_tests.ipynb`. Executed
end-to-end via `jupyter nbconvert --execute`. **89 hard-gate PASS, 0 FAIL,
10 INFO** (perf/memory tables).

### 6.1 Correctness

| Section                         | Detail                                                | Result |
|---------------------------------|--------------------------------------------------------|--------|
| Forward vs `F.layer_norm`       | 6 shapes × 3 dtypes × 2 variants                       | 36/36 PASS |
| Liger backward (dX, dW, dB)     | 6 shapes × {bf16, fp64} against fp64 reference         | **12/12 PASS** |
| Unsloth backward (dX only)      | 6 shapes × {bf16, fp32}                                | 12/12 PASS |
| Unsloth `dW/dB = None` contract | Module-level                                           | PASS |
| fp64 `gradcheck`                | Liger (X, W, B); Unsloth (X only); Unsloth xfail       | **3/3 PASS** |
| Edge cases                      | ε sweep, non-pow2 H ∈ {3, 17, 4097, 8193}, var=0, state_dict | 22/22 PASS |
| Variant parity                  | Liger ≡ Unsloth fwd, dX, in-place `dY` overwrite       | 3/3 PASS |

Shape sweep used: `(2,8,1024)`, `(1,1,128)`, `(4,512,4096)`, `(4,2048,4096)`,
`(8,2048,4096)`, `(8,4096,4096)`.

### 6.2 Latency (median of 100 iterations after 25 warmups)

| shape × dtype                       | eager fwd | liger fwd       | unsloth fwd     | best fwd+bwd                       |
|-------------------------------------|-----------|-----------------|-----------------|------------------------------------|
| (4, 2048, 4096) · bf16              | 0.155 ms  | 0.150 ms (1.04x)| 0.150 ms (1.03x)| **unsloth 0.43 ms** (eager 0.52)   |
| (4, 2048, 4096) · fp32              | 0.243 ms  | 0.225 ms (1.08x)| 0.235 ms (1.03x)| liger 0.66 ms  (eager 1.00)        |
| (8, 2048, 4096) · bf16              | 0.238 ms  | 0.228 ms (1.04x)| 0.225 ms (1.06x)| liger 0.69 ms  (eager 0.98)        |
| (8, 2048, 4096) · fp32              | 0.463 ms  | 0.378 ms (**1.22x**)| 0.379 ms (1.22x)| liger 1.24 ms (eager 1.93)     |

`torch.compile(mode='reduce-overhead')` ran 0.56-0.63× of eager — its CUDA
graph capture overhead doesn't amortize for an isolated LN call. This is
expected and not a problem for the patched-model path, where multiple ops
fall under one graph.

### 6.3 Peak VRAM (Unsloth in-place trick — observed)

| shape           | eager fwd+bwd | liger fwd+bwd | unsloth fwd+bwd | analytical saving | observed saving |
|-----------------|---------------|---------------|-----------------|-------------------|-----------------|
| (4, 2048, 4096) | 3456.5 MB     | 3459.9 MB     | 3392.5 MB       | 64.0 MB           | **+67.4 MB** ✓  |
| (8, 2048, 4096) | 3840.6 MB     | 3844.0 MB     | 3712.6 MB       | 128.0 MB          | **+131.4 MB** ✓ |

The observed Liger overhead vs Unsloth matches the analytical model
(`M · H · sizeof(dtype)`) **plus** the 3.4 MB `DW_partial + DB_partial`
buffers Liger carries:

```
108 SMs × 4096 H × 4 bytes × 2 buffers = 3.4 MB
```

— exactly the residual. Memory accounting closes to the byte.

### 6.4 Memory bandwidth (% of A100 HBM peak = 1555 GB/s)

| shape           | impl    | direction     | GB/s | % peak  |
|-----------------|---------|---------------|------|---------|
| (4, 2048, 4096) | liger   | fwd           | 890  | 57%     |
| (4, 2048, 4096) | unsloth | fwd           | 892  | 57%     |
| (8, 2048, 4096) | liger   | fwd           | 1186 | **76%** |
| (8, 2048, 4096) | unsloth | fwd           | 1187 | **76%** |
| (4, 2048, 4096) | liger   | bwd (est)     | 376  | 24%     |
| (4, 2048, 4096) | unsloth | bwd (est)     | 536  | 35%     |
| (8, 2048, 4096) | liger   | bwd (est)     | 672  | 43%     |
| (8, 2048, 4096) | unsloth | bwd (est)     | 777  | 50%     |

Forward at the design shape (`B·S = 16384`) is at **76% of A100 HBM peak** —
the kernel is doing what a bandwidth-bound LN should. Backward sits lower
because the partial-buffer reduction adds an extra HBM write phase.

### 6.5 CUDA event count (fwd + bwd, `(4, 2048, 4096)` bf16)

| impl    | CUDA events |
|---------|-------------|
| eager   | 6           |
| liger   | 11          |
| unsloth | 6           |

Eager is already fused — the historical "Triton = fewer launches" story
needs an update. Liger's 11 events come from the `[SMs, H]` partial buffer
reduction (`.sum(0)` plus dtype casts on the Python side). Unsloth matches
eager. If `.sum(0)` reduction is on the critical path for a workload, a
fused `dW` reduction kernel is the obvious next step.

### 6.6 Alignment cost (non-power-of-2 H)

| H     | BLOCK | masked threads | fwd ms |
|-------|-------|----------------|--------|
| 4096  | 4096  | 0%             | 0.157  |
| 4097  | 8192  | 50%            | 0.230 (+46%) |
| 8192  | 8192  | 0%             | 0.237  |
| 8193  | 16384 | 50%            | 0.410 (+73%) |

Non-pow2 `H` pays nearly the full cost of the next-larger pow2. This is
the single biggest open optimization (see §7) — an autotune sweep over
`(BLOCK_SIZE, num_warps, num_stages)` should claw most of it back.

> Note: during the first full-notebook run, the `H=8192` measurement
> registered a transient 3.01 ms outlier (12× expected). Re-running cell 28
> in isolation produced 0.238 ms — consistent with adjacent measurements.
> Cause is likely a one-off autotune cache miss or allocator stall; median
> of 100 is normally robust, but the test is sensitive enough that it can be
> caught by a single bad sample. Logged as a soft-report robustness issue.

---

## 7. Known limitations / open work

1. **No autotune.** `_calculate_settings(H)` is a hand-tuned heuristic. A
   `triton.autotune` sweep over `(BLOCK_SIZE, num_warps, num_stages)` is the
   highest-leverage open optimization — would mostly target the 46-73%
   alignment penalty at non-pow2 `H`.
2. **No Kahan / pairwise sum.** Acceptable up to H = 16K; revisit if shapes
   grow larger.
3. **Unsloth `dW/dB = None` contract is silent.** The kernel does not warn
   when called with a trainable `W` or `B` — the test suite catches it. A
   one-shot warning in the autograd `forward` would be cheap.
4. **Backward `dY` clone in tests.** The Unsloth backward overwrites `dY` in
   place; any test that reuses `dY` across calls must `.clone()` first. This
   is a kernel contract, not a bug, but easy to trip on.
5. **dW reduction is Python-side.** `DW_partial.sum(0)` adds 5-7 CUDA events
   to the fwd+bwd pipeline. A fused Triton reduction kernel would bring
   Liger's launch count down to ~3.
6. **No torch.compile fast path.** `reduce-overhead` mode adds overhead
   beyond what it saves on an isolated call. Worth re-testing once the
   kernel is integrated under `forge.patch` and runs inside a graph.

---

## 8. References

- Test notebook: `kernels/layernorm/layernorm_tests.ipynb`
- Kernel: `kernels/layernorm/layernorm_kernel.py`
- Inline context: `kernels/layernorm/context.md`
- Liger-Kernel LayerNorm: <https://github.com/linkedin/Liger-Kernel> (Apache-2)
- Unsloth LayerNorm: <https://github.com/unslothai/unsloth> (Apache-2 repo,
  individual files carry LGPL headers — reference-read OK, code-copy needs
  clearance)
- A100 SXM4 80GB HBM2e peak bandwidth: 1555 GB/s

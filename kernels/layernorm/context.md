# LayerNorm — Context

## What this kernel does

LayerNorm normalizes each row of an `(B, S, H)` activation tensor across the
last (hidden) dimension and then applies a per-channel affine transform:

```
mean(x)  = (1/H) * sum_i x_i
var(x)   = (1/H) * sum_i (x_i - mean)^2
rstd     = 1 / sqrt(var + eps)
y_i      = (x_i - mean) * rstd * w_i + b_i
```

There are two Triton variants in `layernorm_kernel.py`:

- `ForgeLayerNormLiger` — full backward, computes `dX`, `dW`, `dB`
- `ForgeLayerNormUnsloth` — `dX`-only backward, written **in-place** into the
  upstream `dY` buffer; returns `None` for `dW, dB`

Both are wrapped in a `torch.autograd.Function` and a `nn.Module` shim and
match `F.layer_norm` for the forward pass.

## What's improved vs PyTorch eager

PyTorch's eager `F.layer_norm` decomposes into ~10+ small CUDA kernels: one
per reduction (mean, var), an `affine` multiply, an `add` for bias, and so on.
Each kernel reads `X` from HBM and writes a temporary back. LayerNorm is
**bandwidth-bound** — its math is trivial, the cost is moving `X` and `Y` in
and out of HBM. The Triton kernels here fuse the work and stop paying those
round trips.

Concretely:

1. **Single-pass fwd kernel.** One row per program. We load `x_row` into SRAM
   once, compute `mean`, `var`, `rstd`, the normalized value, and the affine
   in the same kernel. `Y`, `Mean`, `RSTD` are the only HBM writes.

2. **Welford-free reduction.** We deliberately don't run a two-pass / Welford
   reduction. `var = mean(x^2) - mean(x)^2` is numerically dicey at low
   precision, so we do `(x - mean)^2` after the mean reduction — still one
   pass over SRAM-resident data — and keep the partial sums in fp32 even when
   `X` is bf16/fp16. This is the cheap, correct path at LN's typical scale.

3. **Partial-accumulator backward (Liger).** Computing `dW = sum_rows dy * x_hat`
   and `dB = sum_rows dy` over `M = B*S` rows is the classic atomic vs.
   reduction tradeoff. Atomics would serialize SM writes to a single `[H]`
   buffer. Instead we launch `min(M, SM_count)` programs; each accumulates a
   private `[H]` strip in registers/SRAM and writes it out as a row of an
   `[SMs, H]` partial buffer. Python then does the `partial.sum(0)` reduction.
   No atomics, no contention; the extra reduction is itself bandwidth-bound and
   cheap.

4. **In-place dY → dX (Unsloth).** When `W, B` are frozen (or grads computed
   elsewhere), we don't need `dW, dB` and we don't need to allocate a separate
   `dX` buffer. The Unsloth kernel reads `dy_row` into SRAM and writes `dx_row`
   back over the same `dY` HBM slot. Saves ≈ `M * H * elem_size` bytes of peak
   VRAM and one HBM allocation.

5. **Power-of-2 BLOCK_SIZE tuning.** `_calculate_settings(H)` rounds the per-row
   block up to the next power of 2 (capped at 65536) and picks `num_warps` so
   the warp count scales with block size (`min(max(BLOCK // 256, 1), 8)`).
   Power-of-2 alignment matters: at `H=4097`, BLOCK becomes 8192 and ~50% of
   every thread's work is masked-out — `test_alignment_impact.py` exposes
   this diagnostically.

6. **fp32 reductions, native-dtype tensor ops.** All sums/reductions happen in
   fp32 even when `X` is bf16. We only cast back at the final `tl.store` for
   `Y`, `dX`. This is what keeps the bf16 path within a few ULP of eager
   `F.layer_norm`.

## Two variants — when to use which

| Variant   | dX | dW | dB | Peak VRAM (bwd) | Use case |
|-----------|----|----|----|-----------------|----------|
| Liger     | ✓  | ✓  | ✓  | higher          | Training W, B (default LN behavior) |
| Unsloth   | ✓  | ✗  | ✗  | lower (in-place)| Frozen LN (e.g. LoRA / inference / norms locked) |

**Important contract — Unsloth returns `None` for `dW, dB`.** If you wire
`ForgeLayerNormUnsloth` into a module whose `weight` and `bias` are
`requires_grad=True`, those parameters will silently never update. The test
suite enforces this contract explicitly via
`test_unsloth_dw_db_are_none_by_contract` and an `xfail` gradcheck.

## How we want to test it

Goal: every claim the kernel makes — correctness, gradient correctness, perf,
memory — has a test that surfaces a regression. Tests are split into hard
gates (default `pytest`) and soft perf reports (`pytest -m bench`).

### Test matrix

| Concern        | File                          | What it checks                                          | Pass/fail |
|----------------|-------------------------------|---------------------------------------------------------|-----------|
| Forward        | `test_correctness.py`         | Both variants vs `F.layer_norm` across bf16/fp16/fp32 and a shape sweep (1×1×128 → 8×4096×4096) | hard |
| Backward       | `test_correctness.py`         | Liger: dX, dW, dB vs autograd. Unsloth: dX vs autograd (W, B frozen); dW/dB are `None`. | hard |
| Gradcheck      | `test_gradcheck.py`           | fp64 `torch.autograd.gradcheck`. Liger over (X,W,B); Unsloth over (X,) only; xfail documents the full-input failure for Unsloth. | hard |
| Edge cases     | `test_edge_cases.py`          | eps sweep, minimal shape, non-power-of-2 H (3, 17, 4097, 8193), near-block-cap H, constant X (var=0), state_dict round-trip. | hard |
| Variant parity | `test_variant_comparison.py`  | Liger==Unsloth forward at fp32; dX matches when W, B frozen; saved-tensor shapes; in-place `dY` overwrite verified. | hard |
| Latency        | `test_perf_time.py`           | fwd / fwd+bwd ms for eager / `torch.compile` / Liger / Unsloth on design shapes. | soft (report) |
| VRAM           | `test_perf_memory.py`         | Peak MB for fwd, fwd+bwd. Confirms Unsloth's in-place savings vs Liger ≈ `M·H·elem_size`. | soft |
| Bandwidth      | `test_bandwidth.py`           | Achieved GB/s and % of A100 40GB peak (1555 GB/s) for fwd and bwd. | soft |
| Fusion         | `test_launch_count.py`        | CUDA-event count via `torch.profiler` — eager ~10+, Triton ~2. | soft |
| Alignment      | `test_alignment_impact.py`    | Diagnostic: fwd time for H ∈ {4096, 4097, 8192, 8193}; shows masked-element waste. | soft |

### Tolerances

`_helpers.py` centralizes:

```python
TOL_BF16 = dict(rtol=1e-2, atol=1e-2)
TOL_FP16 = dict(rtol=1e-3, atol=1e-3)
TOL_FP32 = dict(rtol=1e-5, atol=1e-5)
```

bf16 LayerNorm reductions are noisier than other ops — `rtol=1e-5` (the value
suggested by other kernel docstrings) is too tight at `(8, 2048, 4096)`. `1e-2`
is the empirically validated floor against `F.layer_norm`. Gradcheck stays
strict at fp64.

### Why soft thresholds for perf

The hackathon goal is "show fusion wins, surface regressions." A hard
"speedup ≥ X" check is hostile to dtype changes, GPU variance, and `torch.compile`
warm-up. Instead, perf tests print tables; we eyeball them. Hard correctness is
non-negotiable; perf numbers are diagnostic.

### How to run

On an A100:

```bash
# Default — correctness, gradcheck, edge cases, variant comparison
pytest tests/test_kernels/layernorm/ -v

# Perf, memory, bandwidth, launch count, alignment tables
pytest tests/test_kernels/layernorm/ -m bench -s

# Standalone harness summary
python -m benchmarks.bench_layernorm

# Global summary including all kernels
python -m benchmarks.bench_all
```

On any other GPU the suite auto-skips. Override with `FORGE_ALLOW_NON_A100=1`
if you want to run elsewhere (numbers will be device-dependent).

## Known limitations / open questions

- **bf16 reductions.** Sum across H=4096+ at fp32 accumulator works in practice
  but isn't a Kahan or pairwise sum. Future: revisit if shapes grow past 16K.
- **No autotune.** `_calculate_settings(H)` is a hardcoded heuristic. A
  `triton.autotune` sweep over `BLOCK_SIZE`, `num_warps`, `num_stages` would
  squeeze more out, especially for non-power-of-2 H.
- **Unsloth dW/dB contract is silent.** Returning `None` for `dW, dB` when the
  user passes `requires_grad=True` doesn't raise — the test contract catches
  this, but the kernel itself doesn't. Future: warn once when called with a
  trainable W, B.
- **Backward dY clone in tests.** Unsloth bwd overwrites `dY` in place, so any
  test that re-uses `dY` across calls must `.clone()` first. This is a kernel
  contract, not a bug — but easy to trip on.
- **Liger backward partial buffer size.** `num_programs × H × 4 bytes` of fp32
  partials. At `H=4096` and 108 SMs on A100 that's ~1.7 MB — negligible. At
  much larger H it would matter.

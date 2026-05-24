# ForgeRMSNorm — Evolution Report (v1 → v2 → v3)

## §0 Executive summary

RMSNorm is the team's smallest kernel by FLOP count but among the most
frequently called — every transformer block uses two (input + post-attention).
This report covers the evolution from the pre-hackathon placeholder (v1) to
the shipping autotuned kernel (v3) that closes H7 and H8.

**Headline numbers (A100 80GB, qwen3_8b_short = 4×512×4096, bf16, forward only):**

| Version | Median (μs) | Speedup vs PyTorch eager | A100 HBM peak (2039 GB/s) used |
|---|---|---|---|
| pytorch_eager  | 822 | 1.00× | — |
| Liger (BSD-2)  |  76 | 10.8× | — |
| Unsloth        |  85 |  9.7× | — |
| **Forge v1**   |  57 | **14.4×** | — |
| **Forge v2**   |  29 | **28.3×** | — |
| **Forge v3**   |  32 | **25.4×** | **51%** (1035 GB/s) |

(Numbers from `kernels/rmsnorm/tests/results/v3_summary.md`. v3 occasionally
ties or slightly trails v2 on a single shape because autotune's chosen warps
configuration differs from v2's heuristic on this specific (shape, dtype) cell
— but it wins on average across the matrix and on the Gemma2-2B `gemma` path
where v2's warp heuristic is suboptimal.)

## §1 Context and constraints

**Scope (H7 + H8 hackathon items):**
- Port RMSNorm into the `forge.kernels.*` package so `forge.patch(model)` can
  activate it on Qwen3 and Gemma model classes.
- Add the Gemma `+1` offset path. The pre-hackathon `kernels/rmsnorm/rmsnorm.py`
  flagged "No Gemma `weight + 1` offset mode yet" in its own Known Boundaries.
- Pass `torch.autograd.gradcheck` at fp64 for both Qwen3 (offset=0) and Gemma
  (offset=1) paths. The pre-hackathon tests used `torch.testing.assert_close`
  against the PyTorch oracle but never exercised fp64 gradcheck — gap closed
  here.
- Stay FSDP2-safe (closures close over `module.weight` directly, no copies).
- "Small kernel, even small gain is fine" — correctness and completeness over
  absolute perf.

**Correctness oracle:** HF's `LlamaRMSNorm` (offset=0) and `Gemma2RMSNorm`
(offset=1). The in-repo oracle at `forge_rmsnorm_v2.py:torch_rmsnorm_reference_v2`
matches both forms.

## §2 Theory anchor — RMSNorm is bandwidth-bound

For a row of length H, RMSNorm reads `H` x-elements + `H` weight-elements and
writes `H` y-elements. That is `~4 bytes/element bf16 × 3 elements = 12 B/element`,
or ~12·B·S·H bytes of HBM traffic per call (read x, read w, write y; rstd is
small).

FLOPs are negligible relative to bytes: one `x²` multiply, one division by H,
one rsqrt, one `(x*rstd)*w` multiply per element ≈ 4 FLOPs/element. Roofline:
~0.33 FLOPs/byte at bf16 — same regime as LayerNorm and RoPE, far below A100's
~10 FLOPs/byte cusp. **The kernel is bandwidth-bound.**

For (b=2, s=2048, h=4096) bf16: total bytes ≈ 67 MB. Theoretical floor at A100's
2039 GB/s = **33 μs forward**. Forge v3 hits ~148 μs at this shape, so we're
~22% of the peak. The shorter-context shape (b=4, s=512, h=4096, bf16) hits
**51% of peak** (1035 GB/s @ 32 μs) — the "headline" number above.

## §3 v1 design — the placeholder

`kernels/rmsnorm/forge_rmsnorm_v1.py` (renamed from `rmsnorm.py` during the
hackathon). Properties:

- Grid `(n_rows,)`, one Triton program per row.
- Forward: load x in input dtype, cast to fp32 for rsqrt, cast back to input
  dtype, multiply by weight (in input dtype). Matches LlamaRMSNorm exactly.
- Backward: per-row-block partials with 16 rows/program. Buffer shape
  `(ceil(n_rows / 16), n_cols)`. Reduced via `dweight_partial.sum(0)` in Python.
- No offset support (Gemma broken at this version).
- No casting-mode flexibility (fp32 reduction is mandatory and good, but the
  affine multiply is always in input dtype — which precludes the Gemma
  fp32-throughout pattern).
- File header originally said `"These are placeholder files for testing patching."`

The v1 backward partial buffer over-allocates: at the Qwen3-8B train shape
(n_rows=4096), it produces `256 × 4096 × 4 B = 4 MB` of partials, then sums
them in Python — significant HBM thrash.

## §4 v2 changes — the production design

`kernels/rmsnorm/forge_rmsnorm_v2.py`. Three deltas, each justified:

### (a) `OFFSET: tl.constexpr` — the Gemma `+1`

Triton specializes a separate compiled binary per OFFSET value. Inside the
kernel, `(w + OFFSET)` fuses into the existing fp32 weight load with **zero
runtime cost**.

The alternative — applying the offset in the closure factory — was rejected
on memory and correctness grounds. It would either materialize a fresh
`weight + 1.0` tensor every forward (extra HBM alloc + read + write of a
`[H]` tensor) or cache it (violating the locked rule that closures must close
over `module.weight` directly, breaking LoRA).

### (b) `CASTING_MODE: tl.constexpr` — Llama vs Gemma fp32 policy

Three modes:

| Mode | Code | Where fp32 is dropped |
|---|---|---|
| `LLAMA` (0) | `(x_acc * rstd).to(input_dtype) * (w + OFFSET)` | After rstd, before affine. Matches `LlamaRMSNorm` and `Qwen3RMSNorm`. |
| `GEMMA` (1) | `(x_acc * rstd) * (w_acc + OFFSET)` then cast at store | After affine. Matches `Gemma2RMSNorm` where `1.0 + weight.float()` would lose precision in bf16 at near-zero weight init. |
| `NONE` (2) | Native dtype throughout (debug) | — |

The closure factory at `forge/forge/patching/core.py:_make_rmsnorm_forward`
defaults `casting_mode = "gemma" if offset == 1.0 else "llama"`, but the
mapping config can override.

### (c) SM-proportional dW partials backward

Replaces v1's `(ceil(n_rows / 16), n_cols)` partial buffer with
`(min(n_rows, sm_count), n_cols)`. At Qwen3-8B train shape on A100 (108 SMs):
**108 × 4096 × 4 B = 1.7 MB** partial buffer vs v1's **4 MB** — 2.3× smaller
final reduction, smaller HBM round-trip.

Atomics-free, matches the team-locked pattern from `kernels/layernorm/context.md §3`
("SM-count atomics on `[H]` from 108 SMs serialize hard"). The same shape Liger
uses in its single-row backward.

### (d) Accumulation dtype tracks input

`ACC_DTYPE: tl.constexpr` is `tl.float32` for bf16/fp16/fp32 inputs and
`tl.float64` for fp64 inputs. The host picks this from `x.dtype`. Without this,
the kernel would downcast fp64 inputs to fp32 internally — `torch.autograd.gradcheck`'s
1e-6 perturbations would be lost in the cast, producing zero numerical gradient
where the analytical gradient is non-zero. This was the bug that initially
prevented v2 from passing fp64 gradcheck and the fix that unblocked it.

### Backward derivation (offset-aware)

For `y_i = x_i * rstd * (w_i + o)` with `rstd = 1/√(mean(x²) + ε)`:

```
scaled_dy_i = dy_i * (w_i + o)               # in fp32
dot = Σ_i (scaled_dy_i * x_i)
dx_j = rstd * (scaled_dy_j - x_j * rstd² * dot / N)
dw_j = Σ_rows (dy_j * x_j * rstd)            # accumulated per-program-strip
```

The offset only affects `dx` (through `scaled_dy`). `dw` is independent of
offset — `dy/dw = x*rstd` either way.

## §5 v3 autotune surface

`kernels/rmsnorm/forge_rmsnorm_v3.py`. **Body identical to v2**, decorated with
`@triton.autotune` over:

```python
configs = [Config({}, num_warps=nw, num_stages=ns)
           for nw in (4, 8, 16) for ns in (2, 3)]
key = ["n_cols", "ACC_DTYPE"]
```

Six configs explored per (n_cols, ACC_DTYPE) cache cell. First call per cell
takes ~1.3 s (six configs compiled + benchmarked); cached calls are essentially
free (~0.13 ms vs first 1300 ms = **10000× speedup post-warmup**, well above
the 20× threshold the test asserts).

Expected gain over v2: 5–15% on shapes where v2's static heuristic (`num_warps`
picked by `BLOCK_SIZE` bucket) is suboptimal. Measured wins are biggest at
small-hidden Gemma shapes where v2 picks `num_warps=4` but v3 finds `num_warps=8`
better — at Gemma2-2B (H=2304) fp16 offset=0, v3 forward 36 μs vs v2 40 μs
= 1.1× extra over v2.

## §6 Measured numbers

See `kernels/rmsnorm/benchmarks/results/v3_summary.md` for the full table.
Key cells (median ms, bf16 forward):

| Shape | PT eager | Liger | Unsloth | Forge v1 | Forge v2 | **Forge v3** | v3 vs PT | v3 BW (GB/s) |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b_short (4×512×4096)   | 0.822 | 0.076 | 0.085 | 0.057 | 0.029 | **0.032** | 25.4× | 1035 |
| qwen3_8b_train (2×2048×4096)  | 2.059 | 0.150 | 0.139 | 0.171 | 0.171 | **0.148** | 13.9× |  454 |
| gemma2_9b (2×2048×3584)       | 1.104 | 0.095 | 0.069 | 0.062 | 0.078 | **0.067** | 16.4× |  874 |

## §7 Correctness verification

- `kernels/rmsnorm/tests/test_v2.py`: 42/42 forward, 12/12 backward, both
  fp64 gradchecks (offset=0 and offset=1) PASS.
- `kernels/rmsnorm/tests/test_v3.py`: 42/42 forward, 12/12 backward, both
  gradchecks PASS, autotune cache speedup 10323× (PASS).
- `kernels/rmsnorm/tests/test_v1.py`: 12/12 forward, 4/4 backward, gradcheck
  marked **expected failure** (v1 forces fp32 internal accumulation; fp64
  perturbations lost — v2's `ACC_DTYPE` fixes this).
- `tests/test_rmsnorm.py` (legacy + new): 14/14 pass including the new
  `test_rmsnorm_v2_gemma_offset` cases at bf16 and fp32.
- `forge/tests/verify_patch_qwen3.py` (extended): RMSNorm-only patch path
  added as `[4/6]` between embedding and all-kernels (see Step 4 in the
  bisection pattern).

## §8 Known limitations

Carried forward from the original `docs/rmsnorm.md` Known Boundaries plus what
v2/v3 add:

- **No no-affine mode** (weight=None) — not used by Qwen3 or Gemma. Deferred.
- **No DTensor / FSDP-sharded weight handling.** Liger has a `X.full_tensor()`
  path; we explicitly skip it (`_DTensor` is a sentinel that never matches in
  practice). The closure-factory contract (close over `module.weight` directly)
  is FSDP2-compatible at the parameter-update level; native DTensor inputs are
  a v-future concern.
- **Hidden dimensions > 131072** are rejected by `_calculate_settings`. Real
  models top out at ~16384; not a practical limit.
- **No in-place dY → dX backward** (Unsloth's memory-saver trick). Adds a
  "dY is corrupted after backward" footgun that conflicts with Gemma2's
  sequential RMSNorm + residual pattern. Deferred until LoRA path stabilizes.

## §9 Next steps (post-hackathon)

1. **FP8-aware mode** — TransformerEngine's `zero_centered_gamma` pattern is
   already what our OFFSET constexpr does. Plumb FP8 quantization into the
   forward + backward via a new casting mode. CP4 work.
2. **In-place dY → dX** for the non-residual path. Cuts dx allocation. CP2.
3. **Larger BLOCK_ROW** in the backward — currently each program handles a
   variable strip; a constexpr `ROWS_PER_PROGRAM` constexpr unroll may help at
   small n_rows. v4 candidate.
4. **DTensor input support** — when FSDP2 work lands the test harness, mirror
   Liger's `X.full_tensor()` branch via the existing `_DTensor` sentinel.

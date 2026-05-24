# ForgeRMSNorm — Comparative Analysis

Phase 2 (Comparative Study) deliverable: a per-dimension tradeoff table across
Liger, Unsloth, HF, Apex, and TransformerEngine, with Forge's locked decisions
and rationale. The raw upstream files live in `../rmsnorm_knowledge_base/0X_*/`.

## §1 Sources

| # | Source | License | Where in the repo |
|---|---|---|---|
| 1 | Unsloth | Apache 2.0 (✓ this file specifically) | `rmsnorm_knowledge_base/01_unsloth/` + `baselines/unsloth/` |
| 2 | Liger-Kernel | BSD-2-Clause | `rmsnorm_knowledge_base/02_liger/` + `baselines/liger/` |
| 3 | HF transformers (Llama / Qwen3 / Gemma2) | Apache 2.0 | `rmsnorm_knowledge_base/03_hf_transformers/` |
| 4 | NVIDIA Apex (`FusedRMSNorm`) | BSD-3-Clause | `rmsnorm_knowledge_base/04_apex/` (study only) |
| 5 | NVIDIA TransformerEngine | Apache 2.0 | `rmsnorm_knowledge_base/05_transformer_engine/` (study only) |

(Apex and TE are CUDA-only; we cite them as perf references but don't run them
in the benchmark suite — would require installing Apex/TE which adds
non-trivial setup. Their numbers in the comparison below come from each
project's published benchmarks.)

## §2 Tradeoff table — 8 dimensions

### D1: Grid shape (forward)

| Source | Grid | Notes |
|---|---|---|
| Liger | `(n_rows,)` if `BLOCK_SIZE>256 or n_rows<32K`; else `(ceil(n_rows/16), 1)` | Dynamic dual-kernel selection |
| Unsloth | `(n_rows,)` always | Simpler |
| Apex / TE | CUDA blocks per `n_rows`, threads tile `n_cols` | Production CUDA pattern |
| **Forge v2/v3** | `(n_rows,)` always | Liger's single-row path. We skip the block-row dual kernel — not worth the complexity at our hidden-size regime (H≥2304 for Gemma, H≥4096 for Qwen3/Llama). |

### D2: Casting mode (where fp32 ends)

| Source | Llama path | Gemma path |
|---|---|---|
| Liger | constexpr `casting_mode=0`: cast x*rstd to input dtype before w | constexpr `casting_mode=1`: all-fp32 through affine, cast at store |
| Unsloth | `_rms_layernorm_forward`: cast back before w | `_gemma_rms_layernorm_forward`: **separate kernel**, fp32 throughout |
| HF Llama | `weight * x.to(input_dtype)` | — |
| HF Gemma2 | — | `(1.0 + weight.float()) * x_fp32` then cast at return |
| **Forge v2/v3** | constexpr `CASTING_MODE=0` (LLAMA) | constexpr `CASTING_MODE=1` (GEMMA) |

Liger's pattern wins on code-size and on the cast-policy being a configurable
constexpr rather than a separate kernel. We adopt it.

### D3: Offset placement (Gemma `+1`)

| Source | Strategy |
|---|---|
| Liger | `tl.constexpr offset` parameter inside the kernel, fused into the existing fp32 weight load |
| Unsloth | Separate kernel `_gemma_rms_layernorm_forward` (no shared parameter — implicit `(W + 1.0)` in backward) |
| TransformerEngine | `zero_centered_gamma=True` flag on the host, forwarded to a CUDA kernel constexpr-like switch |
| **Forge v2/v3** | `OFFSET: tl.constexpr` (Liger pattern) |

Justification for OFFSET-in-kernel vs OFFSET-in-closure (which we considered
and rejected): the closure approach would either materialize a fresh
`weight + 1.0` tensor each forward (extra HBM alloc/read/write on a kernel
whose entire job is reading H bytes per row) or cache it (violating the locked
LoRA-safe rule that closures must close over `module.weight` directly).

### D4: Backward dW strategy

| Source | dW computation |
|---|---|
| Liger | Fused with dx: each program accumulates its strip's `dw` in fp32, writes one row of `(num_programs, n_cols)` partials, Python sums |
| Unsloth | Separate kernel call after the dX kernel; dY is repurposed as dX-buffer in non-Gemma mode |
| Apex | CUDA backend with shared-memory atomic reduction |
| **Forge v2/v3** | SM-proportional partials + Python `partial.sum(0)`. Matches Liger's pattern. `num_programs = min(n_rows, sm_count)` gives a 2.3× smaller buffer than v1's `ceil(n_rows / 16)`. |

We considered atomic `tl.atomic_add` to `dW` directly (no partials needed) and
rejected it — see `kernels/layernorm/context.md §3` for the team-locked
decision: SM-count atomic writes to `[H]` serialize on H4096-sized vectors.

### D5: Saved tensors

| Source | Saved |
|---|---|
| Liger | `(X, W, rstd)` — rstd in fp32 per row |
| Unsloth | `(X, W, rstd)` — same shape |
| Apex `memory_efficient` | `(Y, W)` — recompute normalization in backward |
| **Forge v2/v3** | `(X, W, rstd)`. Apex's recompute trick saves H·4 B/row of activation memory but adds one extra rsqrt + multiply in backward; deferred — useful for LoRA path later. |

### D6: DTensor / FSDP support

| Source | Strategy |
|---|---|
| Liger | `if isinstance(X, DTensor): X = X.full_tensor()` — gather to local before kernel |
| Unsloth | No DTensor handling |
| **Forge v2/v3** | No DTensor handling (out of scope for hackathon). Patching closures close over `module.weight` directly; FSDP2 weight-update semantics work via that, but native DTensor inputs are deferred. We do import a `_DTensor` sentinel in `baselines/liger/rms_norm.py` so the vendored code remains import-safe across torch 2.4-2.5 path differences. |

### D7: Block-size heuristic

| Source | num_warps picker |
|---|---|
| Liger | `num_warps = min(max(BLOCK_SIZE // 256, 1), 16)` — heuristic on block size |
| Unsloth | Hardcoded buckets: 4 for ≥512, 8 for ≥2048, 16 for ≥8192, 32 for ≥32768 |
| **Forge v2** | Same buckets as Unsloth (host-side `_calculate_settings`) |
| **Forge v3** | `@triton.autotune` over `num_warps ∈ {4, 8, 16} × num_stages ∈ {2, 3}` keyed on `(n_cols, ACC_DTYPE)` |

v3 trades a one-time ~1.3 s compile cost for a runtime cache hit on every
subsequent call. Production training will see the cache hit ratio approach 1.

### D8: In-place backward (dY → dX memory saver)

| Source | Strategy |
|---|---|
| Liger | `in_place=True` default; Gemma2 path uses `in_place=False` because its sequential RMSNorm + residual pattern needs `dY` preserved |
| Unsloth | Conditional on `GEMMA` flag — in-place when `False`, separate when `True` |
| **Forge v2/v3** | Always allocate fresh `dX`. Defers the in-place optimization until the LoRA path is wired (where the same "preserve dY for residual" pitfall applies). |

## §3 Forge's locked decisions

| Dim | Choice | Why |
|---|---|---|
| D1 Grid | Single-row only | H≥2304 saturates SMs already; dual-kernel adds complexity for marginal small-H gain |
| D2 Casting mode | 3-mode constexpr (LLAMA/GEMMA/NONE) | Liger pattern; matches HF Gemma2 fp32 affine policy exactly |
| D3 Offset | `tl.constexpr` (kernel) | Free; closure-factory alternative violates LoRA-safe rule |
| D4 Backward dW | SM-proportional partials | Atomics rejected per LayerNorm team decision; smaller-than-v1 buffer |
| D5 Saved tensors | `(X, W, rstd)` | Standard; recompute trick deferred |
| D6 DTensor | Skip | Out of scope; FSDP2 weight-update path works via closure |
| D7 Heuristic vs autotune | v2 heuristic, v3 autotune | "Build the deterministic version first; layer autotune on top" mirrors RoPE v2→v3 |
| D8 In-place backward | Skip | Footgun on residual paths; defer |

## §4 What Forge does NOT take from Liger

- The **block-row dual kernel** (`_block_rms_norm_forward_kernel`, BLOCK_ROW=16).
  Unnecessary complexity for our shape regime — we never hit the small-H/large-n_rows
  case that path is designed for.
- The **DTensor `.full_tensor()` branch** — out of scope.
- The **`in_place=True` backward**. Liger sets it to False for Gemma2 anyway;
  our always-out-of-place backward avoids the conditional.

## §5 Open questions for v4+

- **In-place backward with explicit "needs residual" override.** When is the
  memory saving worth the API complexity? Probably during LoRA training where
  activation memory is the bottleneck.
- **Dual single-row + block-row kernel** — does the block-row path actually win
  on small-H Gemma shapes (e.g. Gemma-3 270M)? Worth measuring before adopting.
- **FP8-aware mode** — TransformerEngine's pattern. CP4 territory.
- **Welford's algorithm vs the current `sum(x²)/N` reduction** — Welford
  improves numerical stability at very long H. We're far from where it matters
  (H≤16384 in production), but worth a footnote.

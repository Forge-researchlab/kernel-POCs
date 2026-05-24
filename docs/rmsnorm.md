# RMSNorm

> **Status (2026-05-24, post-hackathon):** v1 (placeholder, this doc), v2
> (Liger-style: offset constexpr + 3 casting modes + SM-proportional dW), and
> v3 (v2 + `@triton.autotune`) all live under `kernels/rmsnorm/`. The shipping
> kernel is **v3** (re-exported as `forge.kernels.rmsnorm.apply_rmsnorm`).
> Detailed evolution story + measured numbers + comparative analysis live at
> `kernels/rmsnorm/docs/{evolution_report.md, comparative_analysis.md}`.

## Scope

This doc describes Forge RMSNorm **v1** — the placeholder baseline kept as
the no-offset comparison point. v2/v3 supersede it for production use
(Gemma support, fp64 gradcheck, autotune). It implements
the Llama/Qwen-style operation:

```python
x_fp32 = x.float()
rstd = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
y = (x_fp32 * rstd).to(x.dtype) * weight
```

The public patch-friendly entry point is `kernels.rmsnorm.rmsnorm(x, weight,
eps=1e-6)`. On CUDA it uses the Triton autograd path. On CPU it falls back to
the PyTorch reference so model patching tests can still run without a GPU.

## Backward Math

For one row with hidden size `H`:

```text
r = rsqrt(mean(x^2) + eps)
y_i = x_i * r * w_i
g_i = dy_i * w_i
dx_i = r * (g_i - x_i * r^2 * sum_j(g_j * x_j) / H)
dw_i = sum_rows(dy_i * x_i * r)
```

The backward Triton kernel computes `dx` and per-row-block partial `dw` values.
The final `dw` reduction currently uses `torch.sum` over those partials. This is
simple and stable enough for a placeholder, but not the final memory-optimal
design.

## Supported Surface

- Input shape: any tensor with hidden dimension in the last axis.
- Weight shape: 1D tensor of length `x.shape[-1]`.
- Dtypes: `fp32`, `fp16`, and `bf16` on CUDA, with fp32 accumulation. Output
  dtype follows PyTorch promotion between `x` and `weight`.
- Layout: inputs are made contiguous internally.
- CUDA path: Triton forward and backward via `ForgeRMSNormFunction`.
- CPU path: PyTorch fallback through `rmsnorm()`.

## Competitor Notes

- Liger has a full RMSNorm implementation with offset/casting modes, DTensor
  handling, optional in-place backward, and a more complete parameter-gradient
  reduction.
- Unsloth has a compact RMS layernorm kernel and a Gemma variant. Its backward
  path focuses on `dx`; Liger extends this into a broader HuggingFace-compatible
  API.

This Forge version intentionally does not copy those full APIs yet. It gives the
patching work a stable, minimal kernel surface for Qwen/Llama-style RMSNorm.

## Known Boundaries (v1 placeholder — closed in v2/v3)

- ~~No Gemma `weight + 1` offset mode yet.~~ → **Closed in v2** via `OFFSET: tl.constexpr`.
- No no-affine mode yet. (Still deferred — not used by Qwen3 or Gemma.)
- No distributed tensor support yet. (Still deferred for FSDP2.)
- Hidden dimensions requiring a Triton block larger than `131072` are rejected.
- ~~The current `dw` implementation materializes partial gradients and should be
  replaced before serious benchmark claims.~~ → **Improved in v2** via
  SM-proportional partials (`(min(n_rows, sm_count), n_cols)` vs v1's
  `(ceil(n_rows/16), n_cols)`).

See `kernels/rmsnorm/docs/evolution_report.md` for the v2/v3 design and
measured perf, and `kernels/rmsnorm/docs/comparative_analysis.md` for the
Liger/Unsloth/HF/Apex/TE tradeoff table.

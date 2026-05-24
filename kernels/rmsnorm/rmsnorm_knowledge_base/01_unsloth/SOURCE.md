# Unsloth — RMS LayerNorm Triton kernel

- **Upstream URL:** https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/kernels/rms_layernorm.py
- **Repo:** unslothai/unsloth
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-24
- **License:** Apache-2.0 (file header). Unlike `rope_embedding.py` in the same repo (which has an LGPL-3.0-or-later file header), this file is consistently Apache-2.0 with the repo top-level LICENSE. Safe to study and adopt design patterns.
- **What this is:** Three Triton kernels — `_rms_layernorm_forward` (Llama/Qwen path), `_gemma_rms_layernorm_forward` (fp32-throughout for Gemma), `_rms_layernorm_backward` — plus a `Fast_RMS_Layernorm(autograd.Function)` and the `fast_rms_layernorm(layernorm_module, X, gemma=False)` high-level entry point that pulls `weight` and `eps` from an `nn.Module`.
- **Key design choices:**
  - Single row per Triton program (`tl.program_id(0)`); grid = `(n_rows,)`.
  - `GEMMA` flag splits forward into two distinct kernels rather than carrying a constexpr branch — duplicates the code but keeps each kernel tight.
  - In-place backward when `GEMMA=False`: `dX = dY` is reused as the dX buffer (memory saver). Gemma path allocates a separate dX because residual paths need dY preserved.
  - Backward saves `(X, W, rstd)`; reduction-style fused `dw` is **not** computed inside the kernel — Unsloth's `_rms_layernorm_backward` returns only `dX` and computes `dW` elsewhere (look for `dW = X` comment around line 215).
- **What to read first:** `_rms_layernorm_forward` (the Llama path), then `_gemma_rms_layernorm_forward` to see exactly how the fp32-throughout treatment diverges. Then `_rms_layernorm_backward` and the `Fast_RMS_Layernorm.backward` wrapper.

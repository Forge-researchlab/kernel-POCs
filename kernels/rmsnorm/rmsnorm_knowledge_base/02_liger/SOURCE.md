# Liger-Kernel — RMSNorm (dual kernel, offset constexpr, 3 casting modes)

- **Upstream URLs:**
  - https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/rms_norm.py
  - https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/transformers/rms_norm.py
- **Repo:** linkedin/Liger-Kernel
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-24
- **License:** BSD-2-Clause. The file header notes that the design "incorporates code from Unsloth licensed under the Apache License, Version 2.0" (Unsloth's `rms_layernorm.py` Apache-2.0 path, not the LGPL `rope_embedding.py`). Modifications by Yanning Chen, 2024. Direct pattern reuse with attribution is OK for Forge OSS launch.
- **What this is:** A production-grade Triton RMSNorm with four kernels:
  - `_rms_norm_forward_kernel` — single row per program (the path that handles Qwen3 hidden=4096).
  - `_rms_norm_backward_kernel` — fused per-row `dx` + per-program-strip `dw` partials (`tl.atomic_add` is **not** used; partials are reduced outside the kernel).
  - `_block_rms_norm_forward_kernel`, `_block_rms_norm_backward_kernel` — block-rows variant (BLOCK_ROW=16) chosen when `BLOCK_SIZE ≤ 256` and `n_rows ≥ 32K`. Smaller-hidden / larger-batch regime.
- **Key design choices:**
  - **`offset: tl.constexpr`** — Gemma `+1` baked at compile time into the existing fp32 weight load.
  - **3 casting modes** (`_CASTING_MODE_LLAMA=0`, `_CASTING_MODE_GEMMA=1`, `_CASTING_MODE_NONE=-1`) — controls where the cast-back-to-input-dtype happens. Llama casts after rstd, Gemma stays in fp32 through the affine multiply.
  - **Dynamic kernel selection** in `rms_norm_forward`: `if BLOCK_SIZE > 256 or n_rows < 4096*8 or row_mode: <single-row>; else <block-rows>`. SM-proportional partials in backward.
  - **DTensor / TP gathering** — `if isinstance(X, DTensor): X = X.full_tensor()`. Useful pattern reference for FSDP2 work; not adopted in Forge v2.
  - **`in_place` backward** — defaults True, set False for residual-paired RMSNorms (Gemma2 uses two RMSNorms sequentially with a residual between them).
- **What to read first:** `_rms_norm_forward_kernel` (the path Forge v2 mirrors), then `_rms_norm_backward_kernel` for the SM-proportional partial-dW design, then `rms_norm_forward` host function for the dual-kernel routing.

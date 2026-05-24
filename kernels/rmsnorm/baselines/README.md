# RMSNorm Baselines

Vendored, self-contained implementations of competitor RMSNorm kernels. Used by `benchmarks/bench_v{2,3}.py` and `tests/test_v{2,3}.py` to validate and benchmark `ForgeRMSNorm` against:

1. **Liger-Kernel** — dual single-row + block-row Triton kernels with explicit `offset` constexpr and 3 casting modes. BSD-2-Clause.
2. **Unsloth** — single-row Triton kernel with a separate `_gemma_rms_layernorm_forward` variant that forces fp32 throughout. Apache-2.0.

Both vendored on **2026-05-24** from upstream `main`. Original sources also live in `../rmsnorm_knowledge_base/0{1,2}_*/` for reference reading.

## Unified API

```python
from kernels.rmsnorm.baselines import liger, unsloth

# Qwen3 / Llama style — offset=0, casting_mode="llama" (rstd-only fp32)
y_liger   = liger.apply_rmsnorm(x, weight, eps=1e-6, offset=0.0, casting_mode="llama")
y_unsloth = unsloth.apply_rmsnorm(x, weight, eps=1e-6, offset=0.0)

# Gemma style — offset=1.0, casting_mode="gemma" (all-fp32 through the affine multiply)
y_liger   = liger.apply_rmsnorm(x, weight, eps=1e-6, offset=1.0, casting_mode="gemma")
y_unsloth = unsloth.apply_rmsnorm(x, weight, eps=1e-6, offset=1.0)
```

### Shape contract

| Tensor | Shape | Notes |
|---|---|---|
| `x`      | `(..., H)` | bf16 / fp16 / fp32. Normalized over the last dim. |
| `weight` | `(H,)`     | Affine scale. Gemma weight inits at zero — offset shifts it. |

Returns a tensor with the same shape and dtype as `x`.

## Layout

```
baselines/
├── README.md                  # this file
├── __init__.py                # exposes `liger` and `unsloth` submodules
├── liger/
│   ├── __init__.py            # apply_rmsnorm(x, weight, eps, offset, casting_mode)
│   ├── rms_norm.py            # vendored from Liger-Kernel/main, import-patched
│   └── utils.py               # minimal stub of liger_kernel.ops.utils
└── unsloth/
    ├── __init__.py            # apply_rmsnorm(x, weight, eps, offset, casting_mode)
    ├── rms_layernorm.py       # vendored from Unsloth/main (Apache-2.0)
    └── utils.py               # minimal stub of unsloth.kernels.utils
```

## License posture

- **Liger** (BSD-2-Clause): compatible with Forge OSS launch. Forge v2's offset-constexpr + casting-mode design follows Liger's pattern.
- **Unsloth `rms_layernorm.py`** (**Apache-2.0**): compatible. Note this differs from Unsloth's `rope_embedding.py` which is LGPL — the RMSNorm file is the safer one. Confirmed via the file's own header (`Copyright 2023-present Daniel Han-Chen & the Unsloth team. ... Licensed under the Apache License, Version 2.0`).

## Updating from upstream

1. Re-fetch into `../rmsnorm_knowledge_base/0{1,2}_*/` per that folder's `SOURCE.md`.
2. `cp` the new file into `baselines/{liger,unsloth}/` and re-apply the import patches:
   - `liger/rms_norm.py`: change `from liger_kernel.ops.utils import …` and `from liger_kernel.utils import is_npu_available` to `from .utils import …`.
   - `unsloth/rms_layernorm.py`: no patch needed; it already imports from `.utils` (single-GPU only — module-load skips multi-device branches automatically).
3. Smoke-test:
   ```python
   import torch
   from kernels.rmsnorm.baselines import liger, unsloth
   x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
   w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
   liger.apply_rmsnorm(x, w, offset=1.0, casting_mode="gemma")  # Gemma path
   unsloth.apply_rmsnorm(x, w, offset=1.0)                       # Gemma path
   ```

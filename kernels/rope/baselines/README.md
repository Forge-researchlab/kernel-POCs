# RoPE Baselines

Vendored, self-contained implementations of competitor RoPE kernels. Used by `benchmarks/bench_rope.py` and `tests/test_rope.py` to validate `ForgeRoPE` against:

1. **Liger-Kernel** — single-launch fused Q+K. BSD-2-Clause.
2. **Unsloth** — two entry points: default (separate Q and K launches) and fused-QK (single launch, originally TRL-specific).

Both vendored on **2026-05-23** from upstream `main`. Original sources also live in `../rope_knowledge_base/` for reference reading.

## Unified API

```python
from kernels.rope.baselines import liger, unsloth

q_out, k_out = liger.apply_rope(q, k, cos, sin)
q_out, k_out = unsloth.apply_rope(q, k, cos, sin)            # separate launches (default)
q_out, k_out = unsloth.apply_rope_qk_fused(q, k, cos, sin)   # genuinely fused
```

### Shape contract (HF Qwen3)

| Tensor | Shape | Notes |
|---|---|---|
| `q` | `(batch, n_q_heads, seq_len, head_dim)` | bf16 / fp16 / fp32 |
| `k` | `(batch, n_kv_heads, seq_len, head_dim)` | `n_kv_heads ≤ n_q_heads` (GQA) |
| `cos`, `sin` | `(1, seq_len, head_dim)` or `(batch, seq_len, head_dim)` | Full `head_dim`, left half == right half |

Returns `(q_out, k_out)` with the same shapes as inputs.

## Smoke-test status (verified 2026-05-23 on this machine)

All three baselines compile, run, and match the HF `apply_rotary_pos_emb` reference within rounding tolerance:

| Dtype | HF vs Liger | HF vs Unsloth-default | HF vs Unsloth-fused-QK |
|---|---|---|---|
| bf16 | 0.0 | 0.0 | 0.0 |
| fp16 | 1.95e-3 | 1.95e-3 | 1.95e-3 |
| fp32 | 2.38e-7 | 2.38e-7 | 2.38e-7 |

(Shape `(b=2, n_q=4, n_kv=2, s=16, d=64)`, max abs diff.)

## Layout

```
baselines/
├── README.md                 # this file
├── __init__.py               # exposes `liger` and `unsloth` submodules
├── liger/
│   ├── __init__.py           # apply_rope(q, k, cos, sin)
│   └── rope.py               # vendored verbatim from Liger-Kernel/main
└── unsloth/
    ├── __init__.py           # apply_rope (default) + apply_rope_qk_fused
    ├── rope_embedding.py     # vendored from Unsloth/main, import patched
    └── utils.py              # minimal stub of Unsloth helpers (calculate_settings, etc.)
```

## License posture

- **Liger** (BSD-2-Clause): Compatible with Forge OSS launch.
- **Unsloth** (file headers say **LGPL-3.0-or-later**, repo top-level LICENSE is Apache-2.0): Reference reading and benchmarking is fine. **Do not copy patterns** from `rope_embedding.py` into `forge/kernels/rope.py`. Forge's own kernel must be written from the design doc, not paraphrased from this file. See `memory/feedback_license_unsloth.md`.

## Updating

To refresh from upstream:
1. Re-fetch into `../rope_knowledge_base/0X_<source>/` (see that folder's `SOURCE.md`).
2. `cp` the new file here and re-apply the import patches:
   - `unsloth/rope_embedding.py`: change `from ..device_type import DEVICE_COUNT` and `from .utils import ...` to `from .utils import DEVICE_COUNT, calculate_settings, torch_gpu_device, torch_device_stream`.
   - `liger/rope.py`: no patches needed (self-contained).
3. Re-run the smoke-test snippet in this README to confirm.

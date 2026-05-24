# RMSNorm Kernel — Knowledge Base

Reference implementations of Root Mean Square Normalization from 5 sources, collected as Phase 2 (Comparative Study) input for the H7 + H8 RMSNorm completion (Forge Hackathon, May 23-24, 2026).

**Purpose:** Inform our design choices for `kernels/rmsnorm/forge_rmsnorm_v2.py` and `forge_rmsnorm_v3.py`. This folder holds raw source code only — analysis (tradeoff table, design choice) lives in `../docs/comparative_analysis.md`.

**Fetch date:** 2026-05-24

---

## Reading order (priority)

| # | Source | Folder | Status | Why read it |
|---|--------|--------|--------|-------------|
| 6 | **Forge internal curriculum** | `06_forge_internal/` | ⏳ Pending | Team's pre-existing math + Triton position. Read first when available. |
| 3 | **HF transformers (Llama + Qwen3 + Gemma2)** | `03_hf_transformers/` | ✅ | The correctness oracles. Forge output must match Qwen3RMSNorm (offset=0) and Gemma2RMSNorm (offset=1, cast-throughout-fp32). |
| 2 | **Liger-Kernel** | `02_liger/` | ✅ | The design Forge v2 mirrors — explicit `offset` constexpr, 3 casting modes, SM-proportional dW partials, dual single-row/block-row kernels. BSD-2 — safe to pattern-match. |
| 1 | **Unsloth** | `01_unsloth/` | ✅ | Two-kernel design (separate `_gemma_rms_layernorm_forward`); informs the "split-by-fp32-policy" alternative we considered and rejected for v2. Apache-2.0 (NOT LGPL like Unsloth's RoPE). |
| 5 | **NVIDIA TransformerEngine** | `05_transformer_engine/` | ✅ | Production CUDA reference. `zero_centered_gamma` flag is the same offset-constexpr pattern. FP8-aware — informs Forge v3+ FP8 work. |
| 4 | **NVIDIA Apex** | `04_apex/` | ✅ | Older production CUDA reference (`FusedRMSNorm`). Includes `memory_efficient` mode that recomputes in backward — informs the activation-memory tradeoff we consider for the LoRA path. |

---

## One-line orientation per source

- **Unsloth** (`01_unsloth/rms_layernorm.py`, ~330 lines): three Triton kernels — `_rms_layernorm_forward`, `_gemma_rms_layernorm_forward` (fp32-throughout), `_rms_layernorm_backward`. Entry point: `fast_rms_layernorm(layernorm_module, X, gemma=False)`. Backward writes dX in-place into dY when `gemma=False` for memory savings.
- **Liger** (`02_liger/rms_norm.py` ~700 lines + `rms_norm_module.py` ~125 lines): four Triton kernels — single-row fwd/bwd plus block-row fwd/bwd (BLOCK_ROW=16) for small-hidden / large-batch regime. Has `offset` constexpr + 3 casting modes + DTensor handling. SM-proportional dW partials in backward.
- **HF Llama/Qwen3** (`03_hf_transformers/modeling_{llama,qwen3}.py`): `LlamaRMSNorm` / `Qwen3RMSNorm` — pure PyTorch, weight init = ones, formula `weight * (x * rsqrt(mean(x²) + eps))`.
- **HF Gemma2** (`03_hf_transformers/modeling_gemma2.py`): `Gemma2RMSNorm` — pure PyTorch, weight init = zeros, formula `(1.0 + weight.float()) * (x_fp32 * rsqrt(...))`. The +1 lives on weight, fp32 throughout the affine multiply.
- **Apex** (`04_apex/fused_layer_norm.py`, ~1100 lines): `FusedRMSNorm` nn.Module + `FusedRMSNormAffineFunction` autograd Function. CUDA backend with Welford reduction. `memory_efficient` mode recomputes normalization in backward.
- **TransformerEngine** (`05_transformer_engine/rmsnorm.py`, ~150 lines): PyTorch wrapper around TE's C++/CUDA `tex.rmsnorm_fwd`/`tex.rmsnorm_bwd`. FP8-aware. `zero_centered_gamma=True` is Gemma offset pattern.

---

## Open questions this knowledge base should answer

Phase 2 analysis (`../docs/comparative_analysis.md`) needs to fill out a tradeoff table with these dimensions:

1. **Grid shape** — single row per program vs block-rows. When does each win?
2. **Casting mode** — Llama (rstd-only fp32) vs Gemma (all-fp32) vs None. When does the difference visibly affect bf16 numerics?
3. **Offset placement** — kernel constexpr (Liger) vs separate kernel (Unsloth) vs in-closure tensor materialization. Memory + perf tradeoff.
4. **Backward dW strategy** — SM-proportional partials + Python reduction (Liger, Forge v2) vs in-place dX = dY trick (Unsloth) vs atomic-add (rejected by team per `kernels/layernorm/context.md`).
5. **Saved tensors** — rstd per row vs recompute (Apex's memory_efficient).
6. **DTensor / TP awareness** — Liger has it; out of scope for Forge v2.
7. **FP8 / zero-centered gamma** — TE has it as a first-class flag; informs Forge's offset constexpr.
8. **In-place backward correctness pitfalls** — Unsloth's dX = dY breaks residual paths (why Gemma2 needs the separate-buffer path); Forge v2 always allocates a fresh dX for safety.

The locked decisions and the per-dimension Forge choice live in `../docs/comparative_analysis.md`.

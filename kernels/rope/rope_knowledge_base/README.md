# RoPE Kernel — Knowledge Base

Reference implementations of Rotary Position Embedding from 7 sources, collected as Phase 2 (Comparative Study) input for the H2 RoPE Fused Q+K kernel (Forge Hackathon, May 23-24, 2026).

**Purpose:** Inform our design choices for `forge/kernels/rope.py`. This folder holds raw source code only — analysis (tradeoff table, design choice) lives separately.

**Fetch date:** 2026-05-23

---

## Reading order (priority)

| # | Source | Folder | Status | Why read it |
|---|--------|--------|--------|-------------|
| 6 | **Forge internal curriculum** | `06_forge_internal/` | ⏳ Pending | Team's pre-existing math + Triton position. **Read first when available.** |
| 3 | **HF transformers (Qwen3)** | `03_hf_transformers/` | ✅ | The correctness oracle — our Triton kernel's output must match `apply_rotary_pos_emb` within rtol=1e-5. |
| 1 | **Unsloth** | `01_unsloth/` | ✅ ⚠️ LGPL | The design prior we're validating: single fused Q+K kernel launch. **Read for ideas, not for copy-paste** (see license note below). |
| 4 | **Megatron-LM** | `04_megatron/` | ✅ | Training-scale integration pattern. The Python file is a dispatcher — fused work happens in TransformerEngine. |
| 2 | **Liger-Kernel** | `02_liger/` | ✅ | Contrast point: applies RoPE to Q and K in separate launches. Validates *why* fusion is the right call. |
| 5 | **NVIDIA TransformerEngine** | `05_transformer_engine/` | ✅ | The actual production fused CUDA kernel Megatron calls. Includes both Q+K-fused and QKV-fused variants. |
| 7 | **TorchTitan** | `07_torchtitan/` | ✅ | Canonical PyTorch reference. Shared RoPE module covers Llama3/4 + DeepSeek V3 + Qwen3 + GPT-OSS, with YaRN scaling. |

---

## One-line orientation per source

- **Unsloth** (`01_unsloth/rope_embedding.py`, 465 lines): `_rope_embedding_QK` Triton kernel + `Fast_RoPE_Embedding_QK` autograd.Function — fuses Q and K in one launch. Entry point: `fast_rope_embedding`.
- **Liger** (`02_liger/rope.py` 239 + `rope_module.py` 64): standard Triton RoPE with Q and K applied in **separate launches**. Use as a baseline to quantify the fused-launch win.
- **HF Qwen3** (`03_hf_transformers/modeling_qwen3_rope.py`): `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` + `rotate_half`. Pure PyTorch — the function `forge.patch` will replace.
- **Megatron-LM** (`04_megatron/rotary_pos_embedding.py` 367 + `fused_rope.py` 351): `RotaryEmbedding` + `MultimodalRotaryEmbedding` classes; `rope_utils.py` is a dispatcher to TransformerEngine's fused kernels. Supports BSHD/THD layouts. Shows the *integration* pattern at scale.
- **TransformerEngine** (`05_transformer_engine/rope.py` 541 + `fused_rope.cu` 716): `FusedRoPEFunc` + `FusedQKVRoPEFunc` (the latter goes one step further — fuses RoPE with the QKV projection). CUDA backend supports THD/SBHD/BSHD layouts and context-parallel dual-chunk indexing. `sincosf` + shared-memory cos/sin caching.
- **TorchTitan** (`07_torchtitan/llama_rope.py`, 408 lines): a single `RoPE` `nn.Module` covering Llama3/4 (`apply_rotary_emb_complex`, `torch.polar` complex layout), DeepSeek-V3 MLA (`apply_rotary_emb_single_complex`), and **Qwen3 / GPT-OSS** (`apply_rotary_emb_cos_sin`). Includes YaRN frequency scaling.

---

## Open questions this knowledge base should answer

Phase 2 analysis (next step — separate doc) needs to fill out a tradeoff table with these dimensions:

1. **HBM reads/writes per call** — the bandwidth budget that determines kernel speed
2. **Grid shape** — does Unsloth's `(num_rows, 2)` for Q+K hold up, or does TE do something better?
3. **GQA handling** — Qwen3 has fewer K heads than Q heads. How does each source handle the shape asymmetry inside a fused launch?
4. **cos/sin layout** — recomputed in-kernel vs loaded from HBM; cached on module vs per-call
5. **Rotation layout** — split-half (HF, TorchTitan cos/sin path) vs interleaved (original RoFormer) vs complex-number (TorchTitan Llama path)
6. **Partial RoPE** (`rotary_dim < head_dim`) — Qwen3 doesn't use this, but our API should leave room
7. **Backward strategy** — re-launch fwd with negated sin (Unsloth) vs dedicated bwd kernel (TE)
8. **Saved tensors** — RoPE backward only needs cos/sin; confirm all sources agree
9. **Numerical precision** — fp32 accumulation in bf16 path
10. **`base` parameterization + scaling extensibility** — Gemma uses different base; YaRN/NTK-aware scaling (TorchTitan demonstrates) should be reachable from our API

---

## License notes

| Source | License | Reuse posture for Forge OSS launch (CP5) |
|--------|---------|--------------------------------------------|
| Unsloth | ⚠️ **LGPL-3.0-or-later** (file header) despite Apache repo LICENSE | **Reference reading only.** Copying patterns requires clean-room rewrite or explicit clearance. |
| Liger-Kernel | BSD-2-Clause | Compatible — direct pattern reuse with attribution is OK. |
| HF transformers | Apache-2.0 | Compatible. |
| Megatron-LM | BSD-3-Clause (verify against repo LICENSE) | Compatible. |
| TransformerEngine | Apache-2.0 | Compatible. |
| TorchTitan | BSD-3-Clause | Compatible. |

The Unsloth posture is **not** a "don't read it" — it's the reference the hackathon plan tells us to validate against. It just means our final code in `forge/kernels/rope.py` needs to be written from the math + design decisions, not pattern-matched against Unsloth's file.

See `kernel-POCs/memory/feedback_license_unsloth.md` for the durable note.

---

## Folder layout

Each subfolder contains:
- The fetched source file(s) — raw, verbatim from upstream
- `SOURCE.md` — upstream URL, ref, fetch date, license, one-line orientation, "what to read first"

`06_forge_internal/` is a placeholder — see its README for what to drop in once available.

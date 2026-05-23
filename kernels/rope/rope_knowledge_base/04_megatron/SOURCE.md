# Megatron-LM — Rotary Positional Embedding (training-scale reference)

- **Upstream URL(s):**
  - https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py
  - https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/models/common/embeddings/rope_utils.py
- **Repo:** NVIDIA/Megatron-LM
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** BSD-3-Clause (or check the LICENSE file at root)
- **What this is:** Megatron-LM's RoPE used in NVIDIA's large-scale LLM training stack. GQA-aware, supports interleaved/non-interleaved layouts, integrates with TransformerEngine fused kernels.
- **What to read first:** the rotary embedding module class, then `apply_rotary_pos_emb` (or whatever the apply function is called here).

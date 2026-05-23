# HuggingFace transformers — Qwen3 RoPE (correctness oracle)

- **Upstream URL:** https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py
- **Repo:** huggingface/transformers
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** Apache-2.0
- **What this is:** Reference pure-PyTorch implementation of RoPE used by Qwen3 in production HF Transformers. Our Triton kernel's output must match this within rtol=1e-5.
- **What was extracted:** `Qwen3RotaryEmbedding` class (with `compute_default_rope_parameters` and `forward`), `apply_rotary_pos_emb` function, `rotate_half` helper function.
- **What to read first:** `apply_rotary_pos_emb` — this is the function `forge.patch` will replace.

# Unsloth — Fused RoPE (Q+K) Kernel

- **Upstream URL:** https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/kernels/rope_embedding.py
- **Repo:** unslothai/unsloth
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** ⚠️ **Conflict** — file header declares `LGPL-3.0-or-later`; repo top-level `LICENSE` is Apache-2.0. Unsloth's kernel files have historically used LGPL while the package as a whole ships under Apache. **Treat this file as LGPL for any OSS-Forge reuse decisions.** Reference-only reading is fine; copying patterns into our Apache/BSD release needs a clean-room rewrite or explicit license clearing.
- **What this is:** Fused forward + backward Triton kernel that applies rotary embeddings to Q and K (with optional per-token rope indices), launchable either as a single Q+K fused kernel or as a grouped per-tensor kernel.
- **What to read first:** `_rope_embedding_QK` Triton kernel and its `Fast_RoPE_Embedding_QK` autograd.Function wrapper (entry point: `fast_rope_embedding`).

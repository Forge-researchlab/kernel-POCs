# NVIDIA TransformerEngine — Fused RoPE (Hopper-tuned)

- **Upstream URL(s):**
  - https://raw.githubusercontent.com/NVIDIA/TransformerEngine/main/transformer_engine/pytorch/attention/rope.py
  - https://raw.githubusercontent.com/NVIDIA/TransformerEngine/main/transformer_engine/common/fused_rope/fused_rope.cu
- **Repo:** NVIDIA/TransformerEngine
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** Apache-2.0
- **What this is:** NVIDIA's production-grade fused RoPE used in TransformerEngine. Often the first to land FP8 / Hopper-specific optimizations.
- **What to read first:** the Python entry point (autograd.Function or similar), then the CUDA kernel if fetched.

# NVIDIA Apex — FusedRMSNorm (production CUDA reference)

- **Upstream URL:** https://raw.githubusercontent.com/NVIDIA/apex/master/apex/normalization/fused_layer_norm.py
- **Repo:** NVIDIA/apex
- **Branch/ref:** master (HEAD as of fetch)
- **Fetch date:** 2026-05-24
- **License:** BSD-3-Clause
- **What this is:** NVIDIA's production CUDA `FusedRMSNorm` — the kernel that Megatron-LM and many fine-tuning stacks called into before TransformerEngine became dominant. Used here as a **perf reference point**, not a runtime baseline (would require building Apex from source).
- **Key design choices:**
  - `FusedRMSNormAffineFunction.forward(ctx, input, weight, normalized_shape, eps, memory_efficient)` — same surface as Apex's LayerNorm cousin, takes a `normalized_shape` tuple rather than a flat hidden dim.
  - `memory_efficient` mode saves only `(output, weight)` and recomputes the normalization in backward, trading FLOPs for activation memory.
  - CUDA backend (`fused_layer_norm_cuda_kernel.cu`) uses Welford-style two-pass reduction with shared-memory accumulators.
- **What to read first:** `FusedRMSNorm` (the nn.Module wrapper, ~line 200), then `FusedRMSNormAffineFunction` (the autograd Function, ~line 100). The CUDA kernel source isn't in this file — it lives in `fused_layer_norm_cuda_kernel.cu` in the same repo and is non-trivial to pull without the rest of the Apex build system.
- **How Forge uses this:** Comparison reference in `docs/comparative_analysis.md` — "Apex's memory-efficient mode" cited for the recompute-vs-save tradeoff. Not invoked at benchmark runtime.

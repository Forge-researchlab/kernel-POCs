# NVIDIA TransformerEngine — RMSNorm (Hopper-tuned, FP8-aware)

- **Upstream URL:** https://raw.githubusercontent.com/NVIDIA/TransformerEngine/main/transformer_engine/pytorch/module/rmsnorm.py
- **Repo:** NVIDIA/TransformerEngine
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-24
- **License:** Apache-2.0
- **What this is:** The PyTorch wrapper around TransformerEngine's CUDA RMSNorm. The CUDA kernel itself lives in `transformer_engine/common/normalization/rmsnorm/rmsnorm.cu` (not fetched — large + requires the rest of the TE build to be useful). Production reference for Hopper SM90 / FP8 RMSNorm.
- **Key design choices:**
  - `_RMSNorm.forward` dispatches between fwd/bwd through the C++/CUDA backend via `tex.rmsnorm_fwd` / `tex.rmsnorm_bwd`.
  - FP8 mode is first-class: forward returns both the bf16 output and an FP8-quantized version when called within an `fp8_autocast` context.
  - Welford's algorithm with shared-memory accumulators in the CUDA backend (similar to Apex but Hopper-tuned with WGMMA-style scheduling).
  - Zero-centered weight option (`zero_centered_gamma=True` — adds 1 to weight before the affine multiply, the Gemma pattern). This is exactly the offset-constexpr design that Liger and Forge v2 adopt.
- **What to read first:** `_RMSNorm` autograd Function — note how `zero_centered_gamma` is forwarded to the CUDA call. That's the same hook Forge's `OFFSET` constexpr plays.
- **How Forge uses this:** Reference perf data point ("TE bf16 forward on Hopper hits ~X% of HBM peak"). Cited in `docs/comparative_analysis.md`. Not invoked at benchmark runtime.

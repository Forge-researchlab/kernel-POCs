# HuggingFace Transformers — RMSNorm reference oracles

- **Upstream URLs:**
  - https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py
  - https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py
  - https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma2/modeling_gemma2.py
- **Repo:** huggingface/transformers
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-24
- **License:** Apache-2.0
- **What this is:** Pure-PyTorch reference implementations of RMSNorm for three architectures Forge targets. They are the **correctness oracles** — Forge's Triton kernel must match these within bf16 noise.
- **Key class signatures:**
  - `LlamaRMSNorm` — `hidden_states.to(float32) * rsqrt(variance + eps); return weight * hidden_states.to(input_dtype)`. Weight initialized to ones. The `weight` multiply is applied AFTER cast back to input dtype.
  - `Qwen3RMSNorm` — structurally identical to `LlamaRMSNorm`. The `variance_epsilon` attribute holds eps (`Gemma2RMSNorm` uses `eps` directly).
  - `Gemma2RMSNorm` — **the key divergence**: `output = x * rsqrt(x.pow(2).mean(-1) + eps); return output * (1.0 + self.weight.float())`. The `+1` lives on weight (which initializes to zero — so the layer starts as identity), and the `(1 + weight)` multiply happens in fp32 BEFORE the cast back. This is what Forge v2's `casting_mode="gemma"` mirrors.
- **What to read first:** `LlamaRMSNorm.forward` (the no-offset baseline), then `Gemma2RMSNorm.forward` (the offset path) — confirm the cast-back-to-input-dtype-after-affine vs cast-back-after-rstd-before-affine distinction.
- **How Forge uses these:** `kernels/rmsnorm/forge_rmsnorm_v1.py:torch_rmsnorm_reference` is the in-repo oracle; v2 will extend it to accept an `offset` arg matching the Gemma form. Tests compare Forge output against this expanded oracle.

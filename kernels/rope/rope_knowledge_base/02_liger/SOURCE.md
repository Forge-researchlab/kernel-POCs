# Liger-Kernel — RoPE Kernel (Q and K applied separately)

- **Upstream URL:**
  - https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/rope.py
  - https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/transformers/rope.py
- **Repo:** linkedin/Liger-Kernel
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** BSD-2-Clause
- **What this is:** A Triton implementation of HuggingFace-style Llama/Mistral RoPE that applies the rotation to Q and K within a single kernel launch using paired tile loads/stores per token, but performs the rotation on Q and K as separate tile operations inside the kernel (unlike Unsloth's fully fused version which fuses Q and K rotations more aggressively into one fused launch path).
- **What to read first:** Start with the `_triton_rope` Triton JIT kernel and the `LigerRopeFunction` `torch.autograd.Function` class (with `rope_forward` / `rope_backward` host wrappers); then read `liger_rotary_pos_emb` in `rope_module.py`.

"""Forge patching — monkey-patch HF Causal LMs to use Forge kernels.

Two patch modalities, both reversible via unpatch():

1. Per-module-instance forward replacement (most kernels).
   Walks model.named_modules(), looks up each module's class name in the
   architecture mapping (QWEN3_MAPPING / GEMMA_MAPPING), and rebinds
   `module.forward` to a closure that calls the matching Forge kernel.
   This is the pattern locked in design_details.html.

2. Module-level function replacement (RoPE only, for now).
   `apply_rotary_pos_emb` is a module-level function inside
   `transformers.models.qwen2.modeling_qwen2` / `.gemma2.modeling_gemma2`.
   It is called inline from attention.forward — there is no module instance
   whose forward we can monkey-patch. So we swap the function at module level
   and remember the original for unpatch.
"""
from .core import patch, unpatch

__all__ = ["patch", "unpatch"]

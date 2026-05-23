# TorchTitan — Llama RoPE (canonical PyTorch-blessed reference)

- **Upstream URL:** https://raw.githubusercontent.com/pytorch/torchtitan/main/torchtitan/models/common/rope.py
- **Repo:** pytorch/torchtitan
- **Branch/ref:** main (HEAD as of fetch)
- **Fetch date:** 2026-05-23
- **License:** BSD-3-Clause
- **What this is:** PyTorch's official distributed-training reference repo. Their RoPE uses complex-number formulation (`precompute_freqs_cis` returns a complex tensor via `torch.polar`); cleanest "what does idiomatic PyTorch RoPE look like" baseline. Note: in current `main`, the Llama3 `model.py` no longer contains RoPE inline — it has been promoted to a shared `torchtitan/models/common/rope.py` module reused across Llama3, Llama4, DeepSeek V3, Qwen3, and GPT-OSS. The Llama3 model now references it via a `rope` config field on the `Decoder.Config`.
- **What was extracted:** `RoPE` class (`Module` subclass with nested `Config` dataclass), `_precompute_complex` (Llama3/4 complex/`torch.polar` path with optional Llama and YaRN frequency scaling), `_precompute_cos_sin` (Qwen3/GPT-OSS path with YaRN mscale), `_reshape_for_broadcast_complex`, `_reshape_for_broadcast_cos_sin`, `_rotate_half`, `_maybe_wrap_positions` (DTensor helper), `apply_rotary_emb_complex` (Llama3/4 style), `apply_rotary_emb_single_complex` (DeepSeek V3 MLA style), `apply_rotary_emb_cos_sin` (Qwen3/GPT-OSS style).
- **What to read first:** `RoPE._precompute_complex` and `apply_rotary_emb_complex` (note the complex-tensor layout via `torch.view_as_complex` / `torch.polar` — differs from HF's cos/sin split). Then compare `_precompute_cos_sin` + `apply_rotary_emb_cos_sin` for the cos/sin variant.

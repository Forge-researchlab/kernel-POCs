# Forge internal — RMSNorm derivation + design notes

**Status:** ⏳ Pending. Mirrors the placeholder slot in `kernels/rope/rope_knowledge_base/06_forge_internal/`.

This directory will hold:

- `forge_learning/module_4_kernel_math/03_gradient_derivation_rmsnorm.md` — hand-derived forward and backward formulas for RMSNorm with the Gemma offset, including the cast-back-to-input-dtype decision (Llama vs Gemma vs none).
- `forge_learning/module_5_kernel_implementation/03_rmsnorm_triton_kernel.md` — Forge's own implementation walkthrough for v2 (offset constexpr, casting modes, SM-proportional dW partials).

Until those exist in the team's `forge_learning/` curriculum, this README is the placeholder. The shipping kernel at `kernels/rmsnorm/forge_rmsnorm_v2.py` is the source of truth in the meantime; see also `kernels/rmsnorm/docs/evolution_report.md` and `docs/comparative_analysis.md`.

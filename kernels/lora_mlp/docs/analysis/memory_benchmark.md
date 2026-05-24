# LoRA MLP Peak GPU Memory Benchmark

**Date:** 2026-05-24
**GPU:** NVIDIA A100-SXM4-80GB
**Script:** `kernels/lora_mlp/benchmarks/bench_memory.py`
**Methodology:** `torch.cuda.reset_peak_memory_stats()` before each run; peak = `max_memory_allocated - memory_allocated_before`. Median of 3 runs.

## What each column means

- **Weights (MB):** persistent storage for the projection tensors. Training paths store `W + A + B` for gate/up/down (identical across Unsloth, v3, v5, v5_upgrade_1). Inference stores only the merged `W_eff` tensors — no separate A/B.
- **Fwd (MB):** peak `memory_allocated` delta during the forward call. Captures temporary buffers (e.g. v5's packed `W_mega`, the `[M, 2*I + 2*r]` mega-matmul output).
- **Fwd+Bwd (MB):** peak delta across forward + backward. Includes the temporaries above plus everything backward needs simultaneously (`DW`, transposed `A/B`, all six grad buffers, `dX`, and so on).
- **Resident after fwd (MB):** what stays allocated right after the forward returns — the output tensor plus any `save_for_backward` tensors. This is the activation footprint that backward has to live with.

## LLaMA-8B production (batch=4, seq=2048, r=16, bf16)

`batch=4, seq=2048, H=4096, I=14336, r=16, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 337.7 | 736.2 | 801.9 | 512.0 | 1.00x | 1.00x |
| v3 | 337.7 | 1184.8 | 1184.8 | 512.0 | 1.61x | 1.48x |
| v5 | 337.7 | 1585.4 | 1585.4 | 512.0 | 2.15x | 1.98x |
| v5_upgrade_1 | 337.7 | 1412.2 | 1412.2 | 512.0 | 1.92x | 1.76x |
| v5_inference (pre-merged) | 336.0 | 736.0 | — | 64.0 | 1.00x | — |

## LLaMA-8B small (batch=1, seq=2048, r=16, bf16)

`batch=1, seq=2048, H=4096, I=14336, r=16, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 337.7 | 184.1 | 201.8 | 128.0 | 1.00x | 1.00x |
| v3 | 337.7 | 296.2 | 296.2 | 128.0 | 1.61x | 1.47x |
| v5 | 337.7 | 648.9 | 648.9 | 128.0 | 3.53x | 3.22x |
| v5_upgrade_1 | 337.7 | 522.6 | 522.6 | 128.0 | 2.84x | 2.59x |
| v5_inference (pre-merged) | 336.0 | 184.0 | — | 16.0 | 1.00x | — |

## LLaMA-13B small (batch=1, seq=2048, r=16, bf16)

`batch=1, seq=2048, H=5120, I=17920, r=16, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 527.1 | 230.1 | 252.2 | 160.0 | 1.00x | 1.00x |
| v3 | 527.1 | 370.2 | 370.2 | 160.0 | 1.61x | 1.47x |
| v5 | 527.1 | 916.5 | 916.5 | 160.0 | 3.98x | 3.63x |
| v5_upgrade_1 | 527.1 | 722.6 | 722.6 | 160.0 | 3.14x | 2.87x |
| v5_inference (pre-merged) | 525.0 | 230.0 | — | 20.0 | 1.00x | — |

## LLaMA-8B production, rank sweep r=8 (bf16)

`batch=4, seq=2048, H=4096, I=14336, r=8, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 336.8 | 736.1 | 801.0 | 512.0 | 1.00x | 1.00x |
| v3 | 336.8 | 1184.4 | 1184.4 | 512.0 | 1.61x | 1.48x |
| v5 | 336.8 | 1584.7 | 1584.7 | 512.0 | 2.15x | 1.98x |
| v5_upgrade_1 | 336.8 | 1412.1 | 1412.1 | 512.0 | 1.92x | 1.76x |
| v5_inference (pre-merged) | 336.0 | 736.0 | — | 64.0 | 1.00x | — |

## LLaMA-8B production, rank sweep r=32 (bf16)

`batch=4, seq=2048, H=4096, I=14336, r=32, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 339.4 | 736.5 | 803.9 | 512.0 | 1.00x | 1.00x |
| v3 | 339.4 | 1185.5 | 1185.5 | 512.0 | 1.61x | 1.47x |
| v5 | 339.4 | 1587.9 | 1587.9 | 512.0 | 2.16x | 1.98x |
| v5_upgrade_1 | 339.4 | 1412.5 | 1412.5 | 512.0 | 1.92x | 1.76x |
| v5_inference (pre-merged) | 336.0 | 736.0 | — | 64.0 | 1.00x | — |

## LLaMA-8B production, rank sweep r=64 (bf16)

`batch=4, seq=2048, H=4096, I=14336, r=64, bfloat16`

| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth | 342.8 | 737.0 | 807.8 | 512.0 | 1.00x | 1.00x |
| v3 | 342.8 | 1187.0 | 1187.0 | 512.0 | 1.61x | 1.47x |
| v5 | 342.8 | 1592.0 | 1592.0 | 512.0 | 2.16x | 1.97x |
| v5_upgrade_1 | 342.8 | 1413.0 | 1413.0 | 512.0 | 1.92x | 1.75x |
| v5_inference (pre-merged) | 336.0 | 736.0 | — | 64.0 | 1.00x | — |

## Interpretation

**Winner on memory: `v5_inference` (pre-merged).** At LLaMA-8B production (M=8192, r=16) it ties Unsloth on forward peak (736 MB) but only retains 64 MB after the call versus 512 MB for every training path. That's 8x lower activation footprint, achieved by (a) merging `B @ A` into `W_eff` once at LoRA-merge time (no runtime LoRA matmuls) and (b) running outside autograd so nothing is saved for backward (no `e`, no `g`).

**Honest finding (surprise): `v5` training uses ~2.1x MORE peak forward memory than Unsloth, and ~34% more than `v3`.** The wall-clock-equivalent v5 path pays a real memory cost for the packed mega-GEMM:

- `W_mega = cat(W_gate, W_up, A_gate, A_up)` allocates a fresh `[2*I + 2*r, H]` tensor every call (~224 MiB at LLaMA-8B).
- `W_down_packed = cat(W_down, A_down)` allocates another `[H + r, I]` (~118 MiB).
- The mega-matmul output `result = X @ W_mega.t()` is `[M, 2*I + 2*r]` (~448 MiB at M=8192) and stays alive (via the non-contiguous slice views `e_base`, `g_base`, `xa_gate`, `xa_up`) until the Python scope of `_v5_forward_impl` exits.
- The Triton epilogue still allocates contiguous `e_full`, `g_full` (2 × ~224 MiB) for backward, and `h` (~224 MiB) — those don't disappear when `result` does.

Add it up: ~224 (W_mega) + ~118 (W_down_packed) + ~448 (result) + ~672 (h, e_full, g_full) + 64 (output) + 64 (contig copy for addmm_) ≈ **1590 MiB peak** — within rounding of the measured 1585.4 MB. Compare against v3, which doesn't pack: ~224·5 (e_base, g_base, h, e_full, g_full) + 64 (output) ≈ **1184 MiB** — exact match against measured 1184.8 MB.

**`v5_upgrade_1` saves ~173 MB vs `v5`** at LLaMA-8B production (1412.2 vs 1585.4). The win comes from dropping the down packing (no `W_down_packed`, no `down_result`, no `.contiguous()` copy). It still pays the gate+up mega-GEMM memory tax.

**Activation footprint (the thing that actually limits batch size during training) is identical at 512 MiB for all four training paths** at LLaMA-8B production: `e + g + output = 224 + 224 + 64` MiB. They all save the same tensors for backward; the differences are purely in transient forward-time buffers. So if you're trying to fit a bigger batch, picking v3 over v5 buys you headroom only during the forward call — peak `fwd+bwd` is what matters across the whole step, and there v3 wins by ~400 MB vs v5 at LLaMA-8B production.

**Sanity check on the math** (bf16, M=8192, H=4096, I=14336, r=16):

- `[M, I]` (e, g, h) = 8192·14336·2 bytes = **224 MiB** each.
- `[M, H]` (output, X, dX) = 8192·4096·2 = **64 MiB** each.
- Weights `W_gate + W_up + W_down` = 3·14336·4096·2 / 2²⁰ = **336 MiB**; LoRA `A`s and `B`s add ~1.7 MiB. Measured weight memory: **337.7 MiB** ✓.
- Resident after fwd for training = output + saved e + saved g = 64 + 224 + 224 = **512 MiB** ✓ (matches the measurement exactly for all four training paths).

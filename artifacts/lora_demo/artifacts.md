# Forge LoRA fine-tune — artifacts

**Model:** `google/gemma-2-2b`
**Setup:** FSDP2 world=2 · NVIDIA A100-SXM4-80GB · torch 2.6.0+cu124
**Kernels patched by `forge.patch`:** `['lora_qkv', 'lora_mlp']` → counts `{'lora_qkv': 26, 'lora_mlp': 26}`
**LoRA:** r=16, α=32, targets q/k/v/gate/up/down
**Training:** 200 steps, lr=0.0002, effective batch 8 (= 2 × 4 micro-batches), max_seq=96

## Numbers

| Metric | Value |
|---|---|
| First-step loss | 3.6833 |
| Final-step loss | 0.0001 |
| Minimum loss | 0.0001 |
| Loss reduction | 100.0% |
| Final peak VRAM per rank | 6.67 GB, 6.67 GB |

## Charts

| | |
|---|---|
| ![loss](loss_curve.png) | ![vram](vram.png) |
| ![step time](step_time.png) | (one-pager: `summary.png`) |

See `summary.png` for a single-figure dashboard combining all of the above.
See `inference_samples.md` for held-out generations before vs after fine-tune.

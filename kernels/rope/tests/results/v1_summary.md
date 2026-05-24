# ForgeRoPE V1 — Test Results Summary

**Run:** 2026-05-23T15:09:09.048635+00:00
**Device:** NVIDIA A100-SXM4-80GB (compute [8, 0])
**Torch / Triton:** 2.4.1+cu124 / 3.0.0

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness | 36 | 36 |
| Backward correctness | 6 | 6 |
| Gradcheck (fp64) | ✓ | 1 |

## Forward timing (median, lower = better)

| Shape | dtype | PyTorch (ms) | Liger (ms) | Unsloth-default (ms) | Unsloth-fused-QK (ms) | **Forge V1 (ms)** | Forge speedup vs PT |
|---|---|---|---|---|---|---|---|
| qwen25_0p5b_short (1×14/2×512×64) | torch.bfloat16 | 0.0696 | 0.0252 | 0.1224 | 0.0418 | **0.0173** | 4.03× |
| qwen25_0p5b_short (1×14/2×512×64) | torch.float16 | 0.0583 | 0.0256 | 0.1168 | 0.0482 | **0.0172** | 3.38× |
| qwen3_8b_short (4×32/8×512×128) | torch.bfloat16 | 0.2554 | 0.1123 | 0.1295 | 0.0977 | **0.0996** | 2.56× |
| qwen3_8b_short (4×32/8×512×128) | torch.float16 | 0.2542 | 0.1135 | 0.1286 | 0.0978 | **0.0993** | 2.56× |
| qwen3_8b_train (2×32/8×2048×128) | torch.bfloat16 | 0.4733 | 0.2155 | 0.2452 | 0.1833 | **0.1908** | 2.48× |
| qwen3_8b_train (2×32/8×2048×128) | torch.float16 | 0.4734 | 0.2153 | 0.2441 | 0.1809 | **0.1907** | 2.48× |

## HBM bandwidth utilization (Forge V1)

| Shape | dtype | Total traffic (MB) | Forge V1 time (ms) | Achieved bandwidth (GB/s) |
|---|---|---|---|---|
| qwen25_0p5b_short | torch.bfloat16 | 2.2 | 0.0173 | 129 |
| qwen25_0p5b_short | torch.float16 | 2.2 | 0.0172 | 129 |
| qwen3_8b_short | torch.bfloat16 | 42.2 | 0.0996 | 424 |
| qwen3_8b_short | torch.float16 | 42.2 | 0.0993 | 425 |
| qwen3_8b_train | torch.bfloat16 | 84.9 | 0.1908 | 445 |
| qwen3_8b_train | torch.float16 | 84.9 | 0.1907 | 445 |
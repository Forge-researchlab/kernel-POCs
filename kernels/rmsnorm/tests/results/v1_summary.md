# ForgeRMSNorm V1 — Test Results Summary

**Run:** 2026-05-24T14:50:48.443442+00:00
**Device:** NVIDIA A100-SXM4-80GB
**Torch / Triton:** 2.4.1+cu124 / 3.0.0

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness  | 12 | 12 |
| Backward correctness | 4 | 4 |
| fp64 gradcheck       | ✗ | 1 |

> **Note:** v1 fails fp64 gradcheck by design — internal fp32 accumulation loses fp64 perturbations. v2 fixes this via the `ACC_DTYPE` constexpr.

## Forward + backward timing

| Shape | dtype | PT (ms) | **V1 (ms)** | speedup vs PT |
|---|---|---|---|---|
| tiny (1×8×64) | torch.bfloat16 | 0.041 | **0.008** | 4.96× |
| tiny (1×8×64) | torch.float16 | 0.036 | **0.007** | 4.97× |
| qwen25_0p5b (2×512×896) | torch.bfloat16 | 0.054 | **0.011** | 5.00× |
| qwen25_0p5b (2×512×896) | torch.float16 | 0.053 | **0.010** | 5.21× |
| qwen3_8b_short (4×512×4096) | torch.bfloat16 | 0.265 | **0.030** | 8.77× |
| qwen3_8b_short (4×512×4096) | torch.float16 | 0.260 | **0.029** | 8.88× |
| qwen3_8b_train (2×2048×4096) | torch.bfloat16 | 0.484 | **0.049** | 9.81× |
| qwen3_8b_train (2×2048×4096) | torch.float16 | 0.484 | **0.049** | 9.79× |

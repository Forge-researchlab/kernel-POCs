# ForgeRMSNorm V1 — Test Results Summary

**Run:** 2026-05-24T11:47:58.942921+00:00
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
| tiny (1×8×64) | torch.bfloat16 | 0.043 | **0.008** | 5.17× |
| tiny (1×8×64) | torch.float16 | 0.034 | **0.007** | 4.76× |
| qwen25_0p5b (2×512×896) | torch.bfloat16 | 0.054 | **0.010** | 5.29× |
| qwen25_0p5b (2×512×896) | torch.float16 | 0.053 | **0.010** | 5.19× |
| qwen3_8b_short (4×512×4096) | torch.bfloat16 | 0.263 | **0.030** | 8.87× |
| qwen3_8b_short (4×512×4096) | torch.float16 | 0.262 | **0.030** | 8.70× |
| qwen3_8b_train (2×2048×4096) | torch.bfloat16 | 0.483 | **0.049** | 9.78× |
| qwen3_8b_train (2×2048×4096) | torch.float16 | 0.480 | **0.049** | 9.71× |

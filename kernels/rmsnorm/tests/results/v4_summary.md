# ForgeRMSNorm V4 — Test Results Summary

**Run:** 2026-05-24T12:58:27.776092+00:00
**Device:** NVIDIA A100-SXM4-80GB (SMs=108)
**Torch / Triton:** 2.4.1+cu124 / 3.0.0

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness            | 84 | 84 |
| in_place=True ↔ False bit-identical | 24 | 24 |
| Backward correctness (out-of-place) | 12 | 12 |
| fp64 gradcheck (llama, out-of-place) | ✓ | 1 |
| fp64 gradcheck (gemma, out-of-place) | ✓ | 1 |

> in_place=True is non-reentrant by design (each backward modifies dy in-place); correctness is verified via test_in_place_equivalence instead.

## Forward + backward timing (median ms; smaller = better)

| Shape | dtype | offset | PT | Liger | Unsloth(no-dW) | V1 | V3 | V4(ip) | V4(op) | V4(ip) vs Liger | V4(ip) vs V3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny(1×8×64) | torch.bfloat16 | 0.0 | 0.485 | 0.486 | 0.310 | 0.382 | 0.505 | **0.502** | 0.500 | 0.97× | 1.00× |
| tiny(1×8×64) | torch.bfloat16 | 1.0 | 0.557 | 0.493 | 0.219 | — | 0.397 | **0.327** | 0.360 | 1.51× | 1.21× |
| tiny(1×8×64) | torch.float16 | 0.0 | 0.758 | 0.407 | 0.273 | 0.245 | 0.414 | **0.425** | 0.496 | 0.96× | 0.98× |
| tiny(1×8×64) | torch.float16 | 1.0 | 0.581 | 0.582 | 0.255 | — | 0.575 | **0.513** | 0.526 | 1.13× | 1.12× |
| qwen25_0p5b(2×512×896) | torch.bfloat16 | 0.0 | 0.590 | 0.307 | 0.256 | 0.374 | 0.308 | **0.334** | 0.370 | 0.92× | 0.92× |
| qwen25_0p5b(2×512×896) | torch.bfloat16 | 1.0 | 0.663 | 0.369 | 0.227 | — | 0.408 | **0.616** | 0.436 | 0.60× | 0.66× |
| qwen25_0p5b(2×512×896) | torch.float16 | 0.0 | 0.522 | 0.603 | 0.330 | 0.294 | 0.421 | **0.528** | 0.498 | 1.14× | 0.80× |
| qwen25_0p5b(2×512×896) | torch.float16 | 1.0 | 0.567 | 0.462 | 0.312 | — | 0.455 | **0.497** | 0.523 | 0.93× | 0.91× |
| qwen3_8b_short(4×512×4096) | torch.bfloat16 | 0.0 | 0.900 | 0.302 | 0.216 | 0.355 | 0.398 | **0.386** | 0.378 | 0.78× | 1.03× |
| qwen3_8b_short(4×512×4096) | torch.bfloat16 | 1.0 | 0.957 | 0.507 | 0.294 | — | 0.366 | **0.413** | 0.431 | 1.23× | 0.89× |
| qwen3_8b_short(4×512×4096) | torch.float16 | 0.0 | 0.897 | 0.374 | 0.270 | 0.325 | 0.398 | **0.492** | 0.524 | 0.76× | 0.81× |
| qwen3_8b_short(4×512×4096) | torch.float16 | 1.0 | 0.954 | 0.444 | 0.312 | — | 0.392 | **0.338** | 0.547 | 1.32× | 1.16× |
| qwen3_8b_train(2×2048×4096) | torch.bfloat16 | 0.0 | 1.673 | 0.362 | 0.295 | 0.310 | 0.506 | **0.388** | 0.417 | 0.93× | 1.30× |
| qwen3_8b_train(2×2048×4096) | torch.bfloat16 | 1.0 | 1.758 | 0.406 | 0.413 | — | 0.386 | **0.424** | 0.560 | 0.96× | 0.91× |
| qwen3_8b_train(2×2048×4096) | torch.float16 | 0.0 | 1.652 | 0.395 | 0.270 | 0.488 | 0.560 | **0.490** | 0.409 | 0.81× | 1.14× |
| qwen3_8b_train(2×2048×4096) | torch.float16 | 1.0 | 1.762 | 0.400 | 0.319 | — | 0.429 | **0.338** | 0.432 | 1.19× | 1.27× |
| gemma2_2b(2×2048×2304) | torch.bfloat16 | 0.0 | 0.994 | 0.333 | 0.234 | 0.236 | 0.428 | **0.382** | 0.405 | 0.87× | 1.12× |
| gemma2_2b(2×2048×2304) | torch.bfloat16 | 1.0 | 1.060 | 0.372 | 0.277 | — | 0.306 | **0.438** | 0.514 | 0.85× | 0.70× |
| gemma2_2b(2×2048×2304) | torch.float16 | 0.0 | 0.992 | 0.526 | 0.309 | 0.307 | 0.399 | **0.495** | 0.466 | 1.06× | 0.80× |
| gemma2_2b(2×2048×2304) | torch.float16 | 1.0 | 1.055 | 0.300 | 0.217 | — | 0.302 | **0.459** | 0.514 | 0.65× | 0.66× |
| gemma2_9b(2×2048×3584) | torch.bfloat16 | 0.0 | 1.466 | 0.436 | 0.258 | 0.248 | 0.387 | **0.313** | 0.466 | 1.39× | 1.24× |
| gemma2_9b(2×2048×3584) | torch.bfloat16 | 1.0 | 1.559 | 0.444 | 0.361 | — | 0.423 | **0.471** | 0.413 | 0.94× | 0.90× |
| gemma2_9b(2×2048×3584) | torch.float16 | 0.0 | 1.460 | 0.361 | 0.299 | 0.279 | 0.307 | **0.460** | 0.440 | 0.78× | 0.67× |
| gemma2_9b(2×2048×3584) | torch.float16 | 1.0 | 1.557 | 0.417 | 0.264 | — | 0.608 | **0.448** | 0.385 | 0.93× | 1.36× |
| non_pow2(4×128×4097) | torch.bfloat16 | 0.0 | 0.627 | 0.405 | 0.309 | 0.321 | 0.416 | **0.387** | 0.334 | 1.05× | 1.08× |
| non_pow2(4×128×4097) | torch.bfloat16 | 1.0 | 0.506 | 0.419 | 0.342 | — | 0.507 | **0.517** | 0.420 | 0.81× | 0.98× |
| non_pow2(4×128×4097) | torch.float16 | 0.0 | 0.580 | 0.438 | 0.414 | 0.210 | 0.352 | **0.255** | 0.472 | 1.72× | 1.38× |
| non_pow2(4×128×4097) | torch.float16 | 1.0 | 0.485 | 0.486 | 0.330 | — | 0.439 | **0.403** | 0.338 | 1.21× | 1.09× |

> Note: Unsloth's backward returns `None` for dW (designed for frozen-base+LoRA training). Its timing represents that design point — not a fair perf comparison for full fine-tuning workloads where dW is computed (Liger, Forge v1-v4).
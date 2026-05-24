# ForgeRMSNorm V4 — Test Results Summary

**Run:** 2026-05-24T14:52:04.686412+00:00
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
| tiny(1×8×64) | torch.bfloat16 | 0.0 | 0.497 | 0.365 | 0.283 | 0.422 | 0.394 | **0.480** | 0.483 | 0.76× | 0.82× |
| tiny(1×8×64) | torch.bfloat16 | 1.0 | 0.662 | 0.376 | 0.281 | — | 0.382 | **0.351** | 0.364 | 1.07× | 1.09× |
| tiny(1×8×64) | torch.float16 | 0.0 | 0.375 | 0.288 | 0.224 | 0.277 | 0.294 | **0.364** | 0.338 | 0.79× | 0.81× |
| tiny(1×8×64) | torch.float16 | 1.0 | 0.469 | 0.361 | 0.181 | — | 0.365 | **0.341** | 0.380 | 1.06× | 1.07× |
| qwen25_0p5b(2×512×896) | torch.bfloat16 | 0.0 | 0.449 | 0.294 | 0.184 | 0.337 | 0.339 | **0.298** | 0.321 | 0.99× | 1.14× |
| qwen25_0p5b(2×512×896) | torch.bfloat16 | 1.0 | 0.551 | 0.356 | 0.310 | — | 0.485 | **0.420** | 0.498 | 0.85× | 1.15× |
| qwen25_0p5b(2×512×896) | torch.float16 | 0.0 | 0.432 | 0.353 | 0.284 | 0.287 | 0.444 | **0.460** | 0.464 | 0.77× | 0.97× |
| qwen25_0p5b(2×512×896) | torch.float16 | 1.0 | 0.804 | 0.825 | 0.343 | — | 0.591 | **0.591** | 0.540 | 1.40× | 1.00× |
| qwen3_8b_short(4×512×4096) | torch.bfloat16 | 0.0 | 0.903 | 0.525 | 0.245 | 0.346 | 0.421 | **0.444** | 0.373 | 1.18× | 0.95× |
| qwen3_8b_short(4×512×4096) | torch.bfloat16 | 1.0 | 0.967 | 0.431 | 0.260 | — | 0.524 | **0.369** | 0.443 | 1.17× | 1.42× |
| qwen3_8b_short(4×512×4096) | torch.float16 | 0.0 | 0.901 | 0.358 | 0.238 | 0.358 | 0.483 | **0.445** | 0.420 | 0.80× | 1.08× |
| qwen3_8b_short(4×512×4096) | torch.float16 | 1.0 | 0.955 | 0.430 | 0.313 | — | 0.427 | **0.379** | 0.419 | 1.13× | 1.13× |
| qwen3_8b_train(2×2048×4096) | torch.bfloat16 | 0.0 | 1.670 | 0.540 | 0.316 | 0.341 | 0.572 | **0.527** | 0.581 | 1.02× | 1.09× |
| qwen3_8b_train(2×2048×4096) | torch.bfloat16 | 1.0 | 1.778 | 0.322 | 0.202 | — | 0.308 | **0.326** | 0.346 | 0.99× | 0.94× |
| qwen3_8b_train(2×2048×4096) | torch.float16 | 0.0 | 1.658 | 0.341 | 0.221 | 0.520 | 0.575 | **0.486** | 0.499 | 0.70× | 1.18× |
| qwen3_8b_train(2×2048×4096) | torch.float16 | 1.0 | 1.754 | 0.404 | 0.283 | — | 0.420 | **0.470** | 0.389 | 0.86× | 0.89× |
| gemma2_2b(2×2048×2304) | torch.bfloat16 | 0.0 | 0.997 | 0.356 | 0.257 | 0.343 | 0.462 | **0.458** | 0.473 | 0.78× | 1.01× |
| gemma2_2b(2×2048×2304) | torch.bfloat16 | 1.0 | 1.059 | 0.654 | 0.415 | — | 0.409 | **0.420** | 0.445 | 1.56× | 0.97× |
| gemma2_2b(2×2048×2304) | torch.float16 | 0.0 | 0.997 | 0.614 | 0.457 | 0.331 | 0.598 | **0.470** | 0.593 | 1.31× | 1.27× |
| gemma2_2b(2×2048×2304) | torch.float16 | 1.0 | 1.063 | 0.633 | 0.436 | — | 0.587 | **0.567** | 0.614 | 1.12× | 1.04× |
| gemma2_9b(2×2048×3584) | torch.bfloat16 | 0.0 | 1.475 | 0.607 | 0.359 | 0.312 | 0.689 | **0.468** | 0.551 | 1.30× | 1.47× |
| gemma2_9b(2×2048×3584) | torch.bfloat16 | 1.0 | 1.577 | 0.494 | 0.353 | — | 0.608 | **0.595** | 0.391 | 0.83× | 1.02× |
| gemma2_9b(2×2048×3584) | torch.float16 | 0.0 | 1.469 | 0.475 | 0.329 | 0.386 | 0.621 | **0.472** | 0.429 | 1.01× | 1.32× |
| gemma2_9b(2×2048×3584) | torch.float16 | 1.0 | 1.559 | 0.401 | 0.290 | — | 0.403 | **0.431** | 0.431 | 0.93× | 0.94× |
| non_pow2(4×128×4097) | torch.bfloat16 | 0.0 | 0.581 | 0.357 | 0.267 | 0.321 | 0.332 | **0.403** | 0.396 | 0.89× | 0.82× |
| non_pow2(4×128×4097) | torch.bfloat16 | 1.0 | 0.489 | 0.406 | 0.263 | — | 0.420 | **0.424** | 0.389 | 0.96× | 0.99× |
| non_pow2(4×128×4097) | torch.float16 | 0.0 | 0.542 | 0.391 | 0.235 | 0.374 | 0.501 | **0.475** | 0.608 | 0.82× | 1.05× |
| non_pow2(4×128×4097) | torch.float16 | 1.0 | 0.856 | 0.487 | 0.362 | — | 0.588 | **0.589** | 0.822 | 0.83× | 1.00× |

> Note: Unsloth's backward returns `None` for dW (designed for frozen-base+LoRA training). Its timing represents that design point — not a fair perf comparison for full fine-tuning workloads where dW is computed (Liger, Forge v1-v4).
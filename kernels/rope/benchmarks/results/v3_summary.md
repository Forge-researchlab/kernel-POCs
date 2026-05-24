# ForgeRoPE V3 — Test + Benchmark Results

**Run:** 2026-05-23T15:30:00.430603+00:00
**Device:** NVIDIA A100-SXM4-80GB (compute [8, 0])
**Torch / Triton:** 2.4.1+cu124 / 3.0.0
**Kernel design:** V2 base + `@triton.autotune` over num_warps×num_stages, keyed on seq_len

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness | 30 | 30 |
| Backward correctness | 8 | 8 |
| Gradcheck (fp64) | PASS | 1 |

## Forward timing (median ms) — V3 vs V2 vs baselines

| Shape (G) | dtype | PyTorch | Liger | UnslQK | V1 | V2 | **V3** | V3/V2 | autotune (nw, ns) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b_short (G=4) | torch.bfloat16 | 0.2732 | 0.1136 | 0.0980 | 0.0990 | 0.0420 | **0.0386** | 1.09× | (2, 3) |
| qwen3_8b_short (G=4) | torch.float16 | 0.2541 | 0.1125 | 0.0978 | 0.0986 | 0.0409 | **0.0380** | 1.08× | (2, 2) |
| qwen3_8b_train (G=4) | torch.bfloat16 | 0.4729 | 0.2152 | 0.1835 | 0.1918 | 0.0753 | **0.0663** | 1.14× | (2, 2) |
| qwen3_8b_train (G=4) | torch.float16 | 0.4702 | 0.2141 | 0.1803 | 0.1904 | 0.0738 | **0.0655** | 1.13× | (2, 3) |
| mqa_extreme (G=8) | torch.bfloat16 | 0.0884 | 0.0332 | 0.0684 | 0.0297 | 0.0148 | **0.0152** | 0.97× | (2, 2) |
| mqa_extreme (G=8) | torch.float16 | 0.0887 | 0.0329 | 0.0652 | 0.0289 | 0.0140 | **0.0145** | 0.96× | (2, 2) |
| mha_no_gqa (G=1) | torch.bfloat16 | 0.1957 | 0.0909 | 0.0719 | 0.0683 | 0.0693 | **0.0442** | 1.57× | (2, 2) |
| mha_no_gqa (G=1) | torch.float16 | 0.1902 | 0.0907 | 0.0736 | 0.0673 | 0.0694 | **0.0438** | 1.58× | (2, 2) |

## Backward timing (median ms)

| Shape (G) | dtype | PyTorch | Liger | V1 | V2 | **V3** | V3/V2 |
|---|---|---|---|---|---|---|---|
| qwen3_8b_short (G=4) | torch.bfloat16 | 0.3328 | 0.2338 | 0.1016 | 0.1103 | **0.1497** | 0.74× |
| qwen3_8b_short (G=4) | torch.float16 | 0.2916 | 0.0884 | 0.1454 | 0.1395 | **0.1328** | 1.05× |
| qwen3_8b_train (G=4) | torch.bfloat16 | 0.5562 | 0.1674 | 0.1927 | 0.0796 | **0.0660** | 1.21× |
| qwen3_8b_train (G=4) | torch.float16 | 0.5552 | 0.1662 | 0.1919 | 0.0743 | **0.0693** | 1.07× |
| mqa_extreme (G=8) | torch.bfloat16 | 0.3572 | 0.2450 | 0.2643 | 0.1565 | **0.2657** | 0.59× |
| mqa_extreme (G=8) | torch.float16 | 0.3159 | 0.2883 | 0.2184 | 0.2128 | **0.1464** | 1.45× |
| mha_no_gqa (G=1) | torch.bfloat16 | 0.3421 | 0.0727 | 0.1819 | 0.1944 | **0.1162** | 1.67× |
| mha_no_gqa (G=1) | torch.float16 | 0.3083 | 0.1321 | 0.2080 | 0.1379 | **0.1075** | 1.28× |

## HBM bandwidth utilization (Forge V3)

| Shape | dtype | Traffic (MB) | V3 time (ms) | Achieved BW (GB/s) |
|---|---|---|---|---|
| qwen3_8b_short | torch.bfloat16 | 42.2 | 0.0386 | 1093 |
| qwen3_8b_short | torch.float16 | 42.2 | 0.0380 | 1111 |
| qwen3_8b_train | torch.bfloat16 | 84.9 | 0.0663 | 1281 |
| qwen3_8b_train | torch.float16 | 84.9 | 0.0655 | 1297 |
| mqa_extreme | torch.bfloat16 | 10.0 | 0.0152 | 657 |
| mqa_extreme | torch.float16 | 10.0 | 0.0145 | 685 |
| mha_no_gqa | torch.bfloat16 | 34.1 | 0.0442 | 771 |
| mha_no_gqa | torch.float16 | 34.1 | 0.0438 | 778 |
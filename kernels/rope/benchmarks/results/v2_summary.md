# ForgeRoPE V2 — Test + Benchmark Results

**Run:** 2026-05-23T15:20:43.840134+00:00
**Device:** NVIDIA A100-SXM4-80GB (compute [8, 0])
**Torch / Triton:** 2.4.1+cu124 / 3.0.0
**Kernel design:** grid `(b·s, n_kv)`, G=n_q//n_kv Q heads per program, 1 K head per program, cos/sin loaded once and reused

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness | 30 | 30 |
| Backward correctness | 8 | 8 |
| Gradcheck (fp64) | PASS | 1 |

## Forward timing (median ms)

| Shape (b×nq/nkv×s×hd, G) | dtype | PyTorch | Liger | UnslDef | UnslQK | **V1** | **V2** | V2/V1 | V2 vs PT |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b_short (4×32/8×512×128, G=4) | torch.bfloat16 | 0.3023 | 0.1236 | 0.1296 | 0.0986 | 0.0995 | **0.0414** | 2.40× | 7.30× |
| qwen3_8b_short (4×32/8×512×128, G=4) | torch.float16 | 0.2543 | 0.1133 | 0.1389 | 0.0976 | 0.0986 | **0.0407** | 2.42× | 6.25× |
| qwen3_8b_train (2×32/8×2048×128, G=4) | torch.bfloat16 | 0.4722 | 0.2163 | 0.2469 | 0.1836 | 0.1907 | **0.0746** | 2.56× | 6.33× |
| qwen3_8b_train (2×32/8×2048×128, G=4) | torch.float16 | 0.4703 | 0.2142 | 0.2440 | 0.1808 | 0.1900 | **0.0733** | 2.59× | 6.41× |
| mqa_extreme (2×8/1×1024×128, G=8) | torch.bfloat16 | 0.0890 | 0.0331 | 0.0916 | 0.0662 | 0.0290 | **0.0145** | 2.00× | 6.14× |
| mqa_extreme (2×8/1×1024×128, G=8) | torch.float16 | 0.0889 | 0.0337 | 0.0905 | 0.0641 | 0.0295 | **0.0146** | 2.02× | 6.08× |
| mha_no_gqa (2×16/16×1024×128, G=1) | torch.bfloat16 | 0.1937 | 0.0919 | 0.1213 | 0.0755 | 0.0682 | **0.0699** | 0.98× | 2.77× |
| mha_no_gqa (2×16/16×1024×128, G=1) | torch.float16 | 0.1911 | 0.0903 | 0.1234 | 0.0720 | 0.0674 | **0.0696** | 0.97× | 2.75× |

## Backward timing (median ms)

| Shape (G) | dtype | PyTorch | Liger | **V1** | **V2** | V2/V1 |
|---|---|---|---|---|---|---|
| qwen3_8b_short (G=4) | torch.bfloat16 | 0.3175 | 0.1037 | 0.1013 | **0.0676** | 1.50× |
| qwen3_8b_short (G=4) | torch.float16 | 0.2938 | 0.1226 | 0.1021 | **0.0731** | 1.40× |
| qwen3_8b_train (G=4) | torch.bfloat16 | 0.5575 | 0.1674 | 0.1934 | **0.0762** | 2.54× |
| qwen3_8b_train (G=4) | torch.float16 | 0.5561 | 0.1674 | 0.1918 | **0.0835** | 2.30× |
| mqa_extreme (G=8) | torch.bfloat16 | 0.2306 | 0.0808 | 0.0345 | **0.0401** | 0.86× |
| mqa_extreme (G=8) | torch.float16 | 0.2466 | 0.0541 | 0.0529 | **0.0324** | 1.63× |
| mha_no_gqa (G=1) | torch.bfloat16 | 0.2489 | 0.0901 | 0.0684 | **0.0701** | 0.97× |
| mha_no_gqa (G=1) | torch.float16 | 0.2403 | 0.0713 | 0.0677 | **0.0694** | 0.98× |

## HBM bandwidth utilization (Forge V2)

| Shape | dtype | Traffic (MB) | V2 time (ms) | Achieved BW (GB/s) |
|---|---|---|---|---|
| qwen3_8b_short | torch.bfloat16 | 42.2 | 0.0414 | 1019 |
| qwen3_8b_short | torch.float16 | 42.2 | 0.0407 | 1037 |
| qwen3_8b_train | torch.bfloat16 | 84.9 | 0.0746 | 1138 |
| qwen3_8b_train | torch.float16 | 84.9 | 0.0733 | 1158 |
| mqa_extreme | torch.bfloat16 | 10.0 | 0.0145 | 687 |
| mqa_extreme | torch.float16 | 10.0 | 0.0146 | 681 |
| mha_no_gqa | torch.bfloat16 | 34.1 | 0.0699 | 488 |
| mha_no_gqa | torch.float16 | 34.1 | 0.0696 | 490 |
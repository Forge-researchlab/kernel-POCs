# Cross Entropy Kernel

Research workspace for a memory-efficient cross entropy kernel targeting LLM vocabulary sizes.

## Structure

```
cross_entropy/
├── reference/             # Notes or copied references from Liger, PyTorch, etc.
├── experiments/v{N}/      # Versioned Forge kernel experiments
├── benchmarks/
│   ├── bench_cross_entropy.py
│   ├── bench_fused_linear_cross_entropy.py
│   └── results/
├── tests/
│   ├── test_cross_entropy.py
│   └── test_fused_linear_cross_entropy.py
└── docs/
    └── benchmarks.md
```

## Quick Start

```bash
# Correctness tests
uv run pytest kernels/cross_entropy/tests/test_cross_entropy.py -v

# Small smoke benchmark
uv run python kernels/cross_entropy/benchmarks/bench_cross_entropy.py \
  --bt 128 --vocab 32000 --dtype bf16 --providers torch forge liger

# Small fused linear + CE smoke benchmark
uv run python kernels/cross_entropy/benchmarks/bench_fused_linear_cross_entropy.py \
  --bt 128 --hidden 1024 --vocab 32000 --dtype bf16 --providers torch forge liger

# LLaMA-3-style sweep
uv run python kernels/cross_entropy/benchmarks/bench_cross_entropy.py \
  --bt 1024 2048 4096 8192 --vocab 128256 --dtype fp32 \
  --providers torch forge liger --save kernels/cross_entropy/benchmarks/results/

# If Liger is cloned but not installed, point the benchmark at it
uv run python kernels/cross_entropy/benchmarks/bench_cross_entropy.py \
  --liger-path /path/to/Liger-Kernel --providers torch forge liger
```

## Providers

- `torch`: `torch.nn.functional.cross_entropy`
- `forge`: current experiment under `experiments/v2/`
- `liger`: `liger_kernel.transformers.cross_entropy.LigerCrossEntropyLoss`

For the fused benchmark:

- `torch`: `torch.nn.functional.cross_entropy(torch.nn.functional.linear(input, weight, bias), target)`
- `forge`: `experiments/v2/forge_fused_linear_cross_entropy`
- `liger`: `liger_kernel.transformers.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyLoss`

`experiments/v1/` is the first core Triton path. `experiments/v2/` is the current benchmark target and mirrors the full Liger cross entropy surface: class weights, z-loss, softcap, token accuracy, predicted tokens, label smoothing, ignore index, and all reductions. It also contains `fused_linear_cross_entropy.py`, which reuses the CE v2 Triton kernel to compute chunked linear+CE without materializing full `B*T*V` logits.

The benchmark prints one summary section per `BT` value with raw Forge, Torch, and Liger numbers. The comparison columns use Forge as the target: `Torch/Forge` and `Liger/Forge`. Values above `1.00x` mean Forge is better; values below `1.00x` mean Forge is worse. CSV output is optional via `--save`.

## Done Gates

- [ ] Forward matches PyTorch reference for fp32 and bf16
- [ ] Backward gradient matches PyTorch
- [ ] Handles weights, z-loss, softcap, token accuracy, predicted tokens, `ignore_index`, label smoothing, and all reductions
- [ ] Benchmarked against PyTorch and Liger at LLaMA-3 vocab scale
- [ ] Peak activation memory is lower than PyTorch
- [ ] Measurable speedup over PyTorch for long-token regimes

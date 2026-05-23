# Cross Entropy Benchmark Notes

## Baselines

- PyTorch: `torch.nn.functional.cross_entropy`
- Liger: `LigerCrossEntropyLoss`
- Forge: `experiments/v2`, which mirrors Liger's full cross entropy feature surface

## Standard Sweep

```bash
uv run python kernels/cross_entropy/benchmarks/bench_cross_entropy.py \
  --bt 1024 2048 4096 8192 \
  --vocab 128256 \
  --dtype fp32 \
  --providers torch forge liger \
  --save kernels/cross_entropy/benchmarks/results/
```

## Smoke Sweep

```bash
uv run python kernels/cross_entropy/benchmarks/bench_cross_entropy.py \
  --bt 128 \
  --vocab 32000 \
  --dtype bf16 \
  --providers torch forge liger
```

## Metrics

- `latency_ms`: CUDA-event median latency for the selected mode
- `memory_mb`: CUDA peak allocated memory during the selected mode
- `latency_vs_forge`: provider median latency divided by Forge median latency
- `memory_vs_forge`: provider peak memory divided by Forge peak memory

The terminal output keeps the raw Forge, Torch, and Liger values visible, split by `BT`, and shows `Torch/Forge` plus `Liger/Forge` comparison columns.

The benchmark excludes input tensor creation from the timed region. Peak memory also excludes the base input tensor allocation.

For training claims, treat `full` as the headline mode. `forward` and `backward` are diagnostic phase timings because Forge and Liger intentionally materialize the logits gradient during forward.

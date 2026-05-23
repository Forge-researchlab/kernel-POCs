# LoRA MLP Kernel — Benchmark Log

## Methodology

### Baseline
- **PyTorch unfused**: separate `nn.Linear` calls with manual LoRA application
- **Framework**: `triton.testing.do_bench` for GPU timing (median of 5 runs, 3 warmup)

### Configurations

Standard sweep dimensions:

| Parameter | Values |
|-----------|--------|
| Hidden dim | 4096, 5120 |
| Intermediate dim | 14336, 17920 |
| LoRA rank | 8, 16, 32, 64 |
| Sequence length | 512, 1024, 2048, 4096 |
| Batch size | 1, 2, 4 |
| Dtype | bf16, fp32 |

### Primary benchmark config (LLaMA-3 8B scale)
```
hidden=4096, intermediate=14336, rank=16, seq=2048, batch=4, dtype=bf16
```

### Metrics
- **Latency** (ms): wall-clock kernel time
- **Throughput** (TFLOPS): effective compute throughput
- **Memory** (MB): peak GPU memory allocated
- **Speedup**: latency ratio vs PyTorch baseline

---

## Results

### Latest: [Unreleased]

_No benchmark results yet._

<!--
### v1 — YYYY-MM-DD

**Config**: hidden=4096, intermediate=14336, rank=16, seq=2048, batch=4, dtype=bf16

| Kernel | Forward (ms) | Backward (ms) | Memory (MB) | Speedup |
|--------|-------------|---------------|-------------|---------|
| PyTorch baseline | X.XX | X.XX | XXX | 1.00x |
| v1 fused | X.XX | X.XX | XXX | X.XXx |

**Analysis**: (what went well, what's the bottleneck)

**Rank scaling** (hidden=4096, intermediate=14336, seq=2048, batch=4):

| Rank | Baseline (ms) | Fused (ms) | Speedup |
|------|--------------|-----------|---------|
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | | | |

**Sequence scaling** (hidden=4096, intermediate=14336, rank=16, batch=4):

| Seq Len | Baseline (ms) | Fused (ms) | Speedup |
|---------|--------------|-----------|---------|
| 512 | | | |
| 1024 | | | |
| 2048 | | | |
| 4096 | | | |

CSV: `benchmarks/results/v1_YYYYMMDD.csv`
-->

---

## How to Run

```bash
cd kernels/lora_mlp

# Full sweep
python benchmarks/bench_lora_mlp.py --save benchmarks/results/

# Single config
python benchmarks/bench_lora_mlp.py \
  --hidden 4096 --intermediate 14336 --rank 16 --seq 2048 --batch 4

# Compare two versions
python benchmarks/bench_lora_mlp.py \
  --kernels v1 v2 --save benchmarks/results/comparison_v1_v2.csv
```

## Historical Summary

_Updated after each version. Quick reference for how the kernel has evolved._

| Version | Date | Forward Speedup | Backward Speedup | Key Change |
|---------|------|----------------|------------------|------------|
| — | — | — | — | _No versions yet_ |

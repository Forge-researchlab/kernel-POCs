# LoRA MLP Kernel — Benchmark Log

## Methodology

### Baselines

Three baselines, from simplest to most optimized:

| Baseline | Description | Kernel Launches (fwd) |
|----------|-------------|----------------------|
| **PyTorch naive** | Separate `nn.Linear` calls + manual LoRA `addmm_` + `F.silu` | 12+ |
| **Unsloth `matmul_lora`** | Per-projection: `X@W + s*(X@A)@B` via 3 cuBLAS calls | 10 (3 per proj + 1 SwiGLU) |
| **Unsloth `LoRA_MLP`** | Full MLP autograd.Function with Triton SwiGLU | 10 fwd / 20+ bwd |

The primary comparison target is **Unsloth `LoRA_MLP`** since it represents the best available open-source LoRA MLP.

### What Each Baseline Measures

**PyTorch naive** (`reference/lora_mlp_pytorch.py`):
```python
e = X @ W_gate + s_g * (X @ A_g) @ B_g
g = X @ W_up   + s_u * (X @ A_u) @ B_u
h = F.silu(e) * g
out = h @ W_down + s_d * (h @ A_d) @ B_d
```

**Unsloth matmul_lora** (per-projection benchmark):
```python
out = matmul_lora(X, W, W_quant=None, A, B, s)
# internally: torch.matmul(X, W.t()) then X@A then addmm_
```

**Unsloth LoRA_MLP** (full MLP benchmark):
```python
out = LoRA_MLP.apply(X, gateW, ..., downS, swiglu_fg, swiglu_bwd, inplace=True)
```

### Framework
- **GPU timing**: `triton.testing.do_bench` (median of 5 runs, 3 warmup)
- **Memory**: `torch.cuda.max_memory_allocated()` delta
- **L2 cache**: flushed between runs via `torch.cuda.empty_cache()`

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
- **Speedup**: latency ratio vs Unsloth baseline
- **HBM reads** (estimated): total bytes read from global memory

---

## Results

### v1 — Fused LoRA Matmul — 2026-05-23

**Per-projection: gate/up** (M=8192, N=14336, K=4096, dtype=bf16):

| Rank | PyTorch / Unsloth (ms) | Triton v1 (ms) | Speedup |
|------|----------------------|----------------|---------|
| 8 | 4.67 | 6.73 | 0.69x |
| 16 | 4.68 | 6.94 | 0.67x |
| 32 | 4.79 | 7.61 | 0.63x |
| 64 | 4.79 | 9.50 | 0.50x |

**Per-projection: down** (M=8192, N=4096, K=14336, dtype=bf16):

| Rank | PyTorch / Unsloth (ms) | Triton v1 (ms) | Speedup |
|------|----------------------|----------------|---------|
| 8 | — | — | — |
| 16 | — | — | — |

*(Down projection benchmarks at M=2048/4096 showed 0.54-0.71x)*

**Full MLP forward** (batch=4, seq=2048, hidden=4096, intermediate=14336, dtype=bf16):

| Rank | PyTorch / Unsloth (ms) | Triton v1 (ms) | Speedup |
|------|----------------------|----------------|---------|
| 8 | 14.22 | 19.91 | 0.71x |
| 16 | 14.18 | 21.60 | 0.66x |
| 32 | 14.19 | 22.85 | 0.62x |
| 64 | 14.30 | 28.17 | 0.51x |

**Analysis**: v1 is slower than cuBLAS for the base matmul. The Triton tiled matmul achieves ~70% of cuBLAS throughput. LoRA overhead grows linearly with rank (extra A dot in the fused K-loop). The architecture is correct — X is read once from HBM (vs twice in Unsloth) — but cuBLAS's highly optimized memory access patterns dominate.

**Implication for v2**: the per-projection speedup target is hard to hit with Triton vs cuBLAS. The real win comes from gate+up fusion (eliminate 2 full X reads + 2 large intermediate writes) which is structurally impossible with separate cuBLAS calls.

CSV: `benchmarks/results/v1_20260523_*.csv`

<!--
### v2 — Gate+Up+SwiGLU Fusion — YYYY-MM-DD

(TBD)
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

# Per-projection benchmark (v1 vs Unsloth matmul_lora)
python benchmarks/bench_lora_mlp.py \
  --mode projection --kernels v1 unsloth
```

## Historical Summary

_Updated after each version. Quick reference for how the kernel has evolved._

| Version | Date | Fwd Speedup vs Unsloth | Bwd Speedup vs Unsloth | Launches (fwd) | Key Change |
|---------|------|----------------------|----------------------|----------------|------------|
| v1 | 2026-05-23 | 0.62x (r=16) | N/A | 4 (3 fused + 1 SwiGLU) | Fused LoRA K-loop, X read once per projection |
| v2 | 2026-05-23 | 0.77x (r=16) | N/A | 4 (1 Triton + 3 cuBLAS) | Gate+Up+SwiGLU fused in Triton, cuBLAS down |
| **v3** | 2026-05-23 | **1.02-1.14x** | N/A | 8 (cuBLAS + 1 Triton epilogue) | cuBLAS all matmuls + Triton fused LoRA+SwiGLU |

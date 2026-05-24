# LoRA QKV Kernel — Benchmark Log

## Methodology

### Baselines

Three baselines, from simplest to most optimized:

| Baseline | Description | Kernel Launches (fwd) |
|----------|-------------|----------------------|
| **PyTorch naive** | Separate `nn.Linear` calls + manual LoRA `addmm_` for Q, K, V | 9+ |
| **Unsloth `matmul_lora` × 3** | Per-projection: `X@W + s*(X@A)@B` via 3 cuBLAS calls each | 9 (3 per projection × 3 projections) |
| **Packed QKV (no LoRA)** | Single `X @ W_qkv^T` then split — represents cuBLAS lower bound | 1 |

The primary comparison target is **Unsloth `matmul_lora` × 3** since it represents the best available open-source LoRA QKV approach.

### What Each Baseline Measures

**PyTorch naive** (`reference/lora_qkv_pytorch.py`):
```python
Q = X @ W_q.t() + s_q * (X @ A_q.t()) @ B_q.t()
K = X @ W_k.t() + s_k * (X @ A_k.t()) @ B_k.t()
V = X @ W_v.t() + s_v * (X @ A_v.t()) @ B_v.t()
```

**Unsloth matmul_lora** (per-projection benchmark):
```python
Q = matmul_lora(X, W_q, W_q_quant=None, A_q, B_q, s_q)
K = matmul_lora(X, W_k, W_k_quant=None, A_k, B_k, s_k)
V = matmul_lora(X, W_v, W_v_quant=None, A_v, B_v, s_v)
```

### Framework
- **GPU timing**: `triton.testing.do_bench` (median of 50 runs, 10 warmup)
- **Memory**: `torch.cuda.max_memory_allocated()` delta
- **L2 cache**: flushed between runs via `torch.cuda.empty_cache()`

### Configurations

Standard sweep dimensions:

| Parameter | Values |
|-----------|--------|
| Hidden dim | 4096, 8192 |
| Num heads | 32, 64 |
| Head dim | 128 |
| Num KV heads (GQA) | 8, 32 |
| LoRA rank | 8, 16, 32, 64 |
| Sequence length | 512, 1024, 2048, 4096 |
| Batch size | 1, 2, 4, 8 |
| Dtype | bf16, fp32 |

### Primary benchmark config (LLaMA-3 8B scale)
```
hidden=4096, num_heads=32, head_dim=128, num_kv_heads=32,
rank=16, seq=2048, batch=4, dtype=bf16
→ M = batch*seq = 8192
→ N_q = num_heads*head_dim = 4096
→ N_kv = num_kv_heads*head_dim = 4096 (MHA) or 1024 (GQA)
→ K = hidden = 4096
```

### Metrics
- **Latency** (ms): wall-clock kernel time
- **Throughput** (TFLOPS): effective compute throughput
- **Memory** (MB): peak GPU memory allocated
- **Speedup**: latency ratio vs Unsloth baseline
- **HBM reads** (estimated): total bytes read from global memory

---

## Results

<!-- Results will be added as kernel versions are developed -->

<!--
### v1 — Per-Projection Fused LoRA — YYYY-MM-DD

**QKV forward** (batch=4, seq=2048, hidden=4096, rank=16, dtype=bf16):

| Rank | PyTorch/Unsloth (ms) | Triton v1 (ms) | Speedup |
|------|---------------------|----------------|---------|
| 8    | —                   | —              | —       |
| 16   | —                   | —              | —       |
| 32   | —                   | —              | —       |
| 64   | —                   | —              | —       |

CSV: `benchmarks/results/v1_YYYYMMDD_*.csv`
-->

<!--
### v2 — Q+K+V Projection Fusion — YYYY-MM-DD

(TBD)
-->

---

## How to Run

```bash
cd kernels/lora_qkv

# Full sweep
python benchmarks/bench_lora_qkv.py --save benchmarks/results/

# Single config (LLaMA-3 8B)
python benchmarks/bench_lora_qkv.py \
  --hidden 4096 --num-heads 32 --head-dim 128 --rank 16 \
  --seq 2048 --batch 4

# Per-projection benchmark (v1 vs baseline)
python benchmarks/bench_lora_qkv.py --mode projection

# Compare two versions
python benchmarks/bench_lora_qkv.py \
  --kernels v1 v2 --save benchmarks/results/comparison_v1_v2.csv
```

## Historical Summary

_Updated after each version. Quick reference for how the kernel has evolved._

| Version | Date | Fwd Speedup vs Unsloth | Bwd Speedup | Launches (fwd) | Key Change |
|---------|------|----------------------|-------------|----------------|------------|
| — | — | — | — | — | No versions yet |

<!-- Template for future entries:
| v1 | YYYY-MM-DD | X.XXx (r=16) | N/A | N | Fused LoRA per-projection |
| v2 | YYYY-MM-DD | X.XXx (r=16) | N/A | N | Q+K+V fusion |
-->

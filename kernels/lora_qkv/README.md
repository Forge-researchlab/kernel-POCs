# LoRA QKV Fused Kernel

Research project to build a high-performance Triton kernel that fuses the LoRA forward pass for multi-head attention QKV projections — queries, keys, and values with low-rank adapters applied inline, eliminating redundant input reads and intermediate materializations.

## The Problem

Standard LoRA QKV forward requires 9 separate matmuls and reads the input `X` from HBM 6 times:

```
Q = x @ W_q^T + s_q * (x @ A_q^T) @ B_q^T   # 3 cuBLAS calls, X read 2x
K = x @ W_k^T + s_k * (x @ A_k^T) @ B_k^T   # 3 cuBLAS calls, X read 2x
V = x @ W_v^T + s_v * (x @ A_v^T) @ B_v^T   # 3 cuBLAS calls, X read 2x
```

Each LoRA term `(x @ A^T) @ B^T` is a rank bottleneck that fits entirely in SRAM for typical ranks (8–64). Fusing the base matmul with the LoRA delta, and fusing across Q/K/V projections, avoids writing/reading intermediates from HBM and reduces `X` reads from 6x to 1x.

### GQA (Grouped-Query Attention)

In LLaMA-3, K and V use fewer heads than Q (grouped-query attention):
- Q output: `[B*S, num_heads × head_dim]` = `[B*S, 4096]` for LLaMA-3 8B
- K output: `[B*S, num_kv_heads × head_dim]` = `[B*S, 1024]` (8 KV heads)
- V output: `[B*S, num_kv_heads × head_dim]` = `[B*S, 1024]`

This asymmetry means K/V matmuls are 4× smaller than Q, affecting compute/memory balance and tiling strategy.

## Structure

```
lora_qkv/
├── experiments/v{N}/      # Versioned kernel experiments
│   └── *_v{N}.py          #   e.g. lora_qkv_kernel_v1.py
│   └── *_v{N}_{M}.py      #   e.g. lora_qkv_kernel_v1_2.py (minor upgrade)
├── benchmarks/
│   ├── bench_lora_qkv.py  # Benchmark harness
│   └── results/           # CSV outputs (gitignored content)
├── tests/
│   └── test_lora_qkv.py   # Correctness + gradcheck tests
├── reference/              # Read-only reference implementations
│   └── lora_qkv_pytorch.py
├── docs/
│   ├── research.md        # Research context, related work, exploration axes
│   ├── benchmarks.md      # Benchmark methodology, current results
│   ├── analysis/          # Dated analysis reports
│   └── artifacts/         # Baseline code analysis
│       └── ANALYSIS.md
├── CHANGELOG.md            # Version history + improvement log
└── README.md               # This file
```

## Versioning Scheme

| Concept | Convention | Example |
|---------|-----------|---------|
| **Version** | Fundamentally different algorithmic approach | `v1` = per-projection fused LoRA, `v2` = Q+K+V fusion |
| **Minor upgrade** | Tuning / bugfix within same approach | `v1_2` = autotuned tile sizes for v1 |
| **File** | `lora_qkv_kernel_v{N}.py` or `_v{N}_{M}.py` | `experiments/v1/lora_qkv_kernel_v1.py` |

**Rule**: never modify a previous version — copy forward and iterate. Every file is a snapshot.

## Quick Start

```bash
# Correctness tests
pytest tests/test_lora_qkv.py -v

# Benchmark sweep
python benchmarks/bench_lora_qkv.py --save benchmarks/results/

# Single-config benchmark (LLaMA-3 8B scale)
python benchmarks/bench_lora_qkv.py \
  --hidden 4096 --num-heads 32 --head-dim 128 --num-kv-heads 8 \
  --rank 16 --seq 2048 --batch 4

# GQA benchmark
python benchmarks/bench_lora_qkv.py \
  --hidden 4096 --num-heads 32 --num-kv-heads 8 --head-dim 128 --rank 16
```

## Done Gates

- [ ] Forward matches PyTorch reference (rtol=1e-3, atol=1e-3 for bf16)
- [ ] Backward passes `torch.autograd.gradcheck` (fp64, eps=1e-6)
- [ ] Wrapped in `torch.autograd.Function` with correct `ctx.save_for_backward`
- [ ] Handles variable sequence lengths, batch sizes, and LoRA ranks
- [ ] GQA support (num_kv_heads != num_heads)
- [ ] bf16 and fp32 dtypes supported
- [ ] Memory usage <= PyTorch baseline (no extra intermediate buffers)
- [ ] Benchmarked at LLaMA-3 scale (4096 hidden, 32 heads, 128 head_dim, rank 16)
- [ ] Measurable speedup over unfused baseline

## Key Parameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| Hidden dim (H) | 4096 (8B), 8192 (70B) | Model hidden size = num_heads × head_dim |
| Num heads | 32 (8B), 64 (70B) | Number of query attention heads |
| Head dim | 128 | Per-head dimension (H / num_heads) |
| Num KV heads | 8 (GQA), 32 (MHA) | K/V head count; < num_heads for GQA |
| Q output dim | 4096 | num_heads × head_dim |
| K/V output dim | 1024 (GQA), 4096 (MHA) | num_kv_heads × head_dim |
| LoRA rank | 8, 16, 32, 64 | Low-rank bottleneck |
| Sequence length | 512 – 4096 | Tokens per sample |
| Batch size | 1 – 8 | Micro-batch during training |

## Key Differences from LoRA MLP

| Aspect | LoRA MLP | LoRA QKV |
|--------|----------|----------|
| Projections | 3 (gate, up, down) | 3 (Q, K, V) |
| Non-linearity | SwiGLU between gate/up and down | None — projections are independent |
| Output coupling | gate × up elementwise | Q, K, V are independent outputs |
| GQA | N/A | K, V may have different dimensions |
| Fusion opportunity | Activation barrier between gate/up and down | Can fuse all 3 matmuls (no barrier) |

# LoRA MLP Fused Kernel

Research project to build a high-performance Triton kernel that fuses the LoRA forward pass for MLP layers — gate, up, and down projections with low-rank adapters applied inline, eliminating intermediate materializations.

## The Problem

Standard LoRA MLP forward requires 6+ separate matmuls and intermediate buffers:

```
gate_out = x @ W_gate + (x @ A_gate) @ B_gate
up_out   = x @ W_up   + (x @ A_up)   @ B_up
hidden   = SwiGLU(gate_out, up_out)
out      = hidden @ W_down + (hidden @ A_down) @ B_down
```

Each LoRA term `(x @ A) @ B` is a rank-bottleneck that fits entirely in SRAM for typical ranks (8–64). Fusing the base matmul with the LoRA delta avoids writing/reading the LoRA intermediate from HBM.

## Structure

```
lora_mlp/
├── experiments/v{N}/      # Versioned kernel experiments
│   └── *_v{N}.py          #   e.g. lora_mlp_kernel_v1.py
│   └── *_v{N}_upgrade_{M} #   e.g. lora_mlp_kernel_v1_upgrade_1.py
├── benchmarks/
│   ├── bench_lora_mlp.py  # Benchmark harness
│   └── results/           # CSV / JSON outputs (gitignored content)
├── tests/
│   └── test_lora_mlp.py   # Correctness + gradcheck tests
├── reference/              # Read-only reference implementations
├── docs/
│   ├── research.md        # Research context, related work, exploration axes
│   └── benchmarks.md      # Benchmark methodology, current results, analysis
├── CHANGELOG.md            # Version history + improvement log
└── README.md               # This file
```

## Versioning Scheme

| Concept | Convention | Example |
|---------|-----------|---------|
| **Version** | Fundamentally different algorithmic approach | `v1` = naive fused, `v2` = tiled SRAM LoRA |
| **Upgrade** | Tuning / bugfix within same approach | `v1_upgrade_1` = autotuned tile sizes |
| **File** | `lora_mlp_kernel_v{N}.py` or `_v{N}_upgrade_{M}.py` | `experiments/v1/lora_mlp_kernel_v1.py` |

**Rule**: never modify a previous version — copy forward and iterate. Every file is a snapshot.

## Quick Start

```bash
# Correctness tests
pytest tests/test_lora_mlp.py -v

# Benchmark sweep
python benchmarks/bench_lora_mlp.py --save benchmarks/results/

# Single-config benchmark (LLaMA-3 8B scale)
python benchmarks/bench_lora_mlp.py \
  --hidden 4096 --intermediate 14336 --rank 16 --seq 2048 --batch 4
```

## Done Gates

- [x] Forward matches PyTorch reference (rtol=1e-3, atol=1e-3 for bf16) — v2
- [x] Backward passes `torch.autograd.gradcheck` (fp64, eps=1e-6) — v2 LoRAMLPv2
- [x] Wrapped in `torch.autograd.Function` with correct `ctx.save_for_backward` — v2 LoRAMLPv2
- [x] Handles variable sequence lengths, batch sizes, and LoRA ranks — tested r=8,16,32
- [x] bf16 and fp32 dtypes supported — v2
- [ ] Memory usage <= PyTorch baseline (no extra intermediate buffers)
- [x] Benchmarked at LLaMA-3 scale (4096 hidden, 14336 intermediate, rank 16) — v3: 1.02-1.14x Unsloth
- [x] Measurable speedup over unfused PyTorch + LoRA baseline — v3: **1.02-1.14x faster than Unsloth**

## Key Parameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| Hidden dim | 4096 (8B), 5120 (13B) | Model hidden size |
| Intermediate dim | 14336 (8B), 17920 (13B) | MLP intermediate (often 3.5× hidden) |
| LoRA rank | 8, 16, 32, 64 | Low-rank bottleneck |
| Sequence length | 512 – 8192 | Tokens per sample |
| Batch size | 1 – 8 | Micro-batch |

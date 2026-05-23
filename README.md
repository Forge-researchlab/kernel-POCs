# kerna-POCs

> Proof-of-concept Triton kernels for the Forge project — high-performance custom CUDA/Triton kernels for LLM fine-tuning.

## Kernels

| # | Kernel | Branch | Priority | Description |
|---|--------|--------|----------|-------------|
| P1 | SwiGLU | `day1/p1-swiglu-kernel` | CRITICAL | Fused SwiGLU activation (SiLU path) — ~50% of model compute |
| P2 | RoPE | `day1/p2-rope-kernel` | HIGH | Rotary Position Embedding with configurable base frequency |
| P3 | Cross-Entropy | `day1/p3-cross-entropy-kernel` | HIGH | Memory-efficient cross-entropy loss (chunked online softmax) |
| P4 | LoRA MLP Fused | `day1/p4-lora-mlp-fused` | STRUCTURAL | Fused LoRA forward for MLP (gate + up + down projections) |
| P5 | LoRA QKV Fused | `day1/p5-lora-qkv-fused` | STRUCTURAL | Fused LoRA forward for QKV with GQA asymmetry handling |

## Repository Structure

```
kerna-POCs/
├── kernels/
│   ├── swiglu/          # P1: SwiGLU kernel (fwd + bwd)
│   ├── rope/            # P2: RoPE kernel (fwd + bwd)
│   ├── cross_entropy/   # P3: Cross-Entropy kernel (fwd + bwd)
│   ├── lora_mlp/        # P4: LoRA MLP fused kernel
│   └── lora_qkv/        # P5: LoRA QKV fused kernel
├── tests/
│   ├── test_swiglu.py
│   ├── test_rope.py
│   ├── test_cross_entropy.py
│   ├── test_lora_mlp.py
│   └── test_lora_qkv.py
├── benchmarks/
│   └── bench_all.py
├── requirements.txt
└── README.md
```

## Branch Strategy

Each kernel POC is developed on its own branch from `main`:

- `day1/p1-swiglu-kernel` — SwiGLU activation kernel
- `day1/p2-rope-kernel` — Rotary Position Embedding kernel
- `day1/p3-cross-entropy-kernel` — Cross-Entropy loss kernel
- `day1/p4-lora-mlp-fused` — LoRA MLP fused kernel
- `day1/p5-lora-qkv-fused` — LoRA QKV fused kernel

## Kernel Development Workflow

Each kernel follows the same 5-step build process:

1. **Forward kernel** — implement Triton kernel for forward pass, validate against PyTorch reference
2. **Backward kernel** — implement backward pass with analytical gradients
3. **Gradcheck** — verify backward correctness with `torch.autograd.gradcheck`
4. **autograd.Function** — wrap in `torch.autograd.Function` for seamless integration
5. **Benchmark** — measure throughput vs. PyTorch baseline, report speedup

### Done Gates (per kernel)

- [ ] Forward matches PyTorch reference (rtol=1e-3, atol=1e-3 for bf16)
- [ ] Backward passes `torch.autograd.gradcheck` (fp64, eps=1e-6)
- [ ] Wrapped in `autograd.Function` with correct `ctx.save_for_backward`
- [ ] Handles variable sequence lengths and batch sizes
- [ ] bf16 and fp32 dtypes supported
- [ ] No memory leaks (activation memory ≤ PyTorch baseline)
- [ ] Benchmark shows measurable speedup over naive PyTorch

## Requirements

- Python 3.10+
- PyTorch 2.4+
- Triton 3.0+
- CUDA 12.1+

## Setup

```bash
pip install -e .
```

## Context

Part of the [Forge Hackathon (May 23-24, 2026)](https://xhitijc2.github.io/forge-hackathon-plan/index.html) — Day 1 kernel sprint with 5 parallel tracks.

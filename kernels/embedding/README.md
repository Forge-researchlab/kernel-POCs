# Embedding Kernel

Research project to build a high-performance Triton replacement for `torch.nn.Embedding` — targeting both forward (gather) and backward (gradient scatter-add) passes.

## The Problem

The backward pass of embedding must accumulate gradients for all positions that share the same token index. In real training, tokens like padding, BOS/EOS, and high-frequency words can appear hundreds of times per batch. Naive approaches (PyTorch's `index_add_`, Liger's `atomic_add`) leave performance on the table under heavy duplication.

## Structure

```
embedding/
├── reference/             # Reference implementations (Liger, etc.)
├── experiments/v{N}/      # Versioned kernel experiments
├── benchmarks/
│   ├── bench_embedding.py # Benchmark harness
│   └── results/           # CSV outputs
├── tests/
│   └── test_embedding.py  # Correctness tests
└── docs/                  # Design notes
```

## Quick Start

```bash
# Correctness tests
pytest tests/test_embedding.py -v

# Benchmark sweep
python benchmarks/bench_embedding.py --save benchmarks/results/

# Single-config benchmark
python benchmarks/bench_embedding.py --vocab 128256 --dim 4096 --seq 2048
```

## Done Gates

- [ ] Forward matches PyTorch reference (bf16 + fp32)
- [ ] Backward matches PyTorch gradient across duplicate ratios
- [ ] `gradcheck` passes in fp64
- [ ] Benchmarked at LLaMA-3 scale (128k vocab, 4096 dim, 8k seq)
- [ ] Memory usage <= PyTorch baseline
- [ ] Measurable speedup over both PyTorch and Liger

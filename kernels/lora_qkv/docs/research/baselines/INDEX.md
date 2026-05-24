# Known Baselines

> Catalog of known baseline implementations and their characteristics.
> Updated with measured numbers from our benchmarks.

| Baseline | Source | Approach | Time (r=16, QKV) | Memory | Launches | X reads | Strengths | Weaknesses |
|----------|--------|----------|-----------------|--------|----------|---------|-----------|------------|
| cuBLAS bare (no LoRA) | torch.matmul | 3× GEMM | ~1.7ms | ~96 MB | 3 | 3× | Speed floor — cuBLAS optimized | No LoRA |
| PyTorch naive | our reference | 3× (matmul + XA + add) | 2.285ms | 192 MB | 9+ | 6× | Simple, correct | Slow, memory-hungry |
| **Unsloth (addmm\_)** | unsloth/kernels | 3× (matmul + XA + addmm\_) | **2.035ms** | **96 MB** | **9** | **6×** | **Primary target**. In-place, fused add+GEMM | No kernel-level fusion, X read 6× |
| Packed QKV (no LoRA) | qkv_fusion | 1× GEMM + split | ~0.9ms | ~96 MB | 2 | 1× | 2x vs 3× nn.Linear | No LoRA handling |
| **Our v1** | experiments/v1 | 3× Triton fused LoRA | 3.624ms | 96 MB | 3 | 6× | Single kernel per proj | 0.56x Unsloth — Triton matmul gap |
| **Our v2** | experiments/v2 | 3× (packed cuBLAS + epilogue) | **1.869ms** | 128 MB | 6 | 3× | **1.09x Unsloth**, rank-independent | 33% more memory |

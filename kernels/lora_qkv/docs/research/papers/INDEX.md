# Research Papers & Findings Index

> Master index of all papers, blog posts, and code findings.
> Sorted by relevance to the current project. Updated after each research session.

| Date Found | Title | Source | Relevance | Key Takeaway |
|------------|-------|--------|-----------|--------------|
| 2026-05-24 | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | [arxiv 2605.19269](https://arxiv.org/html/2605.19269) | **high** | Formalizes our cuBLAS+epilogue pattern; shows epilogue can pipeline with GEMM on Hopper |
| 2026-05-24 | qkv_fusion — Packed QKV with cuBLAS | [GitHub](https://github.com/hilaryKChen/qkv_fusion) | **high** | 2x speedup from packing Q+K+V into single GEMM; validates our fully-packed approach |
| 2026-05-24 | vLLM 6-way fused projection (Qwen3.5) | [vLLM PR #41457](https://github.com/vllm-project/vllm/pull/41457) | **high** | Packs 6 projections into one GEMM; confirms ~0.5% output dim overhead is negligible |
| 2026-05-24 | addmm is not always better than mm+add | [PyTorch #141210](https://github.com/pytorch/pytorch/issues/141210) | **high** | addmm_ 1.5x slower than mm+add at large shapes; validates our epilogue approach |
| 2026-05-24 | vLLM fused MoE LoRA Triton kernel | [vLLM #31912](https://github.com/vllm-project/vllm/issues/31912) | medium | Two-stage LoRA (shrink+expand) in Triton for MoE; validates fused LoRA kernels in production |
| 2026-05-24 | Triton Grouped GEMM Tutorial | [triton-lang](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) | medium | Handles different M/N/K per group in one launch; could fuse Q/K/V with different N dims |
| 2026-05-24 | ForkKV: Multi-LoRA ResidualAttention | [arxiv 2604.06370](https://arxiv.org/html/2604.06370v1) | low | Fuses LoRA into attention kernel for serving; different use case (inference, not training) |
| 2026-05-24 | Cutting LLM Memory by 84% (fused kernels blog) | [Medium](https://medium.com/data-science-collective/cutting-llm-memory-by-84-a-deep-dive-into-fused-kernels-7028ca28bb75) | low | Cross-entropy memory reduction patterns; useful for general kernel design reference |
| 2026-05-24 | Triton Persistent Matmul Tutorial | [triton-lang](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | low | Persistent kernels keep SMs busy; more complex, defer to future |

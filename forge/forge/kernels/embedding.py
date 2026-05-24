"""Forge Embedding — sort-based gradient scatter-add to avoid atomic contention.

The forward kernel is a parallel gather; the backward sorts indices, groups by
unique token, then runs a fused reduction (with a two-level cooperative path
for high-duplicate cases). This avoids the atomic_add storm that PyTorch's
index_add_ and Liger's atomic-based scatter both suffer from when a few high-
frequency tokens (padding, BOS, EOS) dominate the batch.

API:
    ForgeEmbeddingFunction.apply(weight, indices) -> embeddings
        weight:  (vocab_size, embedding_dim)
        indices: (batch, seq) or any int tensor; flattened internally

Wired into both QWEN3_MAPPING and GEMMA_MAPPING — `nn.Embedding` patches use this.
"""
from kernels.embedding.experiments.v1.embedding_kernel_v1_upgrade_1 import (
    ForgeEmbeddingFunction,
)

__all__ = ["ForgeEmbeddingFunction"]

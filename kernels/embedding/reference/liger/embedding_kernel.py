"""
Liger Kernel - Custom Triton Embedding Implementation
=====================================================

This is the Liger Kernel's reimplementation of PyTorch's nn.Embedding
using Triton GPU kernels for better performance.

See: explanation.md for visual diagrams of how this works.
"""

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


# =============================================================================
# FORWARD KERNEL: Gather operation (scattered read -> contiguous write)
# =============================================================================
@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,      # pointer to the weight matrix (vocab_size x embedding_dim)
    indices_ptr,         # pointer to input token indices
    output_ptr,          # pointer to output tensor
    n_elements,          # total number of indices (flattened)
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,   # block size along the tokens axis
    BLOCK_SIZE_N: tl.constexpr,   # block size along the embedding_dim axis
):
    # Each program instance handles a tile of (BLOCK_SIZE_M tokens) x (BLOCK_SIZE_N dims)
    pid_m = tl.program_id(0)  # which block of tokens
    pid_n = tl.program_id(1)  # which block of embedding dimensions

    # Compute offsets for this block
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)  # token offsets
    mask_m = offsets_m < n_elements                     # bounds check

    # Load the actual vocabulary indices for this block of tokens
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)  # dim offsets
    mask_n = offsets_n < embedding_dim                  # bounds check

    # KEY OPERATION: compute where each embedding lives in the weight matrix
    # indices[:, None] * embedding_dim  -> row start for each token
    # + offsets_n[None, :]              -> column offset within that row
    # This is a SCATTERED read - rows are not contiguous in general
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # Write to CONTIGUOUS output positions
    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets,
        embeddings,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# =============================================================================
# BACKWARD KERNEL: Scatter-add operation (contiguous read -> scattered atomic write)
# =============================================================================
@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,     # gradient flowing back from the next layer
    grad_weight_ptr,     # gradient for the weight matrix (accumulate here)
    indices_ptr,         # same indices from forward pass
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    # Read gradients from CONTIGUOUS output positions
    grad_output = tl.load(
        grad_output_ptr + offsets_m[:, None] * embedding_dim + offsets_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # Write gradients to SCATTERED weight matrix positions
    # MUST use atomic_add because multiple tokens can map to the same row
    grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    tl.atomic_add(
        grad_weight_ptr + grad_weight_offsets,
        grad_output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# =============================================================================
# AUTOGRAD WRAPPER
# =============================================================================
class LigerEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ori_shape = indices.shape
        indices = indices.view(-1)  # flatten to 1D
        output = torch.empty(
            indices.shape[0],
            embeddings.shape[1],
            device=indices.device,
            dtype=embeddings.dtype,
        )

        n_elements = indices.numel()
        embedding_dim = embeddings.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embedding_dim, BLOCK_SIZE_N),
        )

        embedding_forward_kernel[grid](
            embeddings,
            indices,
            output,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(indices, embeddings)
        return output.view(*ori_shape, -1)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embedding_table = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, embedding_table.shape[1])

        grad_weight = torch.zeros_like(embedding_table)

        n_elements = indices.numel()
        embedding_dim = embedding_table.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embedding_dim, BLOCK_SIZE_N),
        )

        embedding_backward_kernel[grid](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return grad_weight, None  # None because indices aren't differentiable

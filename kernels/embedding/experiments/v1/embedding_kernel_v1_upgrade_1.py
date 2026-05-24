"""
Forge Embedding Kernel v1_upgrade_1 — Cooperative Group Reduction
=================================================================

Upgrade over v1: fixes the high-duplicate / low-unique-count regression.

Problem in v1:
  When there are few unique tokens but many duplicates (e.g., 500 unique,
  65 dups/token), v1 launches only 500 programs, each doing a long serial
  loop. This underutilizes the GPU.

Fix — two-level cooperative reduction:
  1. Split each large group across CHUNKS_PER_GROUP sub-programs.
     Each sub-program sums its slice of the group → partial sum.
  2. Partial sums are written to a small scratch buffer.
  3. A second lightweight kernel reduces the partial sums per group
     and writes the final result to grad_weight.

  For groups smaller than CHUNKS_PER_GROUP, only one sub-program runs
  (no overhead). The split only kicks in when groups are large enough.

Everything else (forward kernel, autograd wrapper) is identical to v1.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# FORWARD KERNEL (same as v1 — autotuned)
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256}),
    ],
    key=["n_elements", "embedding_dim"],
)
@triton.jit
def forge_embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
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

    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets,
        embeddings,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# =============================================================================
# BACKWARD KERNEL — original v1 (for small groups / low duplicate counts)
# =============================================================================
@triton.jit
def forge_backward_fused_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    sorted_order_ptr,
    group_offsets_ptr,
    unique_indices_ptr,
    n_groups,
    embedding_dim: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= n_groups:
        return

    token_id = tl.load(unique_indices_ptr + pid_m).to(tl.int64)
    group_start = tl.load(group_offsets_ptr + pid_m)
    group_end = tl.load(group_offsets_ptr + pid_m + 1)

    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for i in range(MAX_GROUP_SIZE):
        if group_start + i < group_end:
            orig_pos = tl.load(sorted_order_ptr + group_start + i).to(tl.int64)
            grad_row = tl.load(
                grad_output_ptr + orig_pos * embedding_dim + offsets_n,
                mask=mask_n, other=0.0,
            )
            acc += grad_row.to(tl.float32)

    tl.store(
        grad_weight_ptr + token_id * embedding_dim + offsets_n,
        acc.to(tl.float32), mask=mask_n,
    )


# =============================================================================
# COOPERATIVE BACKWARD — Phase 1: partial sums per chunk
# =============================================================================
@triton.jit
def forge_backward_cooperative_phase1(
    grad_output_ptr,
    partial_sums_ptr,       # [n_groups, CHUNKS_PER_GROUP, embedding_dim]
    sorted_order_ptr,
    group_offsets_ptr,
    n_groups,
    embedding_dim: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,       # max elements per sub-program
    CHUNKS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # pid_m = group_idx * CHUNKS_PER_GROUP + chunk_idx
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    group_idx = pid_m // CHUNKS_PER_GROUP
    chunk_idx = pid_m % CHUNKS_PER_GROUP

    if group_idx >= n_groups:
        return

    group_start = tl.load(group_offsets_ptr + group_idx)
    group_end = tl.load(group_offsets_ptr + group_idx + 1)

    # This chunk handles [chunk_start, chunk_end) within the group
    chunk_start = group_start + chunk_idx * CHUNK_SIZE
    chunk_end = group_start + (chunk_idx + 1) * CHUNK_SIZE
    # Clamp to actual group end
    actual_end = tl.minimum(chunk_end, group_end)

    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for i in range(CHUNK_SIZE):
        pos = chunk_start + i
        if pos < actual_end:
            orig_pos = tl.load(sorted_order_ptr + pos).to(tl.int64)
            grad_row = tl.load(
                grad_output_ptr + orig_pos * embedding_dim + offsets_n,
                mask=mask_n, other=0.0,
            )
            acc += grad_row.to(tl.float32)

    # Write partial sum: partial_sums[group_idx, chunk_idx, start_n:start_n+BLOCK_SIZE_N]
    partial_offset = (group_idx * CHUNKS_PER_GROUP + chunk_idx) * embedding_dim + offsets_n
    tl.store(
        partial_sums_ptr + partial_offset,
        acc.to(tl.float32), mask=mask_n,
    )


# =============================================================================
# COOPERATIVE BACKWARD — Phase 2: reduce partial sums → grad_weight
# =============================================================================
@triton.jit
def forge_backward_cooperative_phase2(
    partial_sums_ptr,       # [n_groups, CHUNKS_PER_GROUP, embedding_dim]
    grad_weight_ptr,
    unique_indices_ptr,
    group_offsets_ptr,
    n_groups,
    embedding_dim: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # group index
    pid_n = tl.program_id(1)  # dim tile

    if pid_m >= n_groups:
        return

    token_id = tl.load(unique_indices_ptr + pid_m).to(tl.int64)
    group_start = tl.load(group_offsets_ptr + pid_m)
    group_end = tl.load(group_offsets_ptr + pid_m + 1)
    group_size = group_end - group_start

    # How many chunks actually have data for this group
    n_active_chunks = (group_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for c in range(CHUNKS_PER_GROUP):
        if c < n_active_chunks:
            partial_offset = (pid_m * CHUNKS_PER_GROUP + c) * embedding_dim + offsets_n
            partial = tl.load(
                partial_sums_ptr + partial_offset,
                mask=mask_n, other=0.0,
            )
            acc += partial

    tl.store(
        grad_weight_ptr + token_id * embedding_dim + offsets_n,
        acc.to(tl.float32), mask=mask_n,
    )


# =============================================================================
# THRESHOLDS & CONSTANTS
# =============================================================================
SORT_BACKWARD_THRESHOLD = 256
# If max group size exceeds this, use cooperative (split) backward
COOPERATIVE_GROUP_THRESHOLD = 32
# Target chunk size — each sub-program processes this many elements
TARGET_CHUNK_SIZE = 32


# =============================================================================
# AUTOGRAD WRAPPER
# =============================================================================
class ForgeEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor,
                padding_idx: int = None):
        """Forge embedding lookup.

        Args:
            embeddings:  (vocab_size, embedding_dim) weight table.
            indices:     int tensor of token ids, any shape; flattened internally.
            padding_idx: matches nn.Embedding's contract — if set, the gradient
                         row at this index is zeroed in backward so the pad
                         embedding stays frozen during training. PyTorch's
                         nn.Embedding does this; without it the pad row drifts
                         under SGD (subtle but real — caught by the Gemma
                         padding-stress test, see forge/tests/verify_patch_gemma.py).
        """
        embeddings = embeddings.contiguous()
        indices = indices.contiguous()

        ori_shape = indices.shape
        indices_flat = indices.view(-1)
        output = torch.empty(
            indices_flat.shape[0],
            embeddings.shape[1],
            device=indices.device,
            dtype=embeddings.dtype,
        )

        n_elements = indices_flat.numel()
        embedding_dim = embeddings.shape[1]

        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE_M"]),
            triton.cdiv(embedding_dim, meta["BLOCK_SIZE_N"]),
        )

        forge_embedding_forward_kernel[grid](
            embeddings,
            indices_flat,
            output,
            n_elements,
            embedding_dim=embedding_dim,
        )

        ctx.save_for_backward(indices_flat, embeddings)
        ctx.ori_shape = ori_shape
        ctx.padding_idx = padding_idx
        return output.view(*ori_shape, -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices_flat, embedding_table = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, embedding_table.shape[1])

        n_elements = indices_flat.numel()
        embedding_dim = embedding_table.shape[1]
        vocab_size = embedding_table.shape[0]

        grad_weight = torch.zeros(vocab_size, embedding_dim,
                                  device=grad_output.device, dtype=embedding_table.dtype)

        if n_elements >= SORT_BACKWARD_THRESHOLD:
            # Sort indices on GPU
            sorted_indices, sorted_order = torch.sort(indices_flat, stable=True)

            # Find unique tokens and group boundaries
            unique_tokens, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
            n_groups = unique_tokens.shape[0]

            # Group offsets (int32 for fast Triton loads)
            group_offsets = torch.zeros(n_groups + 1, dtype=torch.int32, device=indices_flat.device)
            torch.cumsum(counts.int(), dim=0, out=group_offsets[1:])

            max_gs = int(counts.max().item())
            BLOCK_SIZE_N = min(256, triton.next_power_of_2(embedding_dim))

            if max_gs > COOPERATIVE_GROUP_THRESHOLD:
                # ── Cooperative path: split large groups across sub-programs ──
                CHUNK_SIZE = triton.next_power_of_2(TARGET_CHUNK_SIZE)
                chunks_needed = (max_gs + CHUNK_SIZE - 1) // CHUNK_SIZE
                CHUNKS_PER_GROUP = triton.next_power_of_2(chunks_needed)

                # Allocate scratch for partial sums
                partial_sums = torch.zeros(
                    n_groups * CHUNKS_PER_GROUP, embedding_dim,
                    device=grad_output.device, dtype=torch.float32,
                )

                # Phase 1: each sub-program sums its chunk
                grid_p1 = (n_groups * CHUNKS_PER_GROUP, triton.cdiv(embedding_dim, BLOCK_SIZE_N))
                forge_backward_cooperative_phase1[grid_p1](
                    grad_output,
                    partial_sums,
                    sorted_order.int(),
                    group_offsets,
                    n_groups,
                    embedding_dim=embedding_dim,
                    CHUNK_SIZE=CHUNK_SIZE,
                    CHUNKS_PER_GROUP=CHUNKS_PER_GROUP,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                )

                # Phase 2: reduce partial sums → grad_weight
                grid_p2 = (n_groups, triton.cdiv(embedding_dim, BLOCK_SIZE_N))
                forge_backward_cooperative_phase2[grid_p2](
                    partial_sums,
                    grad_weight,
                    unique_tokens.int(),
                    group_offsets,
                    n_groups,
                    embedding_dim=embedding_dim,
                    CHUNK_SIZE=CHUNK_SIZE,
                    CHUNKS_PER_GROUP=CHUNKS_PER_GROUP,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                )
            else:
                # ── Original v1 path: groups are small, serial loop is fine ──
                MAX_GROUP_SIZE = triton.next_power_of_2(max_gs)
                grid = (n_groups, triton.cdiv(embedding_dim, BLOCK_SIZE_N))

                forge_backward_fused_kernel[grid](
                    grad_output,
                    grad_weight,
                    sorted_order.int(),
                    group_offsets,
                    unique_tokens.int(),
                    n_groups,
                    embedding_dim=embedding_dim,
                    MAX_GROUP_SIZE=MAX_GROUP_SIZE,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                )
        else:
            grad_weight.index_add_(0, indices_flat, grad_output)

        # Match nn.Embedding(padding_idx=...) semantics: the pad row's gradient
        # is dropped so the pad embedding stays frozen across training. Without
        # this the kernel silently diverges from the unpatched path on any model
        # that sets padding_idx (Gemma, Llama, Qwen all do by default).
        # Universal fix: applies to both the sort path and the index_add fallback.
        if ctx.padding_idx is not None:
            grad_weight[ctx.padding_idx].zero_()

        return grad_weight, None, None

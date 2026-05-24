"""
v1 — Fused LoRA Matmul Kernel

Computes Y = X @ W^T + s * (X @ A^T) @ B^T in a single Triton kernel launch.

Approach:
  - Output-stationary tiled matmul over the base weight W.
  - After the K-loop for W, compute the LoRA term for the same output tile:
    1. Load the X tile rows against the full A matrix (r columns) to get
       XA_tile of shape [BLOCK_M, r] — fits in registers for r <= 64.
    2. Load the B rows for this output tile's columns to get B_tile [r, BLOCK_N].
    3. Compute XA_tile @ B_tile -> [BLOCK_M, BLOCK_N], add to accumulator.
  - Single kernel launch replaces Unsloth's 3 cuBLAS calls per projection.

Constraints:
  - LoRA rank r must be a power of 2 and <= BLOCK_R (compile-time constant).
  - All accumulation in fp32, output cast back to input dtype.
  - W stored as [out_dim, in_dim] (nn.Linear convention), accessed as W^T.

Known limitations:
  - Forward pass only (no backward yet — that's v3+).
  - LoRA rank padded to power of 2 inside the kernel (wastes a few registers for r=24 etc).
  - No autotune yet — fixed block sizes for initial correctness validation.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K", "HAS_LORA", "FP32_PRECISE"],
)
@triton.jit
def _fused_lora_matmul_kernel(
    # Pointers
    X_ptr, W_ptr, A_ptr, B_ptr, Y_ptr,
    # Scalar
    lora_scale,
    # Dimensions
    M, N, K, R,
    # Strides for X [M, K]
    stride_xm, stride_xk,
    # Strides for W [N, K] (stored as [out_dim, in_dim])
    stride_wn, stride_wk,
    # Strides for A [R, K] (LoRA A: [r, in_dim])
    stride_ar, stride_ak,
    # Strides for B [N, R] (LoRA B: [out_dim, r])
    stride_bn, stride_br,
    # Strides for Y [M, N]
    stride_ym, stride_yn,
    # Whether LoRA is active
    HAS_LORA: tl.constexpr,
    # Use IEEE fp32 precision (slower but exact; only matters for fp32 inputs)
    FP32_PRECISE: tl.constexpr,
    # Padded rank (power of 2 >= R)
    BLOCK_R: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ── Fused K-loop: accumulate both X@W^T and X@A^T simultaneously ──
    # This reads X from HBM only once, computing both products in the same pass.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_LORA:
        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        offs_r = tl.arange(0, BLOCK_R)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile [BLOCK_M, BLOCK_K] — loaded ONCE, used for both W and A
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Base matmul: load W tile [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += X_tile @ W_tile^T = [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        if FP32_PRECISE:
            acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

        # LoRA: accumulate X @ A^T in the same loop (reuses x_tile from L1/registers)
        if HAS_LORA:
            a_ptrs = A_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak
            a_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            if FP32_PRECISE:
                xa = tl.dot(x_tile, tl.trans(a_tile), acc=xa, input_precision="ieee")
            else:
                xa = tl.dot(x_tile, tl.trans(a_tile), acc=xa)

    # ── LoRA: finish with (XA) @ B^T ──
    if HAS_LORA:
        # Load B tile [BLOCK_R, BLOCK_N] for this output tile's columns
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
        b_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        if FP32_PRECISE:
            lora_out = tl.dot(xa, b_tile.to(tl.float32), input_precision="ieee")
        else:
            lora_out = tl.dot(xa.to(b_tile.dtype), b_tile)

        acc += lora_scale * lora_out

    # ── Store output ──
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


def fused_lora_matmul(
    X: torch.Tensor,
    W: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute Y = X @ W^T + lora_scale * (X @ A^T) @ B^T in a single kernel.

    Args:
        X: [M, K] or [B, S, K] input tensor
        W: [N, K] weight matrix (nn.Linear convention)
        A: [r, K] LoRA down-projection (optional)
        B: [N, r] LoRA up-projection (optional)
        lora_scale: scalar scaling for the LoRA term

    Returns:
        Y: [M, N] or [B, S, N]
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    M, K = X.shape
    N = W.shape[0]
    assert W.shape[1] == K, f"W shape mismatch: {W.shape} vs X {X.shape}"

    has_lora = A is not None and B is not None
    if has_lora:
        R = A.shape[0]
        assert A.shape == (R, K), f"A shape mismatch: {A.shape}, expected ({R}, {K})"
        assert B.shape == (N, R), f"B shape mismatch: {B.shape}, expected ({N}, {R})"
        BLOCK_R = triton.next_power_of_2(R)
        BLOCK_R = max(BLOCK_R, 16)
    else:
        R = 0
        BLOCK_R = 16

    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_lora_matmul_kernel[grid](
        X, W,
        A if has_lora else X,  # dummy pointer when no LoRA
        B if has_lora else X,  # dummy pointer when no LoRA
        Y,
        lora_scale,
        M, N, K, R,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        (A.stride(0) if has_lora else 1),
        (A.stride(1) if has_lora else 1),
        (B.stride(0) if has_lora else 1),
        (B.stride(1) if has_lora else 1),
        Y.stride(0), Y.stride(1),
        HAS_LORA=has_lora,
        FP32_PRECISE=(X.dtype == torch.float32),
        BLOCK_R=BLOCK_R,
    )

    if len(orig_shape) == 3:
        Y = Y.view(orig_shape[0], orig_shape[1], N)
    return Y

"""
v1 — Per-Projection Fused LoRA Matmul for QKV

Computes Y = X @ W^T + s * (X @ A^T) @ B^T in a single Triton kernel launch.
Applied independently to each of Q, K, V (3 launches total for full QKV).

Approach:
  Output-stationary tiled matmul with L2 cache swizzle (GROUP_SIZE_M).
  The K-loop accumulates X @ W^T in the main accumulator. After the K-loop,
  the LoRA term is computed as a post-pass:
    1. Second K-loop over X and A to get XA_tile [BLOCK_M, BLOCK_R]
    2. Load B_tile [BLOCK_R, BLOCK_N] for this output tile
    3. acc += s * XA_tile @ B_tile

  This reads X from HBM twice (once for W loop, once for A loop), but keeps
  XA in registers and avoids materializing it to HBM. Unsloth reads X twice
  too, so this is bandwidth-neutral vs Unsloth for the per-projection case.
  The real win comes in v2 where we fuse across Q/K/V.

Improvements over lora_mlp v1:
  - L2 cache swizzle (GROUP_SIZE_M) from the start — lora_mlp v1 missed this
    and paid ~10-15% penalty on base matmul
  - More autotune configs including larger tiles and higher num_stages
  - Separate post-pass for LoRA instead of fused K-loop — avoids register
    pressure from carrying an XA accumulator through the entire K-loop
  - Cleaner A/B loading with explicit masking

Known limitations (to address in v1_2+):
  - Forward pass only (no backward — that's v3)
  - LoRA rank padded to next power of 2
  - No cross-projection fusion (Q, K, V computed independently)
  - Triton matmul likely ~0.73x cuBLAS (known from lora_mlp; v3 will use
    cuBLAS + Triton epilogue to fix this)

Shapes (LLaMA-3 8B):
  Q: X [M, 4096] @ W_q [4096, 4096] + LoRA → [M, 4096]
  K: X [M, 4096] @ W_k [1024, 4096] + LoRA → [M, 1024]  (GQA: 8 KV heads)
  V: X [M, 4096] @ W_v [1024, 4096] + LoRA → [M, 1024]
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K", "HAS_LORA", "FP32_PRECISE"],
)
@triton.jit
def _fused_lora_matmul_kernel(
    X_ptr, W_ptr, A_ptr, B_ptr, Y_ptr,
    lora_scale,
    M, N, K, R,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ar, stride_ak,
    stride_bn, stride_br,
    stride_ym, stride_yn,
    HAS_LORA: tl.constexpr,
    FP32_PRECISE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache swizzle: reorder blocks for better L2 reuse.
    # Groups of GROUP_SIZE_M rows share access to the same W columns,
    # keeping W tiles in L2 across adjacent thread blocks.
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # === Base matmul K-loop: acc += X @ W^T ===
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        if FP32_PRECISE:
            acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    # === LoRA post-pass: acc += s * (X @ A^T) @ B^T ===
    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)

        # Step 1: Compute XA = X @ A^T → [BLOCK_M, BLOCK_R]
        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            a_ptrs = A_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak
            a_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            if FP32_PRECISE:
                xa = tl.dot(x_tile, tl.trans(a_tile), acc=xa, input_precision="ieee")
            else:
                xa = tl.dot(x_tile, tl.trans(a_tile), acc=xa)

        # Step 2: Load B tile and compute (XA) @ B^T → [BLOCK_M, BLOCK_N]
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
        b_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        if FP32_PRECISE:
            lora_out = tl.dot(xa, b_tile.to(tl.float32), input_precision="ieee")
        else:
            lora_out = tl.dot(xa.to(b_tile.dtype), b_tile)

        acc += lora_scale * lora_out

    # === Store output ===
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


# ============================================================
# Python wrapper: single projection
# ============================================================

def fused_lora_matmul(
    X: torch.Tensor,
    W: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """
    Fused LoRA matmul: Y = X @ W^T + lora_scale * (X @ A^T) @ B^T

    Single Triton kernel launch. Replaces Unsloth's 3 cuBLAS calls.

    Args:
        X: [M, K] input tensor (or [B, S, K] — will be flattened)
        W: [N, K] weight matrix (nn.Linear convention)
        A: [r, K] LoRA down-projection (or None)
        B: [N, r] LoRA up-projection (or None)
        lora_scale: LoRA scaling factor (alpha / rank)

    Returns:
        Y: [M, N] (or [B, S, N] if input was 3D)
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    M, K = X.shape
    N = W.shape[0]
    assert W.shape[1] == K, f"W shape mismatch: W is {W.shape}, X is {X.shape}"

    has_lora = A is not None and B is not None
    if has_lora:
        R = A.shape[0]
        assert A.shape == (R, K), f"A shape {A.shape} != ({R}, {K})"
        assert B.shape == (N, R), f"B shape {B.shape} != ({N}, {R})"
        BLOCK_R = max(triton.next_power_of_2(R), 16)
    else:
        R = 0
        BLOCK_R = 16

    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_lora_matmul_kernel[grid](
        X, W,
        A if has_lora else X,
        B if has_lora else X,
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


# ============================================================
# Full QKV forward: 3 fused kernel launches
# ============================================================

def lora_qkv_v1(
    X: torch.Tensor,
    W_q: torch.Tensor, A_q: Optional[torch.Tensor], B_q: Optional[torch.Tensor], s_q: float,
    W_k: torch.Tensor, A_k: Optional[torch.Tensor], B_k: Optional[torch.Tensor], s_k: float,
    W_v: torch.Tensor, A_v: Optional[torch.Tensor], B_v: Optional[torch.Tensor], s_v: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full QKV forward using v1 fused LoRA matmul.

    3 kernel launches (one per projection) vs Unsloth's 9 cuBLAS calls.
    Supports GQA (W_k/W_v can have different output dimensions from W_q).

    Args:
        X: input [M, H] or [B, S, H]
        W_q/W_k/W_v: frozen weights [N_q, H], [N_kv, H], [N_kv, H]
        A_q/A_k/A_v: LoRA A matrices [r, H] or None
        B_q/B_k/B_v: LoRA B matrices [N_*, r] or None
        s_q/s_k/s_v: LoRA scaling factors

    Returns:
        (Q, K, V) tuple
    """
    Q = fused_lora_matmul(X, W_q, A_q, B_q, s_q)
    K = fused_lora_matmul(X, W_k, A_k, B_k, s_k)
    V = fused_lora_matmul(X, W_v, A_v, B_v, s_v)
    return Q, K, V

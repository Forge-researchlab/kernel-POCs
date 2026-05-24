"""
v2 — Packed W+A cuBLAS Matmul + Triton LoRA Epilogue

Strategy (applying lora_mlp v3 lesson: cuBLAS for matmuls, Triton for fusion):
  1. Pack W and A vertically: W_packed = cat([W, A], dim=0)  → [N+r, K]
  2. One cuBLAS call: out_packed = X @ W_packed^T             → [M, N+r]
  3. Split: base_out = out_packed[:, :N], XA = out_packed[:, N:]
  4. Triton epilogue: Y = base_out + s * XA @ B^T

This reads X only ONCE per projection (vs Unsloth reading X TWICE: once for W,
once for A). The cuBLAS matmul is ~0.4% larger (4096+16=4112 vs 4096 output dims
for rank=16) — negligible overhead.

Pipeline for full QKV:
  1. cuBLAS: Q_packed = X @ [W_q; A_q]^T    → [M, H_q+r]    (X read once)
  2. cuBLAS: K_packed = X @ [W_k; A_k]^T    → [M, H_kv+r]   (X read once)
  3. cuBLAS: V_packed = X @ [W_v; A_v]^T    → [M, H_kv+r]   (X read once)
  4. Triton: Q = Q_packed[:,:H_q] + s_q * Q_packed[:,H_q:] @ B_q^T
  5. Triton: K = K_packed[:,:H_kv] + s_k * K_packed[:,H_kv:] @ B_k^T
  6. Triton: V = V_packed[:,:H_kv] + s_v * V_packed[:,H_kv:] @ B_v^T

Total: 3 cuBLAS + 3 Triton = 6 launches, X read 3 times.
vs Unsloth: 9 cuBLAS launches, X read 6 times.

The Triton epilogue is bandwidth-bound (not compute-bound), so it's very fast:
it reads [M, N] + [M, r], computes a tiny [r, BLOCK_N] dot, and writes [M, N].

Known limitations:
  - Requires pre-packing W and A (one-time cost before training loop)
  - W_packed must be re-created if LoRA weights change shape (rare)
  - Forward only (backward in v3)
  - The packed matmul output is slightly larger ([M, N+r] vs [M, N])
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# ============================================================
# Triton epilogue: Y = base_out + s * XA @ B^T
# ============================================================

@triton.jit
def _lora_epilogue_kernel(
    BASE_ptr,       # [M, N] — base matmul output (first N cols of packed output)
    XA_ptr,         # [M, R] — LoRA intermediate (last r cols of packed output)
    B_ptr,          # [N, R] — LoRA B matrix
    Y_ptr,          # [M, N] — final output
    lora_scale,
    M, N, R,
    stride_base_m, stride_base_n,
    stride_xa_m, stride_xa_r,
    stride_bn, stride_br,
    stride_ym, stride_yn,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load base output tile [BLOCK_M, BLOCK_N]
    base_ptrs = BASE_ptr + offs_m[:, None] * stride_base_m + offs_n[None, :] * stride_base_n
    base_tile = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load XA tile [BLOCK_M, R]
    xa_ptrs = XA_ptr + offs_m[:, None] * stride_xa_m + offs_r[None, :] * stride_xa_r
    xa_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    xa_tile = tl.load(xa_ptrs, mask=xa_mask, other=0.0)

    # Load B tile [R, BLOCK_N] (B stored as [N, R], need transpose)
    b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
    b_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Tiny matmul: XA @ B^T → [BLOCK_M, BLOCK_N]
    lora_out = tl.dot(xa_tile, b_tile).to(tl.float32)

    # Add LoRA to base
    result = base_tile + lora_scale * lora_out

    # Store
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, result.to(Y_ptr.dtype.element_ty), mask=mask)


def lora_epilogue(
    base_out: torch.Tensor,
    xa: torch.Tensor,
    B: torch.Tensor,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """
    LoRA epilogue: Y = base_out + s * XA @ B^T

    Args:
        base_out: [M, N] base matmul output
        xa: [M, r] LoRA intermediate (X @ A^T)
        B: [N, r] LoRA B matrix
        lora_scale: scaling factor

    Returns:
        Y: [M, N]
    """
    M, N = base_out.shape
    R = xa.shape[1]
    BLOCK_R = max(triton.next_power_of_2(R), 16)

    Y = torch.empty_like(base_out)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _lora_epilogue_kernel[grid](
        base_out, xa, B, Y,
        lora_scale,
        M, N, R,
        base_out.stride(0), base_out.stride(1),
        xa.stride(0), xa.stride(1),
        B.stride(0), B.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return Y


# ============================================================
# Per-projection: packed cuBLAS + Triton epilogue
# ============================================================

def fused_lora_matmul_v2(
    X: torch.Tensor,
    W: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    lora_scale: float = 1.0,
    W_packed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Packed W+A cuBLAS matmul + Triton LoRA epilogue.

    If W_packed is provided, uses it directly (pre-packed for hot loop).
    Otherwise, packs W and A on the fly.

    Pipeline:
      1. cuBLAS: out_packed = X @ [W; A]^T      (X read ONCE)
      2. Split: base_out, XA = out_packed[:, :N], out_packed[:, N:]
      3. Triton: Y = base_out + s * XA @ B^T     (bandwidth-bound epilogue)

    Args:
        X: [M, K] input
        W: [N, K] frozen weight
        A: [r, K] LoRA A (or None)
        B: [N, r] LoRA B (or None)
        lora_scale: scaling factor
        W_packed: [N+r, K] pre-packed cat([W, A], dim=0) (optional)

    Returns:
        Y: [M, N]
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    M, K = X.shape
    N = W.shape[0]
    has_lora = A is not None and B is not None

    if not has_lora:
        Y = torch.matmul(X, W.t())
        if len(orig_shape) == 3:
            Y = Y.view(orig_shape[0], orig_shape[1], N)
        return Y

    R = A.shape[0]

    # Pack W and A if not pre-packed
    if W_packed is None:
        W_packed = torch.cat([W, A], dim=0)  # [N+r, K]

    # Single cuBLAS call: reads X once, outputs [M, N+r]
    out_packed = torch.matmul(X, W_packed.t())

    # Split into base output and LoRA intermediate
    base_out = out_packed[:, :N]     # [M, N]
    xa = out_packed[:, N:N+R]        # [M, r]

    # Triton epilogue: base_out + s * XA @ B^T
    Y = lora_epilogue(base_out, xa, B, lora_scale)

    if len(orig_shape) == 3:
        Y = Y.view(orig_shape[0], orig_shape[1], N)
    return Y


# ============================================================
# Full QKV forward
# ============================================================

def lora_qkv_v2(
    X: torch.Tensor,
    W_q: torch.Tensor, A_q: Optional[torch.Tensor], B_q: Optional[torch.Tensor], s_q: float,
    W_k: torch.Tensor, A_k: Optional[torch.Tensor], B_k: Optional[torch.Tensor], s_k: float,
    W_v: torch.Tensor, A_v: Optional[torch.Tensor], B_v: Optional[torch.Tensor], s_v: float,
    W_packed_q: Optional[torch.Tensor] = None,
    W_packed_k: Optional[torch.Tensor] = None,
    W_packed_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full QKV forward: 3 packed cuBLAS + 3 Triton epilogues = 6 launches.

    X is read 3 times (once per cuBLAS), vs Unsloth's 6 times.
    Supports GQA (W_k/W_v can have different output dims from W_q).

    Args:
        X: [M, H] or [B, S, H]
        W_q/W_k/W_v: frozen weights
        A_q/A_k/A_v: LoRA A matrices (or None)
        B_q/B_k/B_v: LoRA B matrices (or None)
        s_q/s_k/s_v: LoRA scaling factors
        W_packed_q/k/v: pre-packed cat([W, A], dim=0) for hot loop (optional)
    """
    Q = fused_lora_matmul_v2(X, W_q, A_q, B_q, s_q, W_packed_q)
    K = fused_lora_matmul_v2(X, W_k, A_k, B_k, s_k, W_packed_k)
    V = fused_lora_matmul_v2(X, W_v, A_v, B_v, s_v, W_packed_v)
    return Q, K, V


def pack_weights(W: torch.Tensor, A: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Pre-pack W and A for the hot training loop."""
    if A is None:
        return None
    return torch.cat([W, A], dim=0).contiguous()

"""
v2_2 — In-Place Epilogue + Fused 3-in-1 QKV Epilogue

Two improvements over v2:

1. **In-place epilogue**: Instead of allocating a new output tensor Y, the Triton
   epilogue writes the LoRA result directly into the packed output's first N columns.
   It reads columns [N:N+r] (XA), computes XA @ B^T, and adds in-place to columns
   [:N] (base_out). No extra allocation needed — saves ~32 MB at LLaMA-8B scale.

2. **Fused 3 epilogues into 1**: Instead of 3 separate Triton kernel launches (one
   per Q, K, V), a single kernel processes all 3 projections. Since Q has a different
   N than K/V (GQA), the kernel iterates over 3 groups with different pointers and
   N values. This reduces Triton launches from 3 to 1.

Pipeline for full QKV:
  1. cuBLAS: Q_packed = X @ [W_q; A_q]^T   → [M, H_q+r]   (X read once)
  2. cuBLAS: K_packed = X @ [W_k; A_k]^T   → [M, H_kv+r]  (X read once)
  3. cuBLAS: V_packed = X @ [W_v; A_v]^T   → [M, H_kv+r]  (X read once)
  4. Triton: fused epilogue for Q, K, V     → in-place      (1 launch)

Total: 3 cuBLAS + 1 Triton = 4 launches. X read 3 times.
vs v2:      3 cuBLAS + 3 Triton = 6 launches.
vs Unsloth: 9 cuBLAS launches, X read 6 times.

Expected: Same speed as v2 (~1.09x Unsloth), memory drops from 128 MB to ~96 MB.

Known limitations:
  - Still 3 cuBLAS calls (one per projection) — v2_3 packs all into 1
  - Forward only (backward in v3)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# ============================================================
# Triton in-place epilogue: packed_out[:, :N] += s * packed_out[:, N:N+R] @ B^T
# ============================================================

@triton.jit
def _lora_epilogue_inplace_kernel(
    PACKED_ptr,     # [M, N+R] — packed cuBLAS output (base_out | XA)
    B_ptr,          # [N, R] — LoRA B matrix
    lora_scale,
    M, N, R,
    stride_packed_m, stride_packed_n,
    stride_bn, stride_br,
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

    # Load base output tile [BLOCK_M, BLOCK_N] from packed[:, :N]
    base_ptrs = PACKED_ptr + offs_m[:, None] * stride_packed_m + offs_n[None, :] * stride_packed_n
    base_tile = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load XA tile [BLOCK_M, R] from packed[:, N:N+R]
    xa_ptrs = PACKED_ptr + offs_m[:, None] * stride_packed_m + (N + offs_r[None, :]) * stride_packed_n
    xa_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    xa_tile = tl.load(xa_ptrs, mask=xa_mask, other=0.0)

    # Load B tile [R, BLOCK_N] (B stored as [N, R], transposed access)
    b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
    b_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Tiny matmul: XA @ B^T → [BLOCK_M, BLOCK_N]
    lora_out = tl.dot(xa_tile, b_tile).to(tl.float32)

    # In-place add: base_out += s * lora_out
    result = base_tile + lora_scale * lora_out

    # Write back to packed[:, :N] (in-place)
    tl.store(base_ptrs, result.to(PACKED_ptr.dtype.element_ty), mask=mask)


def lora_epilogue_inplace(
    packed_out: torch.Tensor,
    N: int,
    R: int,
    B: torch.Tensor,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """
    In-place LoRA epilogue: packed_out[:, :N] += s * packed_out[:, N:N+R] @ B^T

    Modifies packed_out in place. Returns a view packed_out[:, :N].

    Args:
        packed_out: [M, N+R] packed cuBLAS output
        N: output dimension (number of base output columns)
        R: LoRA rank
        B: [N, R] LoRA B matrix
        lora_scale: scaling factor

    Returns:
        Y: [M, N] view into packed_out (the first N columns, now updated)
    """
    M = packed_out.shape[0]
    BLOCK_R = max(triton.next_power_of_2(R), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _lora_epilogue_inplace_kernel[grid](
        packed_out, B,
        lora_scale,
        M, N, R,
        packed_out.stride(0), packed_out.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return packed_out[:, :N]


# ============================================================
# Fused 3-in-1 Triton epilogue for Q, K, V
# ============================================================

@triton.jit
def _fused_qkv_epilogue_kernel(
    # Q projection
    Q_PACKED_ptr,   # [M, H_q+R]
    B_q_ptr,        # [H_q, R]
    s_q,
    H_q,
    stride_qp_m, stride_qp_n,
    stride_bq_n, stride_bq_r,
    # K projection
    K_PACKED_ptr,   # [M, H_kv+R]
    B_k_ptr,        # [H_kv, R]
    s_k,
    H_kv,
    stride_kp_m, stride_kp_n,
    stride_bk_n, stride_bk_r,
    # V projection
    V_PACKED_ptr,   # [M, H_kv+R]
    B_v_ptr,        # [H_kv, R]
    s_v,
    stride_vp_m, stride_vp_n,
    stride_bv_n, stride_bv_r,
    # Shared
    M, R,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused in-place epilogue for all 3 QKV projections in a single launch.

    The grid covers the LARGEST projection (Q). Tiles that fall outside smaller
    projections (K, V) are skipped via masking.
    """
    pid = tl.program_id(0)
    proj_id = tl.program_id(1)  # 0=Q, 1=K, 2=V

    # Select projection parameters based on proj_id
    if proj_id == 0:
        PACKED_ptr = Q_PACKED_ptr
        B_ptr = B_q_ptr
        N_proj = H_q
        scale = s_q
        stride_p_m = stride_qp_m
        stride_p_n = stride_qp_n
        stride_b_n = stride_bq_n
        stride_b_r = stride_bq_r
    elif proj_id == 1:
        PACKED_ptr = K_PACKED_ptr
        B_ptr = B_k_ptr
        N_proj = H_kv
        scale = s_k
        stride_p_m = stride_kp_m
        stride_p_n = stride_kp_n
        stride_b_n = stride_bk_n
        stride_b_r = stride_bk_r
    else:
        PACKED_ptr = V_PACKED_ptr
        B_ptr = B_v_ptr
        N_proj = H_kv
        scale = s_v
        stride_p_m = stride_vp_m
        stride_p_n = stride_vp_n
        stride_b_n = stride_bv_n
        stride_b_r = stride_bv_r

    num_n_blocks = tl.cdiv(N_proj, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    # Skip tiles outside this projection's range
    if pid_m * BLOCK_M >= M or pid_n * BLOCK_N >= N_proj:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_proj)

    # Load base output tile from packed[:, :N]
    base_ptrs = PACKED_ptr + offs_m[:, None] * stride_p_m + offs_n[None, :] * stride_p_n
    base_tile = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load XA tile from packed[:, N:N+R]
    xa_ptrs = PACKED_ptr + offs_m[:, None] * stride_p_m + (N_proj + offs_r[None, :]) * stride_p_n
    xa_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    xa_tile = tl.load(xa_ptrs, mask=xa_mask, other=0.0)

    # Load B tile [R, BLOCK_N]
    b_ptrs = B_ptr + offs_n[None, :] * stride_b_n + offs_r[:, None] * stride_b_r
    b_mask = (offs_n[None, :] < N_proj) & (offs_r[:, None] < R)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Tiny matmul + in-place add
    lora_out = tl.dot(xa_tile, b_tile).to(tl.float32)
    result = base_tile + scale * lora_out

    tl.store(base_ptrs, result.to(PACKED_ptr.dtype.element_ty), mask=mask)


def fused_qkv_epilogue(
    q_packed: torch.Tensor, B_q: torch.Tensor, s_q: float, H_q: int,
    k_packed: torch.Tensor, B_k: torch.Tensor, s_k: float, H_kv: int,
    v_packed: torch.Tensor, B_v: torch.Tensor, s_v: float,
    R: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused in-place LoRA epilogue for all 3 QKV projections. Single kernel launch.

    Modifies q_packed, k_packed, v_packed in place.

    Returns:
        (Q, K, V) as views into the first N columns of each packed output.
    """
    M = q_packed.shape[0]
    BLOCK_R = max(triton.next_power_of_2(R), 16)
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid dim 0 covers the largest projection (Q); K/V tiles that exceed bounds are skipped
    max_N = max(H_q, H_kv)
    grid_dim0 = triton.cdiv(M, BLOCK_M) * triton.cdiv(max_N, BLOCK_N)

    _fused_qkv_epilogue_kernel[(grid_dim0, 3)](
        q_packed, B_q, s_q, H_q,
        q_packed.stride(0), q_packed.stride(1),
        B_q.stride(0), B_q.stride(1),
        k_packed, B_k, s_k, H_kv,
        k_packed.stride(0), k_packed.stride(1),
        B_k.stride(0), B_k.stride(1),
        v_packed, B_v, s_v,
        v_packed.stride(0), v_packed.stride(1),
        B_v.stride(0), B_v.stride(1),
        M, R,
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return q_packed[:, :H_q], k_packed[:, :H_kv], v_packed[:, :H_kv]


# ============================================================
# Per-projection: packed cuBLAS + in-place Triton epilogue
# ============================================================

def fused_lora_matmul_v2_2(
    X: torch.Tensor,
    W: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    lora_scale: float = 1.0,
    W_packed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Packed W+A cuBLAS matmul + in-place Triton LoRA epilogue.

    Same as v2 but writes LoRA result in-place into the packed output buffer,
    avoiding the extra [M, N] allocation.

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

    if W_packed is None:
        W_packed = torch.cat([W, A], dim=0)

    out_packed = torch.matmul(X, W_packed.t())  # [M, N+R]
    Y = lora_epilogue_inplace(out_packed, N, R, B, lora_scale)

    if len(orig_shape) == 3:
        Y = Y.reshape(orig_shape[0], orig_shape[1], N)
    return Y


# ============================================================
# Full QKV forward with fused epilogue
# ============================================================

def lora_qkv_v2_2(
    X: torch.Tensor,
    W_q: torch.Tensor, A_q: Optional[torch.Tensor], B_q: Optional[torch.Tensor], s_q: float,
    W_k: torch.Tensor, A_k: Optional[torch.Tensor], B_k: Optional[torch.Tensor], s_k: float,
    W_v: torch.Tensor, A_v: Optional[torch.Tensor], B_v: Optional[torch.Tensor], s_v: float,
    W_packed_q: Optional[torch.Tensor] = None,
    W_packed_k: Optional[torch.Tensor] = None,
    W_packed_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full QKV forward: 3 packed cuBLAS + 1 fused Triton epilogue = 4 launches.

    Uses in-place epilogue to avoid extra memory allocation.
    Supports GQA (W_k/W_v can have different output dims from W_q).

    Args:
        X: [M, H] or [B, S, H]
        W_q/W_k/W_v: frozen weights
        A_q/A_k/A_v: LoRA A matrices (or None)
        B_q/B_k/B_v: LoRA B matrices (or None)
        s_q/s_k/s_v: LoRA scaling factors
        W_packed_q/k/v: pre-packed cat([W, A], dim=0) for hot loop (optional)
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    has_lora = A_q is not None and B_q is not None
    if not has_lora:
        Q = torch.matmul(X, W_q.t())
        K = torch.matmul(X, W_k.t())
        V = torch.matmul(X, W_v.t())
        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], -1)
            K = K.view(orig_shape[0], orig_shape[1], -1)
            V = V.view(orig_shape[0], orig_shape[1], -1)
        return Q, K, V

    H_q = W_q.shape[0]
    H_kv = W_k.shape[0]
    R = A_q.shape[0]

    # Pack weights if not pre-packed
    if W_packed_q is None:
        W_packed_q = torch.cat([W_q, A_q], dim=0)
    if W_packed_k is None:
        W_packed_k = torch.cat([W_k, A_k], dim=0)
    if W_packed_v is None:
        W_packed_v = torch.cat([W_v, A_v], dim=0)

    # 3 cuBLAS calls
    q_packed = torch.matmul(X, W_packed_q.t())  # [M, H_q+R]
    k_packed = torch.matmul(X, W_packed_k.t())  # [M, H_kv+R]
    v_packed = torch.matmul(X, W_packed_v.t())  # [M, H_kv+R]

    # 1 fused Triton epilogue (in-place)
    Q, K, V = fused_qkv_epilogue(
        q_packed, B_q, s_q, H_q,
        k_packed, B_k, s_k, H_kv,
        v_packed, B_v, s_v,
        R,
    )

    if len(orig_shape) == 3:
        Q = Q.view(orig_shape[0], orig_shape[1], H_q)
        K = K.view(orig_shape[0], orig_shape[1], H_kv)
        V = V.view(orig_shape[0], orig_shape[1], H_kv)
    return Q, K, V


def pack_weights(W: torch.Tensor, A: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Pre-pack W and A for the hot training loop."""
    if A is None:
        return None
    return torch.cat([W, A], dim=0).contiguous()

"""
v2_3 — Fully Packed QKV: Single cuBLAS Call + Single Triton Epilogue

The key optimization: pack ALL projection weights (W_q, A_q, W_k, A_k, W_v, A_v)
into one tall matrix and do a single cuBLAS call. X is read from HBM only ONCE.

Layout of packed weight matrix W_all (stacked vertically):
  W_all = cat([W_q, A_q, W_k, A_k, W_v, A_v], dim=0)
  Shape: [H_q + r + H_kv + r + H_kv + r, K]
  For LLaMA-3 8B (GQA 32/8, rank=16): [4096+16+1024+16+1024+16, 4096] = [6192, 4096]

Single cuBLAS call:
  out_all = X @ W_all^T    → [M, 6192]

Splitting the output (deterministic offsets):
  Q_base  = out_all[:, :H_q]                              → [M, 4096]
  XA_q    = out_all[:, H_q:H_q+r]                         → [M, 16]
  K_base  = out_all[:, H_q+r:H_q+r+H_kv]                 → [M, 1024]
  XA_k    = out_all[:, H_q+r+H_kv:H_q+r+H_kv+r]          → [M, 16]
  V_base  = out_all[:, H_q+2*r+H_kv:H_q+2*r+2*H_kv]      → [M, 1024]
  XA_v    = out_all[:, H_q+2*r+2*H_kv:H_q+2*r+2*H_kv+r]  → [M, 16]

Then a single fused Triton epilogue applies LoRA to all 3 projections in-place.

Pipeline: 1 cuBLAS + 1 Triton = **2 launches total**, X read ONCE.
vs v2_2:   3 cuBLAS + 1 Triton = 4 launches, X read 3 times.
vs Unsloth: 9 cuBLAS launches, X read 6 times.

Expected: ~1.15-1.25x Unsloth.

Known limitations:
  - cuBLAS may pick a different algorithm for the wider output [M, 6192] vs [M, 4096]
  - Requires pre-packing all 6 weight matrices (one-time cost)
  - Forward only (backward in v3)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# ============================================================
# Weight packing
# ============================================================

def pack_weights_all(
    W_q: torch.Tensor, A_q: Optional[torch.Tensor],
    W_k: torch.Tensor, A_k: Optional[torch.Tensor],
    W_v: torch.Tensor, A_v: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Pack all QKV weights into a single matrix for one cuBLAS call.

    Layout: [W_q, A_q, W_k, A_k, W_v, A_v] stacked vertically.
    This layout interleaves base weight and LoRA A so that the split
    after the matmul gives contiguous base_out and XA slices.

    Args:
        W_q: [H_q, K], A_q: [r, K] or None
        W_k: [H_kv, K], A_k: [r, K] or None
        W_v: [H_kv, K], A_v: [r, K] or None

    Returns:
        W_all: [H_q + r + H_kv + r + H_kv + r, K] (or without LoRA rows if None)
    """
    parts = [W_q]
    if A_q is not None:
        parts.append(A_q)
    parts.append(W_k)
    if A_k is not None:
        parts.append(A_k)
    parts.append(W_v)
    if A_v is not None:
        parts.append(A_v)
    return torch.cat(parts, dim=0).contiguous()


# ============================================================
# Fused Triton epilogue operating on slices of the packed output
# ============================================================

@triton.jit
def _fused_qkv_epilogue_packed_kernel(
    # Full packed output
    OUT_ALL_ptr,    # [M, total_cols]
    # B matrices
    B_q_ptr,        # [H_q, R]
    B_k_ptr,        # [H_kv, R]
    B_v_ptr,        # [H_kv, R]
    # Scales
    s_q, s_k, s_v,
    # Dimensions
    M, H_q, H_kv, R,
    # Offsets into out_all for each component
    off_q_base,     # 0
    off_xa_q,       # H_q
    off_k_base,     # H_q + R
    off_xa_k,       # H_q + R + H_kv
    off_v_base,     # H_q + 2*R + H_kv
    off_xa_v,       # H_q + 2*R + 2*H_kv
    # Strides
    stride_out_m, stride_out_n,
    stride_bq_n, stride_bq_r,
    stride_bk_n, stride_bk_r,
    stride_bv_n, stride_bv_r,
    # Constexprs
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    proj_id = tl.program_id(1)  # 0=Q, 1=K, 2=V

    if proj_id == 0:
        N_proj = H_q
        base_offset = off_q_base
        xa_offset = off_xa_q
        B_ptr = B_q_ptr
        scale = s_q
        stride_b_n = stride_bq_n
        stride_b_r = stride_bq_r
    elif proj_id == 1:
        N_proj = H_kv
        base_offset = off_k_base
        xa_offset = off_xa_k
        B_ptr = B_k_ptr
        scale = s_k
        stride_b_n = stride_bk_n
        stride_b_r = stride_bk_r
    else:
        N_proj = H_kv
        base_offset = off_v_base
        xa_offset = off_xa_v
        B_ptr = B_v_ptr
        scale = s_v
        stride_b_n = stride_bv_n
        stride_b_r = stride_bv_r

    num_n_blocks = tl.cdiv(N_proj, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    if pid_m * BLOCK_M >= M or pid_n * BLOCK_N >= N_proj:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_proj)

    # Load base output tile
    base_ptrs = OUT_ALL_ptr + offs_m[:, None] * stride_out_m + (base_offset + offs_n[None, :]) * stride_out_n
    base_tile = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load XA tile
    xa_ptrs = OUT_ALL_ptr + offs_m[:, None] * stride_out_m + (xa_offset + offs_r[None, :]) * stride_out_n
    xa_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    xa_tile = tl.load(xa_ptrs, mask=xa_mask, other=0.0)

    # Load B tile
    b_ptrs = B_ptr + offs_n[None, :] * stride_b_n + offs_r[:, None] * stride_b_r
    b_mask = (offs_n[None, :] < N_proj) & (offs_r[:, None] < R)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Tiny matmul + in-place add
    lora_out = tl.dot(xa_tile, b_tile).to(tl.float32)
    result = base_tile + scale * lora_out

    tl.store(base_ptrs, result.to(OUT_ALL_ptr.dtype.element_ty), mask=mask)


# ============================================================
# Full QKV forward: 1 cuBLAS + 1 Triton = 2 launches
# ============================================================

def lora_qkv_v2_3(
    X: torch.Tensor,
    W_q: torch.Tensor, A_q: Optional[torch.Tensor], B_q: Optional[torch.Tensor], s_q: float,
    W_k: torch.Tensor, A_k: Optional[torch.Tensor], B_k: Optional[torch.Tensor], s_k: float,
    W_v: torch.Tensor, A_v: Optional[torch.Tensor], B_v: Optional[torch.Tensor], s_v: float,
    W_all: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully packed QKV forward: 1 cuBLAS + 1 Triton = 2 launches total.

    X is read from HBM only ONCE. All 6 weight matrices are packed into one
    tall matrix for a single cuBLAS call. The Triton epilogue applies LoRA
    to all 3 projections in-place.

    Args:
        X: [M, K] or [B, S, K]
        W_q/W_k/W_v: frozen weights
        A_q/A_k/A_v: LoRA A matrices (or None)
        B_q/B_k/B_v: LoRA B matrices (or None)
        s_q/s_k/s_v: LoRA scaling factors
        W_all: pre-packed weight matrix (optional, for hot loop)
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    M, K = X.shape
    H_q = W_q.shape[0]
    H_kv = W_k.shape[0]
    has_lora = A_q is not None and B_q is not None

    if not has_lora:
        Q = torch.matmul(X, W_q.t())
        K = torch.matmul(X, W_k.t())
        V = torch.matmul(X, W_v.t())
        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], H_q)
            K = K.view(orig_shape[0], orig_shape[1], H_kv)
            V = V.view(orig_shape[0], orig_shape[1], H_kv)
        return Q, K, V

    R = A_q.shape[0]

    # Pack all weights if not pre-packed
    if W_all is None:
        W_all = pack_weights_all(W_q, A_q, W_k, A_k, W_v, A_v)

    # === 1 cuBLAS call: X read ONCE ===
    out_all = torch.matmul(X, W_all.t())  # [M, H_q+R+H_kv+R+H_kv+R]

    # Compute offsets
    off_q_base = 0
    off_xa_q = H_q
    off_k_base = H_q + R
    off_xa_k = H_q + R + H_kv
    off_v_base = H_q + 2 * R + H_kv
    off_xa_v = H_q + 2 * R + 2 * H_kv

    # === 1 Triton epilogue: in-place LoRA for all 3 projections ===
    BLOCK_R = max(triton.next_power_of_2(R), 16)
    BLOCK_M = 64
    BLOCK_N = 64

    max_N = max(H_q, H_kv)
    grid_dim0 = triton.cdiv(M, BLOCK_M) * triton.cdiv(max_N, BLOCK_N)

    _fused_qkv_epilogue_packed_kernel[(grid_dim0, 3)](
        out_all,
        B_q, B_k, B_v,
        s_q, s_k, s_v,
        M, H_q, H_kv, R,
        off_q_base, off_xa_q, off_k_base, off_xa_k, off_v_base, off_xa_v,
        out_all.stride(0), out_all.stride(1),
        B_q.stride(0), B_q.stride(1),
        B_k.stride(0), B_k.stride(1),
        B_v.stride(0), B_v.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # Extract results as views (non-contiguous but valid — stride[0] = total_cols)
    Q = out_all[:, off_q_base:off_q_base + H_q]
    K = out_all[:, off_k_base:off_k_base + H_kv]
    V = out_all[:, off_v_base:off_v_base + H_kv]

    if len(orig_shape) == 3:
        Q = Q.view(orig_shape[0], orig_shape[1], H_q)
        K = K.view(orig_shape[0], orig_shape[1], H_kv)
        V = V.view(orig_shape[0], orig_shape[1], H_kv)
    return Q, K, V

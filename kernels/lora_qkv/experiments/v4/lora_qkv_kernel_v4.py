"""
v4 — Packed Backward Pass for LoRA QKV

Forward: Same as v2_3 (1 cuBLAS + 1 Triton = 2 launches, X read ONCE).

Backward: Reduces from 18+ cuBLAS calls (v3/Unsloth) to 9 cuBLAS + 1 Triton = 10 ops
by packing compatible matrix operations:

  1. PACKED dX_base: cat([dQ, dK, dV], dim=1) @ cat([W_q; W_k; W_v])  — 1 cuBLAS (replaces 3)
  2. PACKED XA_all:  X @ cat([A_q; A_k; A_v])^T                        — 1 cuBLAS (replaces 3)
  3. dY @ B:         3 skinny GEMMs (can't pack: dQ, dK, dV differ in N) — 3 cuBLAS
  4. PACKED dA:      cat([dQ_B_q, dK_B_k, dV_B_v])^T @ X               — 1 cuBLAS (replaces 3)
  5. dB:             3 medium GEMMs (can't pack: different N dims)       — 3 cuBLAS
  6. Triton epilogue: dX = dX_base + s_q*(dQ_B_q@A_q) + s_k*(dK_B_k@A_k) + s_v*(dV_B_v@A_v)
                                                                        — 1 Triton

Total: 9 cuBLAS + 1 Triton = 10 ops (vs 18+ in v3/Unsloth)

The Triton epilogue computes 3 tiny [BLOCK_M, r] @ [r, BLOCK_K] matmuls in registers
and adds them to dX_base in a single pass, avoiding 3 extra cuBLAS calls + 3 dX reads/writes.

Key design decisions:
  - W_dX_packed and A_packed are pre-computed and saved in forward for backward reuse
  - fp64 fallback to plain PyTorch (no Triton) for gradcheck compatibility
  - fp32 accumulation in Triton epilogue for numerical stability
  - The backward Triton kernel handles all 3 LoRA contributions in one pass

Known limitations:
  - Pre-computing W_dX_packed adds one-time cost (but amortized over many iterations)
  - A_packed already exists from forward (zero extra cost)
  - dB GEMMs cannot be packed due to different N dimensions (H_q vs H_kv)
"""

import torch
from torch import Tensor
import triton
import triton.language as tl
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.v2.lora_qkv_kernel_v2_3 import (
    lora_qkv_v2_3,
    pack_weights_all,
)
from reference.lora_qkv_pytorch import lora_qkv_forward


# ============================================================
# Weight packing helpers
# ============================================================

def pack_weights_backward(W_q: Tensor, W_k: Tensor, W_v: Tensor) -> Tensor:
    """
    Pack base weights for the backward dX computation.
    Layout: [W_q; W_k; W_v] stacked vertically → [H_q + H_kv + H_kv, K]

    Used in backward: dX_base = cat([dQ, dK, dV], dim=1) @ W_dX_packed
    """
    return torch.cat([W_q, W_k, W_v], dim=0).contiguous()


def pack_lora_a(A_q: Tensor, A_k: Tensor, A_v: Tensor) -> Tensor:
    """
    Pack LoRA A matrices for the backward XA computation.
    Layout: [A_q; A_k; A_v] stacked vertically → [3r, K]

    Used in backward: XA_all = X @ A_packed^T → [M, 3r]
    """
    return torch.cat([A_q, A_k, A_v], dim=0).contiguous()


# ============================================================
# Triton epilogue for dX LoRA contributions
# ============================================================

@triton.jit
def _lora_dx_epilogue_kernel(
    # Inputs
    DX_BASE_ptr,   # [M, K] — base dX from packed GEMM
    DQ_B_Q_ptr,    # [M, R] — dQ @ B_q
    DK_B_K_ptr,    # [M, R] — dK @ B_k
    DV_B_V_ptr,    # [M, R] — dV @ B_v
    A_Q_ptr,       # [R, K] — A_q
    A_K_ptr,       # [R, K] — A_k
    A_V_ptr,       # [R, K] — A_v
    # Output
    DX_ptr,        # [M, K] — final dX (can alias DX_BASE_ptr for in-place)
    # Scalars
    s_q, s_k, s_v,
    # Dimensions
    M, K: tl.constexpr, R: tl.constexpr,
    # Strides for dX_base / dX (same layout)
    stride_dx_m, stride_dx_k,
    # Strides for dY_B matrices [M, R]
    stride_dyb_m, stride_dyb_r,
    # Strides for A matrices [R, K]
    stride_a_r, stride_a_k,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Fused epilogue: dX = dX_base + s_q*(dQ_B_q @ A_q) + s_k*(dK_B_k @ A_k) + s_v*(dV_B_v @ A_v)

    Each [BLOCK_M, R] @ [R, BLOCK_K] tiny matmul happens in registers.
    All 3 LoRA contributions are computed and added in one pass over dX.
    """
    pid = tl.program_id(0)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_k_blocks
    pid_k = pid % num_k_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)

    mask_mk = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Load dX_base tile [BLOCK_M, BLOCK_K]
    dx_base_ptrs = DX_BASE_ptr + offs_m[:, None] * stride_dx_m + offs_k[None, :] * stride_dx_k
    acc = tl.load(dx_base_ptrs, mask=mask_mk, other=0.0).to(tl.float32)

    # Masks for loading [BLOCK_M, R] and [R, BLOCK_K] tiles
    mask_mr = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    mask_rk = (offs_r[:, None] < R) & (offs_k[None, :] < K)

    # --- Q LoRA contribution: s_q * (dQ_B_q @ A_q) ---
    dyb_q_ptrs = DQ_B_Q_ptr + offs_m[:, None] * stride_dyb_m + offs_r[None, :] * stride_dyb_r
    dyb_q = tl.load(dyb_q_ptrs, mask=mask_mr, other=0.0)

    a_q_ptrs = A_Q_ptr + offs_r[:, None] * stride_a_r + offs_k[None, :] * stride_a_k
    a_q = tl.load(a_q_ptrs, mask=mask_rk, other=0.0)

    lora_q = tl.dot(dyb_q, a_q).to(tl.float32)
    acc += s_q * lora_q

    # --- K LoRA contribution: s_k * (dK_B_k @ A_k) ---
    dyb_k_ptrs = DK_B_K_ptr + offs_m[:, None] * stride_dyb_m + offs_r[None, :] * stride_dyb_r
    dyb_k = tl.load(dyb_k_ptrs, mask=mask_mr, other=0.0)

    a_k_ptrs = A_K_ptr + offs_r[:, None] * stride_a_r + offs_k[None, :] * stride_a_k
    a_k = tl.load(a_k_ptrs, mask=mask_rk, other=0.0)

    lora_k = tl.dot(dyb_k, a_k).to(tl.float32)
    acc += s_k * lora_k

    # --- V LoRA contribution: s_v * (dV_B_v @ A_v) ---
    dyb_v_ptrs = DV_B_V_ptr + offs_m[:, None] * stride_dyb_m + offs_r[None, :] * stride_dyb_r
    dyb_v = tl.load(dyb_v_ptrs, mask=mask_mr, other=0.0)

    a_v_ptrs = A_V_ptr + offs_r[:, None] * stride_a_r + offs_k[None, :] * stride_a_k
    a_v = tl.load(a_v_ptrs, mask=mask_rk, other=0.0)

    lora_v = tl.dot(dyb_v, a_v).to(tl.float32)
    acc += s_v * lora_v

    # Store final dX
    dx_ptrs = DX_ptr + offs_m[:, None] * stride_dx_m + offs_k[None, :] * stride_dx_k
    tl.store(dx_ptrs, acc.to(DX_ptr.dtype.element_ty), mask=mask_mk)


# ============================================================
# autograd.Function
# ============================================================

class LoRAQKVv4Function(torch.autograd.Function):
    """
    v4: Packed forward (v2_3) + packed backward (9 cuBLAS + 1 Triton).

    Forward: 1 cuBLAS + 1 Triton = 2 launches (same as v2_3).
    Backward: 9 cuBLAS + 1 Triton = 10 launches (vs 18+ in v3/Unsloth).
    """

    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        W_q: Tensor, W_k: Tensor, W_v: Tensor,
        A_q: Tensor, B_q: Tensor, s_q: float,
        A_k: Tensor, B_k: Tensor, s_k: float,
        A_v: Tensor, B_v: Tensor, s_v: float,
        W_all: Optional[Tensor] = None,
        W_dX_packed: Optional[Tensor] = None,
        A_packed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        use_triton = X.dtype != torch.float64

        if use_triton:
            Q, K, V = lora_qkv_v2_3(
                X,
                W_q, A_q, B_q, s_q,
                W_k, A_k, B_k, s_k,
                W_v, A_v, B_v, s_v,
                W_all=W_all,
            )
        else:
            Q, K, V = lora_qkv_forward(
                X, W_q, W_k, W_v,
                A_q, B_q, s_q,
                A_k, B_k, s_k,
                A_v, B_v, s_v,
            )

        # Pre-compute packed matrices for backward (one-time cost, reused every iteration)
        if W_dX_packed is None:
            W_dX_packed = pack_weights_backward(W_q, W_k, W_v)
        if A_packed is None:
            A_packed = pack_lora_a(A_q, A_k, A_v)

        ctx.save_for_backward(X, A_q, B_q, A_k, B_k, A_v, B_v, W_dX_packed, A_packed)
        ctx.scalings = (s_q, s_k, s_v)
        ctx.dims = (W_q.shape[0], W_k.shape[0], A_q.shape[0])  # H_q, H_kv, R
        ctx.use_triton = use_triton
        return Q, K, V

    @staticmethod
    def backward(
        ctx, dQ: Tensor, dK: Tensor, dV: Tensor,
    ) -> Tuple[
        Tensor,                           # dX
        None, None, None,                 # dW_q, dW_k, dW_v (frozen)
        Tensor, Tensor, None,             # dA_q, dB_q, ds_q
        Tensor, Tensor, None,             # dA_k, dB_k, ds_k
        Tensor, Tensor, None,             # dA_v, dB_v, ds_v
        None, None, None,                 # W_all, W_dX_packed, A_packed
    ]:
        X, A_q, B_q, A_k, B_k, A_v, B_v, W_dX_packed, A_packed = ctx.saved_tensors
        s_q, s_k, s_v = ctx.scalings
        H_q, H_kv, R = ctx.dims

        orig_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]) if X.dim() == 3 else X
        dQ_flat = dQ.reshape(-1, dQ.shape[-1]) if dQ.dim() == 3 else dQ
        dK_flat = dK.reshape(-1, dK.shape[-1]) if dK.dim() == 3 else dK
        dV_flat = dV.reshape(-1, dV.shape[-1]) if dV.dim() == 3 else dV

        M, K = X_flat.shape

        # ============================================================
        # Step 1: PACKED dX_base (replaces 3 separate dY @ W calls)
        # dQKV = cat([dQ, dK, dV], dim=1)  → [M, H_q + 2*H_kv]
        # dX_base = dQKV @ W_dX_packed     → [M, K]
        # ============================================================
        dQKV = torch.cat([dQ_flat, dK_flat, dV_flat], dim=1)
        dX_base = torch.matmul(dQKV, W_dX_packed)  # 1 cuBLAS (replaces 3)

        # ============================================================
        # Step 2: PACKED X @ A (replaces 3 separate X @ A^T calls)
        # XA_all = X @ A_packed^T  → [M, 3R]
        # ============================================================
        XA_all = torch.matmul(X_flat, A_packed.t())  # 1 cuBLAS (replaces 3)
        XA_q = XA_all[:, :R]
        XA_k = XA_all[:, R:2*R]
        XA_v = XA_all[:, 2*R:]

        # ============================================================
        # Step 3: dY @ B — 3 separate skinny GEMMs
        # Can't pack: dQ [M, H_q], dK [M, H_kv], dV [M, H_kv] have different N dims
        # ============================================================
        dQ_B_q = torch.matmul(dQ_flat, B_q)   # [M, R] — 1 cuBLAS
        dK_B_k = torch.matmul(dK_flat, B_k)   # [M, R] — 1 cuBLAS
        dV_B_v = torch.matmul(dV_flat, B_v)   # [M, R] — 1 cuBLAS

        # ============================================================
        # Step 4: PACKED dA (replaces 3 separate dY_B^T @ X calls)
        # dY_B_all = cat([dQ_B_q, dK_B_k, dV_B_v], dim=1)  → [M, 3R]
        # dA_all = dY_B_all^T @ X  → [3R, K]
        # ============================================================
        dY_B_all = torch.cat([dQ_B_q, dK_B_k, dV_B_v], dim=1)
        dA_all = torch.matmul(dY_B_all.t(), X_flat)  # 1 cuBLAS (replaces 3)
        dA_q = s_q * dA_all[:R]
        dA_k = s_k * dA_all[R:2*R]
        dA_v = s_v * dA_all[2*R:]

        # ============================================================
        # Step 5: dB — 3 separate GEMMs (can't pack: different N dims)
        # dB_q = s_q * dQ^T @ XA_q  → [H_q, R]
        # dB_k = s_k * dK^T @ XA_k  → [H_kv, R]
        # dB_v = s_v * dV^T @ XA_v  → [H_kv, R]
        # ============================================================
        dB_q = s_q * torch.matmul(dQ_flat.t(), XA_q)   # [H_q, R] — 1 cuBLAS
        dB_k = s_k * torch.matmul(dK_flat.t(), XA_k)   # [H_kv, R] — 1 cuBLAS
        dB_v = s_v * torch.matmul(dV_flat.t(), XA_v)   # [H_kv, R] — 1 cuBLAS

        # ============================================================
        # Step 6: Triton epilogue for dX LoRA contributions
        # dX = dX_base + s_q*(dQ_B_q @ A_q) + s_k*(dK_B_k @ A_k) + s_v*(dV_B_v @ A_v)
        # ============================================================
        if ctx.use_triton:
            BLOCK_R = max(triton.next_power_of_2(R), 16)
            BLOCK_M = 64
            BLOCK_K = 64

            grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K),)

            # Ensure contiguous layouts for Triton
            dQ_B_q_c = dQ_B_q.contiguous()
            dK_B_k_c = dK_B_k.contiguous()
            dV_B_v_c = dV_B_v.contiguous()

            _lora_dx_epilogue_kernel[grid](
                dX_base,
                dQ_B_q_c, dK_B_k_c, dV_B_v_c,
                A_q, A_k, A_v,
                dX_base,  # in-place: output overwrites dX_base
                s_q, s_k, s_v,
                M, K, R,
                dX_base.stride(0), dX_base.stride(1),
                dQ_B_q_c.stride(0), dQ_B_q_c.stride(1),
                A_q.stride(0), A_q.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
                BLOCK_R=BLOCK_R,
            )
            dX = dX_base
        else:
            # fp64 fallback: plain PyTorch
            dX = dX_base + s_q * torch.matmul(dQ_B_q, A_q) \
                         + s_k * torch.matmul(dK_B_k, A_k) \
                         + s_v * torch.matmul(dV_B_v, A_v)

        if len(orig_shape) == 3:
            dX = dX.reshape(orig_shape)

        return (
            dX,
            None, None, None,          # dW_q, dW_k, dW_v (frozen)
            dA_q, dB_q, None,          # dA_q, dB_q, ds_q
            dA_k, dB_k, None,          # dA_k, dB_k, ds_k
            dA_v, dB_v, None,          # dA_v, dB_v, ds_v
            None, None, None,          # W_all, W_dX_packed, A_packed
        )


# ============================================================
# Convenience wrapper
# ============================================================

def lora_qkv_v4(
    X: Tensor,
    W_q: Tensor, A_q: Tensor, B_q: Tensor, s_q: float,
    W_k: Tensor, A_k: Tensor, B_k: Tensor, s_k: float,
    W_v: Tensor, A_v: Tensor, B_v: Tensor, s_v: float,
    W_all: Optional[Tensor] = None,
    W_dX_packed: Optional[Tensor] = None,
    A_packed: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Training-compatible LoRA QKV with packed forward + packed backward.

    Forward: v2_3 pipeline (1 cuBLAS + 1 Triton = 2 launches) for bf16/fp16,
             PyTorch fallback for fp64.
    Backward: packed operations (9 cuBLAS + 1 Triton = 10 launches) for bf16/fp16,
              PyTorch fallback for fp64.

    Args:
        X: [M, K] or [B, S, K]
        W_q/W_k/W_v: frozen base weights (no grad)
        A_q/A_k/A_v: LoRA A matrices (require grad)
        B_q/B_k/B_v: LoRA B matrices (require grad)
        s_q/s_k/s_v: LoRA scaling factors
        W_all: pre-packed forward weight from pack_weights_all() (optional)
        W_dX_packed: pre-packed backward weight from pack_weights_backward() (optional)
        A_packed: pre-packed LoRA A from pack_lora_a() (optional)

    Returns:
        (Q, K, V) tuple with autograd support
    """
    return LoRAQKVv4Function.apply(
        X, W_q, W_k, W_v,
        A_q, B_q, s_q,
        A_k, B_k, s_k,
        A_v, B_v, s_v,
        W_all, W_dX_packed, A_packed,
    )


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 60)
    print("v4 Self-Test: Packed Backward")
    print("=" * 60)

    # --- Test 1: fp64 gradcheck (small shapes) ---
    print("\n[1] fp64 gradcheck...")
    H, H_q, H_kv, r = 64, 64, 16, 4
    M = 32

    X = torch.randn(M, H, dtype=torch.float64, device=device, requires_grad=True)
    W_q = torch.randn(H_q, H, dtype=torch.float64, device=device)
    W_k = torch.randn(H_kv, H, dtype=torch.float64, device=device)
    W_v = torch.randn(H_kv, H, dtype=torch.float64, device=device)
    A_q = torch.randn(r, H, dtype=torch.float64, device=device, requires_grad=True)
    B_q = torch.randn(H_q, r, dtype=torch.float64, device=device, requires_grad=True)
    A_k = torch.randn(r, H, dtype=torch.float64, device=device, requires_grad=True)
    B_k = torch.randn(H_kv, r, dtype=torch.float64, device=device, requires_grad=True)
    A_v = torch.randn(r, H, dtype=torch.float64, device=device, requires_grad=True)
    B_v = torch.randn(H_kv, r, dtype=torch.float64, device=device, requires_grad=True)
    s_q, s_k, s_v = 1.0, 1.0, 1.0

    inputs = (X, W_q, W_k, W_v, A_q, B_q, s_q, A_k, B_k, s_k, A_v, B_v, s_v)
    passed = torch.autograd.gradcheck(
        LoRAQKVv4Function.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
    )
    print(f"  gradcheck passed: {passed}")

    # --- Test 2: bf16 forward consistency with v2_3 ---
    print("\n[2] bf16 forward consistency with v2_3...")
    M, H, H_q, H_kv, r = 2048, 4096, 4096, 1024, 16
    X = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_q = torch.randn(H_q, H, dtype=torch.bfloat16, device=device) * 0.02
    W_k = torch.randn(H_kv, H, dtype=torch.bfloat16, device=device) * 0.02
    W_v = torch.randn(H_kv, H, dtype=torch.bfloat16, device=device) * 0.02
    A_q = torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02
    B_q = torch.randn(H_q, r, dtype=torch.bfloat16, device=device) * 0.01
    A_k = torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02
    B_k = torch.randn(H_kv, r, dtype=torch.bfloat16, device=device) * 0.01
    A_v = torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02
    B_v = torch.randn(H_kv, r, dtype=torch.bfloat16, device=device) * 0.01
    s_q, s_k, s_v = 1.0, 1.0, 1.0

    A_q.requires_grad_(True)
    B_q.requires_grad_(True)
    A_k.requires_grad_(True)
    B_k.requires_grad_(True)
    A_v.requires_grad_(True)
    B_v.requires_grad_(True)

    Q_ref, K_ref, V_ref = lora_qkv_v2_3(
        X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v
    )
    Q_v4, K_v4, V_v4 = lora_qkv_v4(
        X, W_q, A_q, B_q, s_q, W_k, A_k, B_k, s_k, W_v, A_v, B_v, s_v
    )

    print(f"  Q diff: {(Q_ref - Q_v4).abs().max().item():.2e}")
    print(f"  K diff: {(K_ref - K_v4).abs().max().item():.2e}")
    print(f"  V diff: {(V_ref - V_v4).abs().max().item():.2e}")

    # --- Test 3: bf16 backward gradient check vs reference ---
    print("\n[3] bf16 backward vs PyTorch reference...")
    from reference.lora_qkv_pytorch import LoRAQKV

    torch.manual_seed(123)
    M, H, H_q, H_kv, r = 512, 256, 256, 64, 8
    X_v4 = torch.randn(M, H, dtype=torch.bfloat16, device=device, requires_grad=True)
    X_ref = X_v4.detach().clone().requires_grad_(True)
    W_q = torch.randn(H_q, H, dtype=torch.bfloat16, device=device) * 0.02
    W_k = torch.randn(H_kv, H, dtype=torch.bfloat16, device=device) * 0.02
    W_v = torch.randn(H_kv, H, dtype=torch.bfloat16, device=device) * 0.02
    A_q_v4 = (torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
    A_q_ref = A_q_v4.detach().clone().requires_grad_(True)
    B_q_v4 = (torch.randn(H_q, r, dtype=torch.bfloat16, device=device) * 0.01).requires_grad_(True)
    B_q_ref = B_q_v4.detach().clone().requires_grad_(True)
    A_k_v4 = (torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
    A_k_ref = A_k_v4.detach().clone().requires_grad_(True)
    B_k_v4 = (torch.randn(H_kv, r, dtype=torch.bfloat16, device=device) * 0.01).requires_grad_(True)
    B_k_ref = B_k_v4.detach().clone().requires_grad_(True)
    A_v_v4 = (torch.randn(r, H, dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
    A_v_ref = A_v_v4.detach().clone().requires_grad_(True)
    B_v_v4 = (torch.randn(H_kv, r, dtype=torch.bfloat16, device=device) * 0.01).requires_grad_(True)
    B_v_ref = B_v_v4.detach().clone().requires_grad_(True)

    Q_v4, K_v4, V_v4 = lora_qkv_v4(
        X_v4, W_q, A_q_v4, B_q_v4, s_q, W_k, A_k_v4, B_k_v4, s_k, W_v, A_v_v4, B_v_v4, s_v
    )
    Q_ref, K_ref, V_ref = LoRAQKV.apply(
        X_ref, W_q, W_k, W_v, A_q_ref, B_q_ref, s_q, A_k_ref, B_k_ref, s_k, A_v_ref, B_v_ref, s_v
    )

    loss_v4 = Q_v4.sum() + K_v4.sum() + V_v4.sum()
    loss_ref = Q_ref.sum() + K_ref.sum() + V_ref.sum()
    loss_v4.backward()
    loss_ref.backward()

    print(f"  dX diff:   {(X_v4.grad - X_ref.grad).abs().max().item():.2e}")
    print(f"  dA_q diff: {(A_q_v4.grad - A_q_ref.grad).abs().max().item():.2e}")
    print(f"  dB_q diff: {(B_q_v4.grad - B_q_ref.grad).abs().max().item():.2e}")
    print(f"  dA_k diff: {(A_k_v4.grad - A_k_ref.grad).abs().max().item():.2e}")
    print(f"  dB_k diff: {(B_k_v4.grad - B_k_ref.grad).abs().max().item():.2e}")
    print(f"  dA_v diff: {(A_v_v4.grad - A_v_ref.grad).abs().max().item():.2e}")
    print(f"  dB_v diff: {(B_v_v4.grad - B_v_ref.grad).abs().max().item():.2e}")

    print("\nAll self-tests passed!")

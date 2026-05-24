"""
v3 — Training-Compatible LoRA QKV with autograd.Function

Wraps the v2_3 forward kernel (fully packed single cuBLAS + Triton epilogue)
in a torch.autograd.Function with a custom backward pass.

Forward: v2_3 pipeline (1 cuBLAS + 1 Triton = 2 launches, X read ONCE).
  For fp64 (gradcheck), falls back to plain PyTorch (no Triton).

Backward: Uses cuBLAS addmm_ following Unsloth's proven pattern:
  For each projection (Y = X @ W^T + s * (X @ A^T) @ B^T):
    dX_proj = dY @ W + s * (dY @ B) @ A
    dA = s * (dY @ B)^T @ X       → [r, K]
    dB = s * dY^T @ (X @ A^T)     → [N, r]

  dX is accumulated across all 3 projections via addmm_:
    dX  = dQ @ W_q
    dX += s_q * (dQ @ B_q) @ A_q    (addmm_)
    dX += dK @ W_k                   (addmm_)
    dX += s_k * (dK @ B_k) @ A_k    (addmm_)
    dX += dV @ W_v                   (addmm_)
    dX += s_v * (dV @ B_v) @ A_v    (addmm_)

  Base weights (W_q, W_k, W_v) are frozen — no gradients computed.
  LoRA matrices (A_q, B_q, A_k, B_k, A_v, B_v) receive gradients.

Known limitations:
  - Backward uses separate cuBLAS calls (not fused) — same as Unsloth
  - Forward fallback to PyTorch for fp64 adds overhead for gradcheck
  - Saving X for backward increases memory (standard tradeoff)
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
# autograd.Function
# ============================================================

class LoRAQKVFunction(torch.autograd.Function):
    """
    Custom autograd.Function for LoRA QKV with fused forward and custom backward.

    Forward uses v2_3 (single cuBLAS + Triton epilogue) for bf16/fp16,
    and plain PyTorch for fp64 (gradcheck compatibility — Triton doesn't support fp64).

    Backward follows Unsloth's pattern: addmm_ for efficient gradient accumulation.
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

        ctx.save_for_backward(X, W_q, W_k, W_v, A_q, B_q, A_k, B_k, A_v, B_v)
        ctx.scalings = (s_q, s_k, s_v)
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
        None,                             # W_all
    ]:
        X, W_q, W_k, W_v, A_q, B_q, A_k, B_k, A_v, B_v = ctx.saved_tensors
        s_q, s_k, s_v = ctx.scalings

        orig_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]) if X.dim() == 3 else X
        dQ_flat = dQ.reshape(-1, dQ.shape[-1]) if dQ.dim() == 3 else dQ
        dK_flat = dK.reshape(-1, dK.shape[-1]) if dK.dim() == 3 else dK
        dV_flat = dV.reshape(-1, dV.shape[-1]) if dV.dim() == 3 else dV

        dtype = X_flat.dtype

        # --- Q LoRA gradients ---
        # dQ_B_q = dQ @ B_q  → [M, r]
        dQ_B_q = torch.matmul(dQ_flat, B_q)
        # dA_q = s_q * dQ_B_q^T @ X  → [r, K]
        dA_q = torch.matmul(dQ_B_q.t(), X_flat)
        dA_q.mul_(s_q)
        # XA_q = X @ A_q^T  → [M, r]
        XA_q = torch.matmul(X_flat, A_q.t())
        # dB_q = s_q * dQ^T @ XA_q  → [H_q, r]
        dB_q = torch.matmul(dQ_flat.t(), XA_q)
        dB_q.mul_(s_q)

        # --- K LoRA gradients ---
        dK_B_k = torch.matmul(dK_flat, B_k)
        dA_k = torch.matmul(dK_B_k.t(), X_flat)
        dA_k.mul_(s_k)
        XA_k = torch.matmul(X_flat, A_k.t())
        dB_k = torch.matmul(dK_flat.t(), XA_k)
        dB_k.mul_(s_k)

        # --- V LoRA gradients ---
        dV_B_v = torch.matmul(dV_flat, B_v)
        dA_v = torch.matmul(dV_B_v.t(), X_flat)
        dA_v.mul_(s_v)
        XA_v = torch.matmul(X_flat, A_v.t())
        dB_v = torch.matmul(dV_flat.t(), XA_v)
        dB_v.mul_(s_v)

        # --- dX: accumulated from all 3 projections ---
        # dX = dQ @ W_q + s_q * (dQ @ B_q) @ A_q
        #    + dK @ W_k + s_k * (dK @ B_k) @ A_k
        #    + dV @ W_v + s_v * (dV @ B_v) @ A_v
        dX = torch.matmul(dQ_flat, W_q)
        dX.addmm_(dQ_B_q, A_q, alpha=s_q)
        dX.addmm_(dK_flat, W_k)
        dX.addmm_(dK_B_k, A_k, alpha=s_k)
        dX.addmm_(dV_flat, W_v)
        dX.addmm_(dV_B_v, A_v, alpha=s_v)

        if len(orig_shape) == 3:
            dX = dX.reshape(orig_shape)

        return (
            dX,
            None, None, None,          # dW_q, dW_k, dW_v (frozen)
            dA_q, dB_q, None,          # dA_q, dB_q, ds_q
            dA_k, dB_k, None,          # dA_k, dB_k, ds_k
            dA_v, dB_v, None,          # dA_v, dB_v, ds_v
            None,                      # W_all
        )


# ============================================================
# Convenience wrapper
# ============================================================

def lora_qkv_v3(
    X: Tensor,
    W_q: Tensor, A_q: Tensor, B_q: Tensor, s_q: float,
    W_k: Tensor, A_k: Tensor, B_k: Tensor, s_k: float,
    W_v: Tensor, A_v: Tensor, B_v: Tensor, s_v: float,
    W_all: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Training-compatible LoRA QKV forward + backward.

    Forward: v2_3 pipeline (1 cuBLAS + 1 Triton = 2 launches) for bf16/fp16,
             PyTorch fallback for fp64.
    Backward: cuBLAS addmm_ chain for gradient computation.

    Args:
        X: [M, K] or [B, S, K]
        W_q/W_k/W_v: frozen base weights (no grad)
        A_q/A_k/A_v: LoRA A matrices (require grad)
        B_q/B_k/B_v: LoRA B matrices (require grad)
        s_q/s_k/s_v: LoRA scaling factors
        W_all: pre-packed weight matrix from pack_weights_all() (optional)

    Returns:
        (Q, K, V) tuple with autograd support
    """
    return LoRAQKVFunction.apply(
        X, W_q, W_k, W_v,
        A_q, B_q, s_q,
        A_k, B_k, s_k,
        A_v, B_v, s_v,
        W_all,
    )

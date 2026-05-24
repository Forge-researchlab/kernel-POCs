"""
Unsloth-style baseline for LoRA QKV projections.

This mirrors Unsloth's approach: separate cuBLAS calls per projection via matmul_lora().
Used as the primary performance baseline — our fused kernel must beat this.

Implementation follows unsloth/kernels/fast_lora.py matmul_lora() pattern:
  1. out = torch.matmul(X, W.t())         # cuBLAS GEMM
  2. XA = torch.matmul(X, A.t())          # cuBLAS skinny GEMM
  3. out.addmm_(XA, B.t(), alpha=s)       # cuBLAS fused add + GEMM

Key differences from our PyTorch reference (lora_qkv_pytorch.py):
  - Uses addmm_ (in-place fused add+matmul) instead of out + s * (XA @ B.t())
  - addmm_ is faster: one cuBLAS call does scalar multiply + matmul + addition
  - No temporary allocation for the LoRA output tensor
  - This is the pattern used in production (Unsloth) — it's the bar we need to clear

Source: unsloth/kernels/utils.py matmul_lora() + fast_lora.py LoRA_QKV
Reference: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py

Forward operation count (QKV):
  - 9 cuBLAS calls: 3 per projection × 3 projections
  - X read from HBM: 6 times (once per W matmul + once per A matmul)
  - LoRA intermediates X@A: 3 writes + 3 reads to HBM (shape [M, r] each)
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


def matmul_lora_unsloth(
    X: Tensor,
    W: Tensor,
    A: Optional[Tensor],
    B: Optional[Tensor],
    s: float = 1.0,
) -> Tensor:
    """
    Single LoRA-augmented projection, Unsloth-style with addmm_.

    Computes: Y = X @ W.t() + s * (X @ A.t()) @ B.t()

    Uses addmm_ for the LoRA path, which fuses the scalar multiply, matmul,
    and addition into a single cuBLAS call. This is strictly faster than the
    naive pattern `out = out + s * (XA @ B.t())` which allocates a temporary
    tensor and requires a separate elementwise add.

    Performance characteristics:
      - 3 cuBLAS calls (base matmul + LoRA down + fused LoRA up+add)
      - X read from HBM twice (once for W, once for A)
      - XA intermediate ([M, r]) written to HBM then read by addmm_

    Args:
        X: input [M, K] or [B, S, K]
        W: frozen base weight [N, K]
        A: LoRA down-projection [r, K] or None
        B: LoRA up-projection [N, r] or None
        s: LoRA scaling factor (alpha / r)

    Returns:
        Y: output [M, N] or [B, S, N]
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W.t())

    if A is not None and B is not None:
        XA = torch.matmul(X, A.t())
        out.addmm_(XA, B.t(), alpha=s)

    if reshape:
        out = out.view(orig_shape[0], orig_shape[1], -1)
    return out


def qkv_lora_unsloth(
    X: Tensor,
    W_q: Tensor, A_q: Optional[Tensor], B_q: Optional[Tensor], s_q: float,
    W_k: Tensor, A_k: Optional[Tensor], B_k: Optional[Tensor], s_k: float,
    W_v: Tensor, A_v: Optional[Tensor], B_v: Optional[Tensor], s_v: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Full QKV projection with LoRA, Unsloth-style.

    Calls matmul_lora_unsloth() independently for each of Q, K, V.
    This is exactly what Unsloth does in production.

    Performance characteristics:
      - 9 cuBLAS calls total (3 per projection)
      - X read from HBM 6 times
      - 3 LoRA intermediates ([M, r]) written/read to/from HBM

    Supports GQA: W_k, W_v can have different output dimensions than W_q.

    Args:
        X:   input [M, H] or [B, S, H]
        W_q: query weight [H_q, H]
        A_q: query LoRA A [r, H] or None
        B_q: query LoRA B [H_q, r] or None
        s_q: query LoRA scale
        W_k: key weight [H_kv, H]
        A_k: key LoRA A [r, H] or None
        B_k: key LoRA B [H_kv, r] or None
        s_k: key LoRA scale
        W_v: value weight [H_kv, H]
        A_v: value LoRA A [r, H] or None
        B_v: value LoRA B [H_kv, r] or None
        s_v: value LoRA scale

    Returns:
        (Q, K, V) tuple
    """
    Q = matmul_lora_unsloth(X, W_q, A_q, B_q, s_q)
    K = matmul_lora_unsloth(X, W_k, A_k, B_k, s_k)
    V = matmul_lora_unsloth(X, W_v, A_v, B_v, s_v)
    return Q, K, V


class LoRAQKVUnsloth(torch.autograd.Function):
    """
    Unsloth-style autograd.Function for LoRA QKV projections.

    Mirrors Unsloth's LoRA_QKV class from fast_lora.py, using addmm_ throughout
    for both forward and backward passes.

    Forward: 9 cuBLAS calls (3 matmul_lora × 3 projections)
    Backward: 12+ cuBLAS calls (dX accumulation + 6 LoRA gradient computations)
    """

    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        W_q: Tensor, A_q: Tensor, B_q: Tensor, s_q: float,
        W_k: Tensor, A_k: Tensor, B_k: Tensor, s_k: float,
        W_v: Tensor, A_v: Tensor, B_v: Tensor, s_v: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        Q = matmul_lora_unsloth(X, W_q, A_q, B_q, s_q)
        K = matmul_lora_unsloth(X, W_k, A_k, B_k, s_k)
        V = matmul_lora_unsloth(X, W_v, A_v, B_v, s_v)

        ctx.save_for_backward(A_q, B_q, A_k, B_k, A_v, B_v, X)
        ctx.scales = (s_q, s_k, s_v)
        ctx.base_weights = (W_q, W_k, W_v)
        return Q, K, V

    @staticmethod
    def backward(
        ctx, dQ: Tensor, dK: Tensor, dV: Tensor
    ) -> Tuple:
        A_q, B_q, A_k, B_k, A_v, B_v, X = ctx.saved_tensors
        s_q, s_k, s_v = ctx.scales
        W_q, W_k, W_v = ctx.base_weights

        orig_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]) if X.dim() == 3 else X
        dQ_flat = dQ.reshape(-1, dQ.shape[-1]) if dQ.dim() == 3 else dQ
        dK_flat = dK.reshape(-1, dK.shape[-1]) if dK.dim() == 3 else dK
        dV_flat = dV.reshape(-1, dV.shape[-1]) if dV.dim() == 3 else dV

        dtype = X_flat.dtype
        A_q_t, B_q_t = A_q.to(dtype).t(), B_q.to(dtype).t()
        A_k_t, B_k_t = A_k.to(dtype).t(), B_k.to(dtype).t()
        A_v_t, B_v_t = A_v.to(dtype).t(), B_v.to(dtype).t()

        d_A_q = torch.empty_like(A_q_t)
        d_B_q = torch.empty_like(B_q_t)
        d_A_k = torch.empty_like(A_k_t)
        d_B_k = torch.empty_like(B_k_t)
        d_A_v = torch.empty_like(A_v_t)
        d_B_v = torch.empty_like(B_v_t)

        # Q LoRA gradients (addmm_ with beta=0 for initialization)
        d_A_q.addmm_(X_flat.t(), dQ_flat @ B_q_t.t(), alpha=s_q, beta=0)
        d_B_q.addmm_(A_q_t.t() @ X_flat.t(), dQ_flat, alpha=s_q, beta=0)

        # K LoRA gradients
        d_A_k.addmm_(X_flat.t(), dK_flat @ B_k_t.t(), alpha=s_k, beta=0)
        d_B_k.addmm_(A_k_t.t() @ X_flat.t(), dK_flat, alpha=s_k, beta=0)

        # V LoRA gradients
        d_A_v.addmm_(X_flat.t(), dV_flat @ B_v_t.t(), alpha=s_v, beta=0)
        d_B_v.addmm_(A_v_t.t() @ X_flat.t(), dV_flat, alpha=s_v, beta=0)

        # dX: accumulated from all three projections using addmm_ chain
        dX = torch.matmul(dQ_flat, W_q)
        dX.addmm_(dQ_flat @ B_q_t.t(), A_q_t.t(), alpha=s_q)
        dX.addmm_(dK_flat, W_k)
        dX.addmm_(dK_flat @ B_k_t.t(), A_k_t.t(), alpha=s_k)
        dX.addmm_(dV_flat, W_v)
        dX.addmm_(dV_flat @ B_v_t.t(), A_v_t.t(), alpha=s_v)

        if len(orig_shape) == 3:
            dX = dX.view(orig_shape)

        return (
            dX,
            None, d_A_q.t(), d_B_q.t(), None,
            None, d_A_k.t(), d_B_k.t(), None,
            None, d_A_v.t(), d_B_v.t(), None,
        )


# ---------------------------------------------------------------------------
# Self-test: verify Unsloth-style matches our PyTorch reference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    H, H_q, H_kv, r = 64, 64, 16, 4
    M = 32

    X = torch.randn(M, H, dtype=dtype, device=device, requires_grad=True)
    W_q = torch.randn(H_q, H, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(H_kv, H, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(H_kv, H, dtype=dtype, device=device) * 0.02
    A_q = torch.randn(r, H, dtype=dtype, device=device, requires_grad=True) * 0.02
    B_q = torch.zeros(H_q, r, dtype=dtype, device=device, requires_grad=True)
    A_k = torch.randn(r, H, dtype=dtype, device=device, requires_grad=True) * 0.02
    B_k = torch.zeros(H_kv, r, dtype=dtype, device=device, requires_grad=True)
    A_v = torch.randn(r, H, dtype=dtype, device=device, requires_grad=True) * 0.02
    B_v = torch.zeros(H_kv, r, dtype=dtype, device=device, requires_grad=True)
    s = 1.0

    # Test single projection
    Y = matmul_lora_unsloth(X, W_q, A_q, B_q, s)
    Y_ref = X @ W_q.t() + s * (X @ A_q.t()) @ B_q.t()
    print(f"Single projection diff: {(Y - Y_ref).abs().max().item():.2e}")

    # Test full QKV
    Q, K, V = qkv_lora_unsloth(X, W_q, A_q, B_q, s, W_k, A_k, B_k, s, W_v, A_v, B_v, s)
    print(f"QKV shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")

    # Test autograd.Function
    Q2, K2, V2 = LoRAQKVUnsloth.apply(
        X, W_q, A_q, B_q, s, W_k, A_k, B_k, s, W_v, A_v, B_v, s
    )
    loss = Q2.sum() + K2.sum() + V2.sum()
    loss.backward()
    print(f"Backward completed. dX shape: {X.grad.shape}")

    # Gradcheck
    X_gc = torch.randn(8, H, dtype=dtype, device=device, requires_grad=True)
    inputs = (
        X_gc,
        W_q, A_q.detach().requires_grad_(True), B_q.detach().requires_grad_(True), s,
        W_k, A_k.detach().requires_grad_(True), B_k.detach().requires_grad_(True), s,
        W_v, A_v.detach().requires_grad_(True), B_v.detach().requires_grad_(True), s,
    )
    passed = torch.autograd.gradcheck(LoRAQKVUnsloth.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"gradcheck passed: {passed}")
    print("All Unsloth baseline checks passed.")

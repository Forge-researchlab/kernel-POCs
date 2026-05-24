"""
PyTorch reference implementation of LoRA QKV projections (LLaMA-style attention).

Mirrors Unsloth's approach of applying matmul_lora() per projection, but uses
only plain PyTorch — no bitsandbytes, no quantization, no Triton. This serves
as the correctness ground truth for testing fused Triton kernels.

Supports GQA (grouped-query attention) where K/V have fewer heads than Q:
  - Q output: [M, H_q] where H_q = num_heads * head_dim
  - K output: [M, H_kv] where H_kv = num_kv_heads * head_dim
  - V output: [M, H_kv]

Three levels of reference are provided:
  1. matmul_lora()      — single projection with LoRA (tests v1)
  2. lora_qkv_forward() — all Q/K/V projections (tests v2)
  3. LoRAQKV            — autograd.Function with forward + backward (tests v3+)

Backward pass math for Y = X @ W^T + s * (X @ A^T) @ B^T:
  dX_proj = dY @ W + s * (dY @ B) @ A
  dA = s * (dY @ B)^T @ X            → [r, K]
  dB = s * dY^T @ (X @ A^T)          → [N, r]
  dW = None (frozen base weights)
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Level 1: Single projection — Y = X @ W^T + s * (X @ A^T) @ B^T
# ---------------------------------------------------------------------------

def matmul_lora(
    X: Tensor,
    W: Tensor,
    A: Optional[Tensor],
    B: Optional[Tensor],
    s: float = 1.0,
) -> Tensor:
    """
    Compute a single LoRA-augmented linear projection.

    Args:
        X: input tensor [M, K] or [B, S, K]
        W: weight matrix [N, K] (stored as in nn.Linear)
        A: LoRA down-projection [r, K] or None
        B: LoRA up-projection [N, r] or None
        s: LoRA scaling factor (alpha / r)

    Returns:
        Y: [M, N] or [B, S, N]
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])

    out = X @ W.t()

    if A is not None and B is not None:
        XA = X @ A.t()
        out = out + s * (XA @ B.t())

    if len(orig_shape) == 3:
        out = out.reshape(orig_shape[0], orig_shape[1], -1)
    return out


# ---------------------------------------------------------------------------
# Level 2: All QKV projections with optional LoRA
# ---------------------------------------------------------------------------

def lora_qkv_forward(
    X: Tensor,
    W_q: Tensor,
    W_k: Tensor,
    W_v: Tensor,
    A_q: Optional[Tensor] = None,
    B_q: Optional[Tensor] = None,
    s_q: float = 1.0,
    A_k: Optional[Tensor] = None,
    B_k: Optional[Tensor] = None,
    s_k: float = 1.0,
    A_v: Optional[Tensor] = None,
    B_v: Optional[Tensor] = None,
    s_v: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute all QKV projections with optional LoRA.

    Supports GQA: W_k/W_v can be [H_kv, H] where H_kv < H_q.

    Args:
        X:   input [M, H] or [B, S, H]
        W_q: query weight [H_q, H]
        W_k: key weight [H_kv, H]
        W_v: value weight [H_kv, H]
        A_q: query LoRA A [r, H] or None
        B_q: query LoRA B [H_q, r] or None
        s_q: query LoRA scale
        A_k: key LoRA A [r, H] or None
        B_k: key LoRA B [H_kv, r] or None
        s_k: key LoRA scale
        A_v: value LoRA A [r, H] or None
        B_v: value LoRA B [H_kv, r] or None
        s_v: value LoRA scale

    Returns:
        (Q, K, V) tuple of tensors
    """
    Q = matmul_lora(X, W_q, A_q, B_q, s_q)
    K = matmul_lora(X, W_k, A_k, B_k, s_k)
    V = matmul_lora(X, W_v, A_v, B_v, s_v)
    return Q, K, V


# ---------------------------------------------------------------------------
# Level 3: autograd.Function with custom backward
# ---------------------------------------------------------------------------

class LoRAQKV(torch.autograd.Function):
    """
    Custom autograd.Function for LoRA QKV with full backward.

    Base weights (W_q, W_k, W_v) are frozen and do not receive gradients.
    LoRA matrices (A_q, B_q, A_k, B_k, A_v, B_v) receive gradients.

    Forward:
        Y_proj = X @ W^T + s * (X @ A^T) @ B^T

    Backward (for each projection, Y = X @ W^T + s * (X @ A^T) @ B^T):
        dX_proj = dY @ W + s * (dY @ B) @ A
        dA = s * (dY @ B)^T @ X
        dB = s * dY^T @ (X @ A^T)
    """

    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        W_q: Tensor, W_k: Tensor, W_v: Tensor,
        A_q: Tensor, B_q: Tensor, s_q: float,
        A_k: Tensor, B_k: Tensor, s_k: float,
        A_v: Tensor, B_v: Tensor, s_v: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        ctx, dQ: Tensor, dK: Tensor, dV: Tensor
    ) -> Tuple[
        Tensor,                            # dX
        None, None, None,                  # dW_q, dW_k, dW_v (frozen)
        Tensor, Tensor, None,              # dA_q, dB_q, ds_q
        Tensor, Tensor, None,              # dA_k, dB_k, ds_k
        Tensor, Tensor, None,              # dA_v, dB_v, ds_v
    ]:
        X, W_q, W_k, W_v, A_q, B_q, A_k, B_k, A_v, B_v = ctx.saved_tensors
        s_q, s_k, s_v = ctx.scalings

        orig_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]) if X.dim() == 3 else X
        dQ_flat = dQ.reshape(-1, dQ.shape[-1]) if dQ.dim() == 3 else dQ
        dK_flat = dK.reshape(-1, dK.shape[-1]) if dK.dim() == 3 else dK
        dV_flat = dV.reshape(-1, dV.shape[-1]) if dV.dim() == 3 else dV

        # --- dX: sum of contributions from all three projections ---
        dX = dQ_flat @ W_q + dK_flat @ W_k + dV_flat @ W_v

        # --- Q LoRA gradients ---
        dA_q = dB_q = None
        if A_q is not None and B_q is not None:
            dQ_B_q = dQ_flat @ B_q       # [M, r]
            dX = dX + s_q * (dQ_B_q @ A_q)
            dA_q = s_q * (dQ_B_q.t() @ X_flat)
            XA_q = X_flat @ A_q.t()       # [M, r]
            dB_q = s_q * (dQ_flat.t() @ XA_q)

        # --- K LoRA gradients ---
        dA_k = dB_k = None
        if A_k is not None and B_k is not None:
            dK_B_k = dK_flat @ B_k
            dX = dX + s_k * (dK_B_k @ A_k)
            dA_k = s_k * (dK_B_k.t() @ X_flat)
            XA_k = X_flat @ A_k.t()
            dB_k = s_k * (dK_flat.t() @ XA_k)

        # --- V LoRA gradients ---
        dA_v = dB_v = None
        if A_v is not None and B_v is not None:
            dV_B_v = dV_flat @ B_v
            dX = dX + s_v * (dV_B_v @ A_v)
            dA_v = s_v * (dV_B_v.t() @ X_flat)
            XA_v = X_flat @ A_v.t()
            dB_v = s_v * (dV_flat.t() @ XA_v)

        if len(orig_shape) == 3:
            dX = dX.reshape(orig_shape)

        return (
            dX,
            None, None, None,          # dW_q, dW_k, dW_v (frozen)
            dA_q, dB_q, None,          # dA_q, dB_q, ds_q
            dA_k, dB_k, None,          # dA_k, dB_k, ds_k
            dA_v, dB_v, None,          # dA_v, dB_v, ds_v
        )


# ---------------------------------------------------------------------------
# Helper: create random LoRA QKV parameters for testing
# ---------------------------------------------------------------------------

def make_lora_qkv_params(
    hidden_dim: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
    rank: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = True,
) -> dict:
    """
    Create a full set of LoRA QKV parameters for testing.

    Args:
        hidden_dim: input dimension (H)
        num_heads: number of query heads
        num_kv_heads: number of key/value heads (< num_heads for GQA)
        head_dim: dimension per head
        rank: LoRA rank
        dtype: tensor dtype
        device: tensor device
        requires_grad: whether LoRA matrices require gradients

    Returns:
        Dict with keys: W_q, W_k, W_v, A_q, B_q, s_q, A_k, B_k, s_k,
                        A_v, B_v, s_v
    """
    H = hidden_dim
    H_q = num_heads * head_dim
    H_kv = num_kv_heads * head_dim
    r = rank
    scale = 1.0

    def make_weight(out_dim, in_dim):
        return torch.randn(out_dim, in_dim, dtype=dtype, device=device) * 0.02

    def make_lora_a(in_dim):
        w = torch.randn(r, in_dim, dtype=dtype, device=device) * 0.02
        if requires_grad:
            w.requires_grad_(True)
        return w

    def make_lora_b(out_dim):
        w = torch.zeros(out_dim, r, dtype=dtype, device=device)
        if requires_grad:
            w.requires_grad_(True)
        return w

    return dict(
        W_q=make_weight(H_q, H),
        W_k=make_weight(H_kv, H),
        W_v=make_weight(H_kv, H),
        A_q=make_lora_a(H),
        B_q=make_lora_b(H_q),
        s_q=scale,
        A_k=make_lora_a(H),
        B_k=make_lora_b(H_kv),
        s_k=scale,
        A_v=make_lora_a(H),
        B_v=make_lora_b(H_kv),
        s_v=scale,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # Small shapes for quick verification
    H, H_q, H_kv, r = 64, 64, 16, 4
    M = 32

    params = make_lora_qkv_params(
        hidden_dim=H, num_heads=H_q // 16, num_kv_heads=H_kv // 16,
        head_dim=16, rank=r, dtype=dtype, device=device, requires_grad=True,
    )
    X = torch.randn(M, H, dtype=dtype, device=device, requires_grad=True)

    # Level 2: functional forward
    Q, K, V = lora_qkv_forward(X, **params)
    print(f"lora_qkv_forward: Q={Q.shape}, K={K.shape}, V={V.shape}")

    # Level 3: autograd.Function forward
    Q2, K2, V2 = LoRAQKV.apply(
        X,
        params["W_q"], params["W_k"], params["W_v"],
        params["A_q"], params["B_q"], params["s_q"],
        params["A_k"], params["B_k"], params["s_k"],
        params["A_v"], params["B_v"], params["s_v"],
    )
    print(f"LoRAQKV.apply:    Q={Q2.shape}, K={K2.shape}, V={V2.shape}")

    # Forward consistency
    diff_q = (Q - Q2).abs().max().item()
    diff_k = (K - K2).abs().max().item()
    diff_v = (V - V2).abs().max().item()
    print(f"Forward consistency: Q diff={diff_q:.2e}, K diff={diff_k:.2e}, V diff={diff_v:.2e}")

    # Backward test
    loss = Q2.sum() + K2.sum() + V2.sum()
    loss.backward()
    print(f"Backward completed. dX shape: {X.grad.shape}")
    print(f"dA_q shape: {params['A_q'].grad.shape}")
    print(f"dB_k shape: {params['B_k'].grad.shape}")

    # Gradcheck
    X_gc = torch.randn(8, H, dtype=dtype, device=device, requires_grad=True)
    inputs = (
        X_gc,
        params["W_q"], params["W_k"], params["W_v"],
        params["A_q"].detach().requires_grad_(True),
        params["B_q"].detach().requires_grad_(True), params["s_q"],
        params["A_k"].detach().requires_grad_(True),
        params["B_k"].detach().requires_grad_(True), params["s_k"],
        params["A_v"].detach().requires_grad_(True),
        params["B_v"].detach().requires_grad_(True), params["s_v"],
    )
    passed = torch.autograd.gradcheck(LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"gradcheck passed: {passed}")
    print("All checks passed.")

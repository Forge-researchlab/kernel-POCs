"""
v2 — Hybrid Gate+Up+SwiGLU Triton Fusion + cuBLAS Down (with backward pass)

Forward:
  - Gate + Up + LoRA + SwiGLU fused into ONE Triton kernel (1 launch)
  - Down projection via cuBLAS (torch.matmul + addmm_)
  - Training mode: kernel also writes e (gate pre-act) and g (up output) for backward
  - Inference mode: only writes h = SiLU(e) * g

Backward:
  - Down projection backward via cuBLAS
  - SwiGLU backward via Unsloth's Triton kernel (in-place, 3 buffers reused)
  - LoRA gradients via cuBLAS addmm_
  - Input gradient dX via cuBLAS

Saved tensors: X, e, g, all LoRA A/B matrices (same as Unsloth).
Base weights W_gate, W_up, W_down are frozen (no gradients).
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from reference.unsloth_baseline import swiglu_DWf_DW_dfg_kernel


# ---------------------------------------------------------------------------
# Triton kernel: fused gate + up + LoRA + SwiGLU (with optional e,g save)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_gate_up_swiglu_kernel(
    X_ptr,
    Wg_ptr, Ag_ptr, Bg_ptr,
    Wu_ptr, Au_ptr, Bu_ptr,
    H_ptr, E_ptr, G_ptr,
    sg, su,
    M, N, K, R,
    stride_xm, stride_xk,
    stride_wgn, stride_wgk,
    stride_wun, stride_wuk,
    stride_agr, stride_agk,
    stride_aur, stride_auk,
    stride_bgn, stride_bgr,
    stride_bun, stride_bur,
    stride_hm, stride_hn,
    stride_em, stride_en,
    stride_gm, stride_gn,
    HAS_LORA: tl.constexpr,
    SAVE_PRE_ACT: tl.constexpr,
    BLOCK_R: tl.constexpr,
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
    offs_r = tl.arange(0, BLOCK_R)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_LORA:
        xa_gate = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        xa_up = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        wg_ptrs = Wg_ptr + offs_n[:, None] * stride_wgn + offs_k[None, :] * stride_wgk
        wg_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        wg_tile = tl.load(wg_ptrs, mask=wg_mask, other=0.0)
        acc_gate = tl.dot(x_tile, tl.trans(wg_tile), acc=acc_gate)

        wu_ptrs = Wu_ptr + offs_n[:, None] * stride_wun + offs_k[None, :] * stride_wuk
        wu_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        wu_tile = tl.load(wu_ptrs, mask=wu_mask, other=0.0)
        acc_up = tl.dot(x_tile, tl.trans(wu_tile), acc=acc_up)

        if HAS_LORA:
            ag_ptrs = Ag_ptr + offs_r[:, None] * stride_agr + offs_k[None, :] * stride_agk
            ag_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K)
            ag_tile = tl.load(ag_ptrs, mask=ag_mask, other=0.0)
            xa_gate = tl.dot(x_tile, tl.trans(ag_tile), acc=xa_gate)

            au_ptrs = Au_ptr + offs_r[:, None] * stride_aur + offs_k[None, :] * stride_auk
            au_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K)
            au_tile = tl.load(au_ptrs, mask=au_mask, other=0.0)
            xa_up = tl.dot(x_tile, tl.trans(au_tile), acc=xa_up)

    if HAS_LORA:
        bg_ptrs = Bg_ptr + offs_n[None, :] * stride_bgn + offs_r[:, None] * stride_bgr
        bg_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        bg_tile = tl.load(bg_ptrs, mask=bg_mask, other=0.0)
        acc_gate += sg * tl.dot(xa_gate.to(bg_tile.dtype), bg_tile)

        bu_ptrs = Bu_ptr + offs_n[None, :] * stride_bun + offs_r[:, None] * stride_bur
        bu_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        bu_tile = tl.load(bu_ptrs, mask=bu_mask, other=0.0)
        acc_up += su * tl.dot(xa_up.to(bu_tile.dtype), bu_tile)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_dtype = H_ptr.dtype.element_ty

    # Save e and g for backward (training mode)
    if SAVE_PRE_ACT:
        e_ptrs = E_ptr + offs_m[:, None] * stride_em + offs_n[None, :] * stride_en
        tl.store(e_ptrs, acc_gate.to(out_dtype), mask=out_mask)
        g_ptrs = G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
        tl.store(g_ptrs, acc_up.to(out_dtype), mask=out_mask)

    # SwiGLU in registers
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    h = silu_gate * acc_up

    h_ptrs = H_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(h_ptrs, h.to(out_dtype), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def fused_gate_up_swiglu(
    X: torch.Tensor,
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    save_pre_act: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused gate + up + LoRA + SwiGLU.

    Returns: (h, e_or_None, g_or_None)
      - h: [M, N] SwiGLU output
      - e: [M, N] gate pre-activation (only if save_pre_act=True)
      - g: [M, N] up projection output (only if save_pre_act=True)
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    M, K = X.shape
    N = W_gate.shape[0]

    has_lora = A_gate is not None and B_gate is not None
    if has_lora:
        R = A_gate.shape[0]
        BLOCK_R = max(triton.next_power_of_2(R), 16)
    else:
        R = 0
        BLOCK_R = 16

    H = torch.empty(M, N, dtype=X.dtype, device=X.device)
    E = torch.empty(M, N, dtype=X.dtype, device=X.device) if save_pre_act else H
    G = torch.empty(M, N, dtype=X.dtype, device=X.device) if save_pre_act else H

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_gate_up_swiglu_kernel[grid](
        X,
        W_gate, A_gate if has_lora else X, B_gate if has_lora else X,
        W_up, A_up if has_lora else X, B_up if has_lora else X,
        H, E, G,
        s_gate, s_up,
        M, N, K, R,
        X.stride(0), X.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        W_up.stride(0), W_up.stride(1),
        A_gate.stride(0) if has_lora else 1, A_gate.stride(1) if has_lora else 1,
        A_up.stride(0) if has_lora else 1, A_up.stride(1) if has_lora else 1,
        B_gate.stride(0) if has_lora else 1, B_gate.stride(1) if has_lora else 1,
        B_up.stride(0) if has_lora else 1, B_up.stride(1) if has_lora else 1,
        H.stride(0), H.stride(1),
        E.stride(0), E.stride(1),
        G.stride(0), G.stride(1),
        HAS_LORA=has_lora,
        SAVE_PRE_ACT=save_pre_act,
        BLOCK_R=BLOCK_R,
    )

    def reshape(t):
        return t.view(orig_shape[0], orig_shape[1], N) if len(orig_shape) == 3 else t

    if save_pre_act:
        return reshape(H), reshape(E), reshape(G)
    return reshape(H), None, None


# ---------------------------------------------------------------------------
# autograd.Function with backward
# ---------------------------------------------------------------------------

class LoRAMLPv2(torch.autograd.Function):
    """
    v2 hybrid LoRA MLP with full backward pass.

    Forward: Triton fused gate+up+SwiGLU (1 launch) + cuBLAS down (2-3 launches)
    Backward: cuBLAS everywhere + Unsloth's Triton SwiGLU backward kernel
    """

    @staticmethod
    def _matmul_lora(X, W, A, B, s):
        """cuBLAS matmul + LoRA (used for down projection and fp64 fallback)."""
        out = X @ W.t()
        if A is not None and B is not None:
            out = out + s * ((X @ A.t()) @ B.t())
        return out

    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W_gate: torch.Tensor, A_gate: torch.Tensor, B_gate: torch.Tensor, s_gate: float,
        W_up: torch.Tensor, A_up: torch.Tensor, B_up: torch.Tensor, s_up: float,
        W_down: torch.Tensor, A_down: torch.Tensor, B_down: torch.Tensor, s_down: float,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        if X.dtype == torch.float64:
            # fp64 fallback (for gradcheck): use PyTorch ops, no Triton
            e = LoRAMLPv2._matmul_lora(X, W_gate, A_gate, B_gate, s_gate)
            g = LoRAMLPv2._matmul_lora(X, W_up, A_up, B_up, s_up)
            h = F.silu(e) * g
            out = LoRAMLPv2._matmul_lora(h, W_down, A_down, B_down, s_down)
        else:
            # Triton fused gate+up+SwiGLU (save e,g for backward)
            h, e, g = fused_gate_up_swiglu(
                X,
                W_gate, A_gate, B_gate, s_gate,
                W_up, A_up, B_up, s_up,
                save_pre_act=True,
            )
            # Down via cuBLAS
            h_flat = h.view(-1, h.shape[-1]) if h.dim() == 3 else h
            out = torch.matmul(h_flat, W_down.t())
            if A_down is not None and B_down is not None:
                XA = torch.matmul(h_flat, A_down.t())
                out.addmm_(XA, B_down.t(), alpha=s_down)
            if X.dim() == 3:
                out = out.view(X.shape[0], X.shape[1], -1)

        ctx.save_for_backward(A_gate, B_gate, A_up, B_up, A_down, B_down, X, e, g)
        ctx.scales = (s_gate, s_up, s_down)
        ctx.base_weights = (W_gate, W_up, W_down)
        return out

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        A_gate, B_gate, A_up, B_up, A_down, B_down, X, e, g = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.scales
        W_gate, W_up, W_down = ctx.base_weights

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X_flat = X.view(-1, X.shape[-1])
        e_flat = e.view(-1, e.shape[-1])
        g_flat = g.view(-1, g.shape[-1])
        dtype = X.dtype

        # Recompute h = SiLU(e) * g (needed for down LoRA grads)
        sig_e = torch.sigmoid(e_flat.float()).to(dtype)
        silu_e = e_flat * sig_e
        h = silu_e * g_flat

        # ── Down projection backward (cuBLAS) ──
        DW = dY @ W_down
        if A_down is not None:
            DW = DW + s_down * ((dY @ B_down) @ A_down)

        # ── Down LoRA grads ──
        d_downA = s_down * ((dY @ B_down).t() @ h)
        d_downB = s_down * (dY.t() @ h @ A_down.t())

        # ── SwiGLU backward ──
        df = DW * silu_e
        dsilu = sig_e * (1.0 + e_flat * (1.0 - sig_e))
        de = DW * g_flat * dsilu

        # ── Up LoRA grads ──
        d_upA = s_up * ((df @ B_up).t() @ X_flat)
        d_upB = s_up * (df.t() @ X_flat @ A_up.t())

        # ── Gate LoRA grads ──
        d_gateA = s_gate * ((de @ B_gate).t() @ X_flat)
        d_gateB = s_gate * (de.t() @ X_flat @ A_gate.t())

        # ── Input gradient dX (cuBLAS) ──
        dX = df @ W_up
        if A_up is not None:
            dX = dX + s_up * ((df @ B_up) @ A_up)
        dX = dX + de @ W_gate
        if A_gate is not None:
            dX = dX + s_gate * ((de @ B_gate) @ A_gate)

        return (
            dX.view(batch, seq_len, hd),
            None, d_gateA, d_gateB, None,
            None, d_upA, d_upB, None,
            None, d_downA, d_downB, None,
        )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def lora_mlp_v2(
    X: torch.Tensor,
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    W_down: torch.Tensor, A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
) -> torch.Tensor:
    """
    Full LoRA MLP forward (v2 hybrid, inference only — no autograd).
    Use LoRAMLPv2.apply() for training with backward.
    """
    h, _, _ = fused_gate_up_swiglu(
        X, W_gate, A_gate, B_gate, s_gate,
        W_up, A_up, B_up, s_up,
        save_pre_act=False,
    )

    h_flat = h.view(-1, h.shape[-1]) if h.dim() == 3 else h
    out = torch.matmul(h_flat, W_down.t())
    if A_down is not None and B_down is not None:
        XA = torch.matmul(h_flat, A_down.t())
        out.addmm_(XA, B_down.t(), alpha=s_down)

    if X.dim() == 3:
        out = out.view(X.shape[0], X.shape[1], -1)
    return out

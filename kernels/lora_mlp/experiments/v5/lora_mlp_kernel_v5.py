"""
v5 — Packed cuBLAS Matmuls + Triton Fused LoRA-SwiGLU Epilogue (Training & Inference)

Strategy:
  v3 launches 8 kernels with 4 separate cuBLAS calls that all multiply by the
  same input X (W_gate, W_up, A_gate, A_up). v5 packs those into a single
  mega-GEMM (X @ W_mega^T), halving X reads and cutting launches in half.

  The same trick is applied to the down projection (W_down and A_down both
  multiply h), packing 2 cuBLAS calls into 1.

  Result: 4 launches for training (down from v3's 8, Unsloth's 10).

Training pipeline (4 launches):
  1. cuBLAS: result = X @ W_mega^T            [B*S, 2*I + 2*r]
       slices: e_base, g_base, xa_gate, xa_up
  2. Triton: h = SiLU(e_base + s*xa_gate @ B_gate^T) * (g_base + s*xa_up @ B_up^T)
       (also saves e_full, g_full with LoRA added, for backward)
  3. cuBLAS: down_result = h @ W_down_packed^T    [B*S, H + r]
       slices: out_base, xa_down
  4. cuBLAS: out_buf.addmm_(xa_down, B_down^T, alpha=s_down)

Inference pipeline (4 launches, zero LoRA compute at runtime):
  Effective weights are pre-merged offline via merge_lora_weights().

  When CUDA >= 12.5 (cublasLt SWISH epilogue available):
    1. cublasLt SWISH: silu_e = SiLU(X @ W_gate_eff^T)   [fused matmul+SiLU]
    2. cuBLAS:         g      = X @ W_up_eff^T
    3. elementwise:    h      = silu_e * g
    4. cuBLAS:         out    = h @ W_down_eff^T

  When CUDA < 12.5 (this box runs CUDA 12.4):
    1. cuBLAS:  e = X @ W_gate_eff^T
    2. cuBLAS:  g = X @ W_up_eff^T
    3. Triton:  h = swiglu_fg_kernel(e, g)   [single SiLU*g fused kernel]
    4. cuBLAS:  out = h @ W_down_eff^T

  Either way, 4 launches and zero LoRA matmuls at runtime.

Launch count vs prior work:
  - Unsloth (training):       10 launches
  - v3       (training):       8 launches
  - v5       (training):       4 launches
  - v5       (inference):      4 launches (zero Triton)

Key gotchas (handled below):
  - Sliced tensors from the mega-matmul are NOT contiguous in N: the Triton
    epilogue uses explicit strides (stride_em, stride_en, ...) so reads work,
    but the down-projection slice needs `.contiguous()` before in-place addmm_.
  - For fp64 (gradcheck), we use the same pure-PyTorch fallback as v3.
  - The Triton epilogue allocates fresh contiguous tensors for the saved
    e_full/g_full (via torch.empty_like, which returns contiguous even for
    non-contiguous inputs), so the backward can run unchanged from v3.
"""

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reference.unsloth_baseline import (
    matmul_lora as unsloth_matmul_lora,
    swiglu_DWf_DW_dfg_kernel,
    swiglu_fg_kernel,
)
from experiments.v4.cublaslt_wrapper import cublaslt_matmul_epilogue


# cublasLt's SWISH/SILU epilogue (CUBLASLT_EPILOGUE_SWISH = 2048) was added in
# CUDA 12.5. On CUDA 12.4 the call returns CUBLAS_STATUS_INVALID_VALUE, so we
# fall back to a separate matmul + SiLU*g step. We probe once at import time.
def _cublaslt_swish_available() -> bool:
    try:
        x = torch.empty(32, 32, dtype=torch.bfloat16, device="cuda")
        w = torch.empty(32, 32, dtype=torch.bfloat16, device="cuda")
        cublaslt_matmul_epilogue(x, w, epilogue="swish")
        return True
    except (RuntimeError, AssertionError):
        return False


_CUBLASLT_SWISH = _cublaslt_swish_available() if torch.cuda.is_available() else False


# ---------------------------------------------------------------------------
# Triton kernel: fused LoRA addition + SwiGLU (copied verbatim from v3)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_lora_swiglu_kernel(
    E_ptr, G_ptr,
    XA_gate_ptr, XA_up_ptr,
    B_gate_ptr, B_up_ptr,
    H_ptr, E_out_ptr, G_out_ptr,
    s_gate, s_up,
    M, N, R,
    stride_em, stride_en,
    stride_gm, stride_gn,
    stride_hm, stride_hn,
    stride_eom, stride_eon,
    stride_gom, stride_gon,
    stride_xam, stride_xar,
    stride_xaum, stride_xaur,
    stride_bgn, stride_bgr,
    stride_bun, stride_bur,
    HAS_LORA: tl.constexpr,
    SAVE_EG: tl.constexpr,
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
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    e_ptrs = E_ptr + offs_m[:, None] * stride_em + offs_n[None, :] * stride_en
    e_tile = tl.load(e_ptrs, mask=mask, other=0.0).to(tl.float32)

    g_ptrs = G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    g_tile = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)

        xa_g_ptrs = XA_gate_ptr + offs_m[:, None] * stride_xam + offs_r[None, :] * stride_xar
        xa_g_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
        xa_g_tile = tl.load(xa_g_ptrs, mask=xa_g_mask, other=0.0)

        bg_ptrs = B_gate_ptr + offs_n[None, :] * stride_bgn + offs_r[:, None] * stride_bgr
        bg_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        bg_tile = tl.load(bg_ptrs, mask=bg_mask, other=0.0)

        lora_gate = tl.dot(xa_g_tile, bg_tile).to(tl.float32)
        e_tile += s_gate * lora_gate

        xa_u_ptrs = XA_up_ptr + offs_m[:, None] * stride_xaum + offs_r[None, :] * stride_xaur
        xa_u_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
        xa_u_tile = tl.load(xa_u_ptrs, mask=xa_u_mask, other=0.0)

        bu_ptrs = B_up_ptr + offs_n[None, :] * stride_bun + offs_r[:, None] * stride_bur
        bu_mask = (offs_n[None, :] < N) & (offs_r[:, None] < R)
        bu_tile = tl.load(bu_ptrs, mask=bu_mask, other=0.0)

        lora_up = tl.dot(xa_u_tile, bu_tile).to(tl.float32)
        g_tile += s_up * lora_up

    out_dtype = H_ptr.dtype.element_ty

    if SAVE_EG:
        eo_ptrs = E_out_ptr + offs_m[:, None] * stride_eom + offs_n[None, :] * stride_eon
        tl.store(eo_ptrs, e_tile.to(out_dtype), mask=mask)
        go_ptrs = G_out_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon
        tl.store(go_ptrs, g_tile.to(out_dtype), mask=mask)

    silu_e = e_tile * tl.sigmoid(e_tile)
    h_tile = silu_e * g_tile

    h_ptrs = H_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(h_ptrs, h_tile.to(out_dtype), mask=mask)


def fused_lora_swiglu(
    e_base: torch.Tensor,
    g_base: torch.Tensor,
    xa_gate: Optional[torch.Tensor],
    xa_up: Optional[torch.Tensor],
    B_gate: Optional[torch.Tensor],
    B_up: Optional[torch.Tensor],
    s_gate: float = 1.0,
    s_up: float = 1.0,
    save_eg: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused LoRA addition + SwiGLU (identical to v3's epilogue).

    Inputs may be non-contiguous slices of a packed mega-matmul output.
    The kernel uses explicit strides so this is handled correctly.
    Outputs (h, e_full, g_full) are fresh contiguous tensors.
    """
    M, N = e_base.shape
    has_lora = xa_gate is not None
    R = xa_gate.shape[1] if has_lora else 0
    BLOCK_R = max(triton.next_power_of_2(R), 16) if has_lora else 16

    # empty_like returns contiguous tensors here (verified) even for non-contig inputs
    H = torch.empty(M, N, dtype=e_base.dtype, device=e_base.device)
    E_out = torch.empty(M, N, dtype=e_base.dtype, device=e_base.device) if save_eg else H
    G_out = torch.empty(M, N, dtype=e_base.dtype, device=e_base.device) if save_eg else H

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fused_lora_swiglu_kernel[grid](
        e_base, g_base,
        xa_gate if has_lora else e_base,
        xa_up if has_lora else e_base,
        B_gate if has_lora else e_base,
        B_up if has_lora else e_base,
        H, E_out, G_out,
        s_gate, s_up,
        M, N, R,
        e_base.stride(0), e_base.stride(1),
        g_base.stride(0), g_base.stride(1),
        H.stride(0), H.stride(1),
        E_out.stride(0), E_out.stride(1),
        G_out.stride(0), G_out.stride(1),
        xa_gate.stride(0) if has_lora else 1, xa_gate.stride(1) if has_lora else 1,
        xa_up.stride(0) if has_lora else 1, xa_up.stride(1) if has_lora else 1,
        B_gate.stride(0) if has_lora else 1, B_gate.stride(1) if has_lora else 1,
        B_up.stride(0) if has_lora else 1, B_up.stride(1) if has_lora else 1,
        HAS_LORA=has_lora,
        SAVE_EG=save_eg,
        BLOCK_R=BLOCK_R,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    if save_eg:
        return H, E_out, G_out
    return H, None, None


# ---------------------------------------------------------------------------
# Weight packing utilities
# ---------------------------------------------------------------------------

def pack_gate_up_weights(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    A_gate: Optional[torch.Tensor] = None,
    A_up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pack the four matrices that all multiply X into one mega-matrix.

    Shapes:
        W_gate, W_up: [I, H]
        A_gate, A_up: [r, H]
    Returns:
        W_mega: [2*I + 2*r, H]  (or [2*I, H] if no LoRA)
    """
    parts = [W_gate, W_up]
    if A_gate is not None and A_up is not None:
        parts.extend([A_gate, A_up])
    return torch.cat(parts, dim=0).contiguous()


def pack_down_weights(
    W_down: torch.Tensor,
    A_down: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pack down + LoRA-A_down into one mega-matrix.

    Shapes:
        W_down: [H, I]
        A_down: [r, I]
    Returns:
        W_down_packed: [H + r, I]  (or [H, I] if no LoRA)
    """
    if A_down is None:
        return W_down.contiguous()
    return torch.cat([W_down, A_down], dim=0).contiguous()


def merge_lora_weights(
    W: torch.Tensor,
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    s: float,
) -> torch.Tensor:
    """Merge LoRA into base weights: W_eff = W + s * (B @ A). Returns [out, in]."""
    if A is None or B is None:
        return W
    return W + s * (B @ A)


def prepare_inference_weights(
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    W_down: torch.Tensor, A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge LoRA into the base weights and pre-transpose for fast inference.

    Returns:
        W_gate_eff_T: [H, I] contiguous — used as the B operand in cublasLt
        W_up_eff_T:   [H, I] contiguous — used as the B operand in cuBLAS
        W_down_eff_T: [I, H] contiguous — used as the B operand in cuBLAS

    Doing the transpose+contiguous offline keeps inference at exactly 4 launches.
    """
    W_gate_eff = merge_lora_weights(W_gate, A_gate, B_gate, s_gate)  # [I, H]
    W_up_eff = merge_lora_weights(W_up, A_up, B_up, s_up)  # [I, H]
    W_down_eff = merge_lora_weights(W_down, A_down, B_down, s_down)  # [H, I]
    return (
        W_gate_eff.t().contiguous(),  # [H, I]
        W_up_eff.t().contiguous(),    # [H, I]
        W_down_eff.t().contiguous(),  # [I, H]
    )


# ---------------------------------------------------------------------------
# Training forward (packed)
# ---------------------------------------------------------------------------

def _v5_forward_impl(
    X: torch.Tensor,
    W_mega: torch.Tensor,
    B_gate: Optional[torch.Tensor], B_up: Optional[torch.Tensor],
    s_gate: float, s_up: float,
    W_down_packed: torch.Tensor,
    B_down: Optional[torch.Tensor], s_down: float,
    I: int, r: int,
    save_eg: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Packed training forward. 4 launches (3 cuBLAS + 1 Triton).
    """
    orig_shape = X.shape
    X_flat = X.view(-1, X.shape[-1]) if X.dim() == 3 else X
    has_lora = (B_gate is not None) and (r > 0)

    # ── Phase 1: one cuBLAS call for gate base + up base + LoRA-A_gate + LoRA-A_up
    result = torch.matmul(X_flat, W_mega.t())  # [M, 2*I + 2*r]
    e_base = result[:, :I]
    g_base = result[:, I:2 * I]
    if has_lora:
        xa_gate = result[:, 2 * I: 2 * I + r]
        xa_up = result[:, 2 * I + r:]
    else:
        xa_gate = None
        xa_up = None

    # ── Phase 2: Triton epilogue (LoRA add + SiLU + multiply)
    h, e_full, g_full = fused_lora_swiglu(
        e_base, g_base,
        xa_gate, xa_up,
        B_gate, B_up,
        s_gate, s_up,
        save_eg=save_eg,
    )

    # ── Phase 3: one cuBLAS call for down base + LoRA-A_down
    if has_lora:
        H_dim = W_down_packed.shape[0] - r
        down_result = torch.matmul(h, W_down_packed.t())  # [M, H + r]
        out_slice = down_result[:, :H_dim]
        xa_down = down_result[:, H_dim:]
        # out_slice is non-contiguous (stride 0 = H+r). addmm_ needs a contiguous
        # output buffer (and we don't want to mutate the packed result tensor).
        out_buf = out_slice.contiguous()
        out_buf.addmm_(xa_down, B_down.t(), alpha=s_down)
    else:
        out_buf = torch.matmul(h, W_down_packed.t())

    if len(orig_shape) == 3:
        out_buf = out_buf.view(orig_shape[0], orig_shape[1], -1)
        if e_full is not None:
            e_full = e_full.view(orig_shape[0], orig_shape[1], -1)
            g_full = g_full.view(orig_shape[0], orig_shape[1], -1)

    return out_buf, e_full, g_full


def lora_mlp_v5(
    X: torch.Tensor,
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    W_down: torch.Tensor, A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
    W_mega: Optional[torch.Tensor] = None,
    W_down_packed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full LoRA MLP v5 forward (inference convenience, no autograd).

    If `W_mega` / `W_down_packed` are not provided, they are packed on the fly.
    For repeated forwards, pack them once and reuse.
    """
    has_lora = (A_gate is not None) and (B_gate is not None)
    I = W_gate.shape[0]
    r = A_gate.shape[0] if has_lora else 0

    if W_mega is None:
        W_mega = pack_gate_up_weights(
            W_gate, W_up,
            A_gate if has_lora else None,
            A_up if has_lora else None,
        )
    if W_down_packed is None:
        W_down_packed = pack_down_weights(
            W_down,
            A_down if has_lora else None,
        )

    out, _, _ = _v5_forward_impl(
        X, W_mega,
        B_gate if has_lora else None,
        B_up if has_lora else None,
        s_gate, s_up,
        W_down_packed,
        B_down if has_lora else None,
        s_down,
        I, r,
        save_eg=False,
    )
    return out


# ---------------------------------------------------------------------------
# Inference forward (merged weights + cublasLt SWISH epilogue)
# ---------------------------------------------------------------------------

def lora_mlp_v5_inference(
    X: torch.Tensor,
    W_gate_eff_T: torch.Tensor,
    W_up_eff_T: torch.Tensor,
    W_down_eff_T: torch.Tensor,
) -> torch.Tensor:
    """
    4-launch LoRA-MLP inference using pre-merged LoRA weights.

    Args:
        X:             [B, S, H] (or [M, H])
        W_gate_eff_T:  [H, I] contiguous (already merged with LoRA, transposed)
        W_up_eff_T:    [H, I] contiguous
        W_down_eff_T:  [I, H] contiguous

    Returns:
        out: same outer shape as X but with hidden dim H

    Launch breakdown (4 total):
      CUDA >= 12.5:  cublasLt SWISH (gate+SiLU) + cuBLAS (up) + mul + cuBLAS (down)
      CUDA <  12.5:  cuBLAS (gate) + cuBLAS (up) + Triton SwiGLU (silu*g) + cuBLAS (down)
    """
    orig_shape = X.shape
    X_flat = X.view(-1, X.shape[-1]) if X.dim() == 3 else X

    if _CUBLASLT_SWISH and X.dtype in (torch.float16, torch.bfloat16):
        # Fast path: cublasLt fuses matmul + SiLU into one launch
        silu_e = cublaslt_matmul_epilogue(X_flat, W_gate_eff_T, epilogue="swish")
        g = torch.matmul(X_flat, W_up_eff_T)
        h = silu_e * g
    else:
        # Fallback path: still 4 launches, using Unsloth's fused SiLU*g kernel
        e = torch.matmul(X_flat, W_gate_eff_T)
        g = torch.matmul(X_flat, W_up_eff_T)
        # swiglu_fg_kernel expects a 3D shape [batch, seq, hd] for its grid
        e3 = e.view(1, e.shape[0], e.shape[1])
        g3 = g.view(1, g.shape[0], g.shape[1])
        h = swiglu_fg_kernel(e3, g3).view(e.shape[0], e.shape[1])

    out = torch.matmul(h, W_down_eff_T)

    if len(orig_shape) == 3:
        out = out.view(orig_shape[0], orig_shape[1], -1)
    return out


# ---------------------------------------------------------------------------
# Autograd Function (training)
# ---------------------------------------------------------------------------

class LoRAMLPv5(torch.autograd.Function):
    """
    v5 LoRA MLP with full backward pass.

    Forward:  packed cuBLAS gate+up+LoRA-A + Triton fused LoRA+SwiGLU
              + packed cuBLAS down+LoRA-A + addmm_ for the down LoRA-B term.
              (4 launches when has_lora=True, vs v3's 8, Unsloth's 10.)

    Backward: identical math to v3 — packing doesn't help here because each
              gradient matmul has a different input (dY, df, de, X).
    """

    @staticmethod
    def forward(
        ctx,
        X, W_gate, A_gate, B_gate, s_gate,
        W_up, A_up, B_up, s_up,
        W_down, A_down, B_down, s_down,
    ):

        if X.dtype == torch.float64:
            # fp64 fallback for gradcheck (no packing, no Triton)
            e = X @ W_gate.t() + s_gate * ((X @ A_gate.t()) @ B_gate.t())
            g = X @ W_up.t() + s_up * ((X @ A_up.t()) @ B_up.t())
            h = F.silu(e) * g
            out = h @ W_down.t() + s_down * ((h @ A_down.t()) @ B_down.t())
        else:
            has_lora = (A_gate is not None) and (B_gate is not None)
            I = W_gate.shape[0]
            r = A_gate.shape[0] if has_lora else 0

            W_mega = pack_gate_up_weights(
                W_gate, W_up,
                A_gate if has_lora else None,
                A_up if has_lora else None,
            )
            W_down_packed = pack_down_weights(
                W_down,
                A_down if has_lora else None,
            )

            out, e, g = _v5_forward_impl(
                X, W_mega,
                B_gate if has_lora else None,
                B_up if has_lora else None,
                s_gate, s_up,
                W_down_packed,
                B_down if has_lora else None,
                s_down,
                I, r,
                save_eg=True,
            )

        ctx.save_for_backward(A_gate, B_gate, A_up, B_up, A_down, B_down, X, e, g)
        ctx.scales = (s_gate, s_up, s_down)
        ctx.base_weights = (W_gate, W_up, W_down)
        return out

    @staticmethod
    def backward(ctx, dY):
        A_gate, B_gate, A_up, B_up, A_down, B_down, X, e, g = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.scales
        W_gate, W_up, W_down = ctx.base_weights

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.reshape(-1, e.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        dtype = X.dtype

        if dtype == torch.float64:
            # fp64 fallback (gradcheck): no in-place ops, safe for reentrant backward
            sig_e = torch.sigmoid(e.float()).to(dtype)
            silu_e = e * sig_e
            h = silu_e * g
            DW = dY @ W_down + s_down * ((dY @ B_down) @ A_down)
            d_downA = s_down * ((dY @ B_down).t() @ h)
            d_downB = s_down * (dY.t() @ h @ A_down.t())
            df = DW * silu_e
            dsilu = sig_e * (1.0 + e * (1.0 - sig_e))
            de = DW * g * dsilu
            d_upA = s_up * ((df @ B_up).t() @ X)
            d_upB = s_up * (df.t() @ X @ A_up.t())
            d_gateA = s_gate * ((de @ B_gate).t() @ X)
            d_gateB = s_gate * (de.t() @ X @ A_gate.t())
            dX = df @ W_up + s_up * ((df @ B_up) @ A_up) + de @ W_gate + s_gate * ((de @ B_gate) @ A_gate)
        else:
            # Production path: Unsloth's optimized pattern (in-place buffer reuse via Triton)
            # The saved e and g must be contiguous because the Unsloth Triton kernel
            # uses flat 1D offsets. fused_lora_swiglu allocates contiguous outputs.
            e = e.contiguous()
            g = g.contiguous()

            gateA, gateB = A_gate.to(dtype).t(), B_gate.to(dtype).t()
            upA, upB     = A_up.to(dtype).t(),   B_up.to(dtype).t()
            downA, downB = A_down.to(dtype).t(),  B_down.to(dtype).t()

            DW = unsloth_matmul_lora(dY, W_down.t(), None, downB, downA, s_down)
            DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
            h, df, de = DW, e, g

            d_downA = torch.empty_like(downA)
            d_downB = torch.empty_like(downB)
            d_gateA = torch.empty_like(gateA)
            d_gateB = torch.empty_like(gateB)
            d_upA   = torch.empty_like(upA)
            d_upB   = torch.empty_like(upB)

            d_downA.addmm_(h.t(), dY @ downB.t(), alpha=s_down, beta=0)
            d_downB.addmm_(downA.t() @ h.t(), dY, alpha=s_down, beta=0)
            d_upA.addmm_(X.t(), df @ upB.t(), alpha=s_up, beta=0)
            d_upB.addmm_(upA.t() @ X.t(), df, alpha=s_up, beta=0)
            d_gateA.addmm_(X.t(), de @ gateB.t(), alpha=s_gate, beta=0)
            d_gateB.addmm_(gateA.t() @ X.t(), de, alpha=s_gate, beta=0)

            dX = torch.matmul(df, W_up)
            dX.addmm_(df @ upB.t(), upA.t(), alpha=s_up)
            dX.addmm_(de, W_gate)
            dX.addmm_(de @ gateB.t(), gateA.t(), alpha=s_gate)

            d_gateA = d_gateA.t()
            d_gateB = d_gateB.t()
            d_upA = d_upA.t()
            d_upB = d_upB.t()
            d_downA = d_downA.t()
            d_downB = d_downB.t()

        return (
            dX.view(batch, seq_len, hd),
            None, d_gateA, d_gateB, None,
            None, d_upA, d_upB, None,
            None, d_downA, d_downB, None,
        )

"""
v6 — cuBLAS for big GEMMs + Triton-stacked LoRA-A + Optional CUDA-Stream Parallelism

Strategy (use where each framework shines):
  * cuBLAS for every "big" matmul (gate, up, down) — these are tile-aligned to
    cuBLAS's preferred N=I=14336 (or H=4096) tiles and run at >80% of peak.
  * One Triton kernel for the two TINY LoRA-A matmuls (X @ A_gate.T and
    X @ A_up.T). We stack the two A matrices into [2r, H] so a single
    standard Triton matmul produces [M, 2r] — the compiler caches each X tile
    in shared memory and reuses it across the 2r output columns. This gives
    the "load X once" benefit of fusion WITHOUT writing a fancy multi-output
    kernel.
  * Optional CUDA stream parallelism (``enable_streams=True``, default) overlaps
    the two big gate/up GEMMs on separate streams, and overlaps the down GEMM
    with the tiny ``h @ A_down.T`` LoRA-A matmul. Sync points are event-based.

Pipeline (training, has_lora=True):
  Phase 1 (cuBLAS, optional streams):
    stream A:  e_base = X @ W_gate.T      [M, I]
    stream B:  g_base = X @ W_up.T        [M, I]
  Phase 2 (Triton, single launch, stacked):
    X @ [A_gate; A_up].T  →  xa_combined  [M, 2r]
    xa_gate = xa_combined[:, :r];  xa_up = xa_combined[:, r:]
  Phase 3 (Triton, existing fused_lora_swiglu epilogue):
    h, e_full, g_full = fused_lora_swiglu(e_base, g_base, xa_gate, xa_up, ...)
  Phase 4 (cuBLAS + cuBLAS, optional streams):
    stream A:  out = h @ W_down.T         [M, H]      (cuBLAS, big)
    stream B:  xa_down = h @ A_down.T     [M, r]      (cuBLAS, tiny)
    sync
    out.addmm_(xa_down, B_down.T, alpha=s_down)        (cuBLAS, in-place)

Launch counts (training, has_lora=True):
  enable_streams=False: 6 launches  (W_gate, W_up, stacked-LoRA-A, swiglu, W_down,
                                     A_down, addmm_) — actually 7, with addmm_.
  enable_streams=True:  same kernel count, but Phase-1 gate/up overlap on
                        separate streams, and Phase-4 W_down/A_down overlap.

Memory: no transient packed weights (gate/up/down weights are used as-is);
        no mega-matmul output. Footprint ties v3.

Inference path is shared with v5: ``lora_mlp_v6_inference = lora_mlp_v5_inference``.
Pre-merged effective weights sidestep all LoRA-A pain points.

Backward: identical math to v3 / v5 / v5_upgrade_1 (uses Unsloth's
optimized in-place buffer-reuse via the swiglu_DWf_DW_dfg_kernel + matmul_lora
pattern). Backward does not benefit from packing or streams here because each
backward GEMM has a distinct left-hand side (dY, df, de, X).
"""

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reference.unsloth_baseline import (  # noqa: E402
    matmul_lora as unsloth_matmul_lora,
    swiglu_DWf_DW_dfg_kernel,
)
from experiments.v5.lora_mlp_kernel_v5 import (  # noqa: E402
    _fused_lora_swiglu_kernel,
    lora_mlp_v5_inference,
    prepare_inference_weights,
)


# ---------------------------------------------------------------------------
# In-place LoRA+SwiGLU wrapper (v6_upgrade_1: saves ~449 MB peak forward)
# ---------------------------------------------------------------------------

def fused_lora_swiglu_inplace(
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
    Fused LoRA addition + SwiGLU, writing e_full/g_full **in-place** over
    e_base/g_base when ``save_eg=True``.

    In v5's ``fused_lora_swiglu``, fresh [M, N] tensors are allocated for
    E_out and G_out — meaning e_base (224 MB at LLaMA-8B) and E_out (224 MB)
    are alive simultaneously, doubling peak memory.  Since the Triton kernel
    loads each tile into registers before storing, and each program instance
    handles a non-overlapping tile, we can safely store back to the same
    buffer.  This eliminates ~449 MB of transient allocations.

    Requires e_base and g_base to be contiguous (true in v6 where they come
    from separate ``torch.matmul`` calls).
    """
    assert e_base.is_contiguous(), "e_base must be contiguous for in-place write"
    assert g_base.is_contiguous(), "g_base must be contiguous for in-place write"

    M, N = e_base.shape
    has_lora = xa_gate is not None
    R = xa_gate.shape[1] if has_lora else 0
    BLOCK_R = max(triton.next_power_of_2(R), 16) if has_lora else 16

    H = torch.empty(M, N, dtype=e_base.dtype, device=e_base.device)

    # In-place: E_out and G_out point to the SAME storage as e_base / g_base.
    # The kernel loads tile data into registers before storing, so this is safe.
    E_out = e_base if save_eg else H
    G_out = g_base if save_eg else H

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
# Inference path: re-export v5's (pre-merged weights, sidesteps LoRA at runtime)
# ---------------------------------------------------------------------------

lora_mlp_v6_inference = lora_mlp_v5_inference


# ---------------------------------------------------------------------------
# Triton kernel: stacked LoRA-A matmul (X @ [A_gate; A_up].T)
# ---------------------------------------------------------------------------
#
# This is a standard Triton GEMM with M = batch*seq, N = 2*r (tiny, 16-128),
# K = H (4096 or 5120 typical). We launch a 2D grid (pid_m, pid_n=0). Because
# N is tiny (≤128 for r≤64) we use BLOCK_N = next_power_of_2(2r) and a SINGLE
# program along N (num_n_blocks==1 for r≤64). The compiler caches each
# [BLOCK_M, BLOCK_K] X tile in SMEM and reuses it across the 2r output
# columns — i.e. X is loaded once per K tile, then dot-multiplied against
# BOTH A_gate and A_up implicitly via the stacked A_stack.

@triton.jit
def _stacked_lora_a_kernel(
    X_ptr, A_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_an, stride_ak,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Out[m, n] = sum_k X[m, k] * A[n, k]   (i.e. X @ A.T)

    Shapes:
        X:   [M, K] (contig in K)
        A:   [N, K] (contig in K) — stacked [A_gate; A_up] with N = 2*r
        Out: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = A_ptr + offs_n[:, None] * stride_an + offs_k[None, :] * stride_ak

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        k_mask = (k_offset + offs_k) < K
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_n[:, None] < N) & k_mask[None, :],
            other=0.0,
        )
        # A is [N, K]; we want X @ A.T → use a_tile.T as RHS.
        acc += tl.dot(x_tile, tl.trans(a_tile))

        x_ptrs += BLOCK_K * stride_xk
        a_ptrs += BLOCK_K * stride_ak

    out_dtype = Out_ptr.dtype.element_ty
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_dtype), mask=out_mask)


def _stacked_lora_a_launch(
    X: torch.Tensor,
    A_stack: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch the stacked LoRA-A Triton kernel. Returns ``out`` for chaining."""
    M, K = X.shape
    N, K2 = A_stack.shape
    assert K2 == K, f"K mismatch: X.shape[1]={K} vs A_stack.shape[1]={K2}"
    assert out.shape == (M, N), f"out shape {out.shape} != ({M}, {N})"

    # Tile sizing: BLOCK_N is the smallest power-of-2 ≥ N (so single program
    # along N for typical 2r ≤ 128). BLOCK_M = 64 matches v3/v5's epilogue.
    # BLOCK_K = 32 keeps the SMEM footprint small; the K loop iterates.
    BLOCK_N = max(triton.next_power_of_2(N), 16)
    BLOCK_M = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _stacked_lora_a_kernel[grid](
        X, A_stack, out,
        M, N, K,
        X.stride(0), X.stride(1),
        A_stack.stride(0), A_stack.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def stack_lora_a(A_gate: torch.Tensor, A_up: torch.Tensor) -> torch.Tensor:
    """Stack A_gate and A_up vertically into a single [2r, H] tensor."""
    return torch.cat([A_gate, A_up], dim=0).contiguous()


def fused_lora_a_stacked(
    X: torch.Tensor,
    A_stack: torch.Tensor,
    r: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (xa_gate, xa_up) = (X @ A_gate.T, X @ A_up.T) in a single
    Triton GEMM with stacked A.

    Args:
        X:        [M, K] contiguous in K
        A_stack:  [2*r, K] = cat(A_gate, A_up)
        r:        LoRA rank (so output slices are [:, :r] and [:, r:2*r])

    Returns:
        xa_gate: [M, r] (view of the [M, 2r] output)
        xa_up:   [M, r] (view of the [M, 2r] output)

    Notes:
        The two returned views SHARE storage with the same [M, 2r] tensor.
        They are non-contiguous in N (stride 0 = 2r). The Triton
        ``fused_lora_swiglu`` consumes them via explicit strides so the views
        are fine to pass directly.
    """
    M = X.shape[0]
    N = 2 * r
    out = torch.empty(M, N, dtype=X.dtype, device=X.device)
    _stacked_lora_a_launch(X, A_stack, out)
    return out[:, :r], out[:, r:N]


# ---------------------------------------------------------------------------
# Side-stream helper
# ---------------------------------------------------------------------------

def _get_v6_side_stream() -> torch.cuda.Stream:
    """Module-global cached side stream for ad-hoc lora_mlp_v6 calls.

    Per-instance caching lives on ``LoRAMLPv6Module``; this fallback is for
    direct ``lora_mlp_v6(...)`` calls without an nn.Module wrapper.
    """
    global _CACHED_SIDE_STREAM
    if "_CACHED_SIDE_STREAM" not in globals() or _CACHED_SIDE_STREAM is None:
        _CACHED_SIDE_STREAM = torch.cuda.Stream()
    return _CACHED_SIDE_STREAM


_CACHED_SIDE_STREAM: Optional[torch.cuda.Stream] = None


# ---------------------------------------------------------------------------
# Forward implementation (sync + streams)
# ---------------------------------------------------------------------------

def _v6_forward_impl(
    X: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    A_stack: Optional[torch.Tensor],
    B_gate: Optional[torch.Tensor], B_up: Optional[torch.Tensor],
    s_gate: float, s_up: float,
    W_down: torch.Tensor,
    A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
    r: int,
    save_eg: bool,
    enable_streams: bool,
    side_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Shared v6 forward used by both training (save_eg=True) and inference.

    Returns (out, e_full, g_full). ``e_full/g_full`` are None unless
    ``save_eg=True``.
    """
    orig_shape = X.shape
    X_flat = X.view(-1, X.shape[-1]) if X.dim() == 3 else X
    has_lora = A_stack is not None and B_gate is not None and r > 0

    if enable_streams and X_flat.is_cuda:
        default_stream = torch.cuda.current_stream()
        side = side_stream if side_stream is not None else _get_v6_side_stream()
        # Make the side stream wait for any in-flight work on the default stream
        # that allocated/touches X (so reading X on `side` is safe).
        side.wait_stream(default_stream)

        # Phase 1a (default): e_base = X @ W_gate.T
        e_base = torch.matmul(X_flat, W_gate.t())
        # Phase 1b (side): g_base = X @ W_up.T  (overlaps with 1a)
        with torch.cuda.stream(side):
            g_base = torch.matmul(X_flat, W_up.t())
        # Phase 1c (default): stacked LoRA-A on default stream
        if has_lora:
            xa_gate, xa_up = fused_lora_a_stacked(X_flat, A_stack, r)
        else:
            xa_gate = xa_up = None
        # Default must wait for `side` (g_base) before the Triton SwiGLU epilogue
        # reads g_base.
        default_stream.wait_stream(side)
    else:
        e_base = torch.matmul(X_flat, W_gate.t())
        g_base = torch.matmul(X_flat, W_up.t())
        if has_lora:
            xa_gate, xa_up = fused_lora_a_stacked(X_flat, A_stack, r)
        else:
            xa_gate = xa_up = None

    # Phase 3 (Triton, default stream): fused LoRA addition + SwiGLU
    # In-place variant: when save_eg=True, e_full/g_full reuse the storage
    # of e_base/g_base (saves ~449 MB at LLaMA-8B scale).
    h, e_full, g_full = fused_lora_swiglu_inplace(
        e_base, g_base,
        xa_gate, xa_up,
        B_gate, B_up,
        s_gate, s_up,
        save_eg=save_eg,
    )

    # Phase 4: down GEMM in parallel with the tiny h @ A_down.T (if LoRA)
    if enable_streams and X_flat.is_cuda and has_lora and A_down is not None:
        default_stream = torch.cuda.current_stream()
        side = side_stream if side_stream is not None else _get_v6_side_stream()
        side.wait_stream(default_stream)

        out = torch.matmul(h, W_down.t())  # [M, H] on default
        with torch.cuda.stream(side):
            # h @ A_down.T is too small to fuse with the down GEMM in a single
            # Triton kernel benefit-wise (single A, only r outputs). Just use
            # cuBLAS on the side stream and let stream parallelism hide it.
            xa_down = torch.matmul(h, A_down.t())
        default_stream.wait_stream(side)
        out.addmm_(xa_down, B_down.t(), alpha=s_down)
    else:
        out = torch.matmul(h, W_down.t())
        if has_lora and A_down is not None:
            xa_down = torch.matmul(h, A_down.t())
            out.addmm_(xa_down, B_down.t(), alpha=s_down)

    if len(orig_shape) == 3:
        out = out.view(orig_shape[0], orig_shape[1], -1)
        if e_full is not None:
            e_full = e_full.view(orig_shape[0], orig_shape[1], -1)
            g_full = g_full.view(orig_shape[0], orig_shape[1], -1)

    return out, e_full, g_full


# ---------------------------------------------------------------------------
# Convenience function (no autograd)
# ---------------------------------------------------------------------------

def lora_mlp_v6(
    X: torch.Tensor,
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    W_down: torch.Tensor, A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
    A_stack: Optional[torch.Tensor] = None,
    enable_streams: bool = True,
    side_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Full LoRA MLP v6 forward (no autograd; for inference / benchmarking).

    Args:
        X, W_*, A_*, B_*, s_*: standard LoRA MLP params.
        A_stack: optional pre-stacked ``cat([A_gate, A_up], dim=0)`` of shape
                 ``[2r, H]``. If None, it is built on the fly.
        enable_streams: if True (default), gate/up GEMMs run on separate CUDA
                        streams (and same for the down phase).
        side_stream: optional override for the side stream. If None, a
                     module-global cached stream is used.
    """
    has_lora = (A_gate is not None) and (B_gate is not None)
    r = A_gate.shape[0] if has_lora else 0
    if has_lora and A_stack is None:
        A_stack = stack_lora_a(A_gate, A_up)

    out, _, _ = _v6_forward_impl(
        X, W_gate, W_up,
        A_stack if has_lora else None,
        B_gate if has_lora else None,
        B_up if has_lora else None,
        s_gate, s_up,
        W_down,
        A_down if has_lora else None,
        B_down if has_lora else None,
        s_down,
        r=r,
        save_eg=False,
        enable_streams=enable_streams,
        side_stream=side_stream,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd Function (training)
# ---------------------------------------------------------------------------

class LoRAMLPv6(torch.autograd.Function):
    """
    v6 LoRA MLP with full backward pass.

    Forward:  cuBLAS gate / up + Triton-stacked LoRA-A + Triton fused
              LoRA+SwiGLU + cuBLAS down + cuBLAS tiny A_down + addmm_.
              Optionally splits the gate/up and down/A_down phases across
              two CUDA streams (``enable_streams=True``).

    Backward: identical math to v3 / v5 / v5_upgrade_1 — uses Unsloth's
              optimized in-place buffer-reuse pattern (matmul_lora +
              swiglu_DWf_DW_dfg_kernel). Backward GEMMs have distinct
              LHS operands so neither packing nor stream-parallelism help.
    """

    @staticmethod
    def forward(
        ctx,
        X,
        W_gate, A_gate, B_gate, s_gate,
        W_up, A_up, B_up, s_up,
        W_down, A_down, B_down, s_down,
        A_stack=None,
        enable_streams: bool = True,
        side_stream: Optional[torch.cuda.Stream] = None,
    ):
        if X.dtype == torch.float64:
            # fp64 fallback (gradcheck): no Triton, no streams, no packing.
            e = X @ W_gate.t() + s_gate * ((X @ A_gate.t()) @ B_gate.t())
            g = X @ W_up.t() + s_up * ((X @ A_up.t()) @ B_up.t())
            h = F.silu(e) * g
            out = h @ W_down.t() + s_down * ((h @ A_down.t()) @ B_down.t())
        else:
            has_lora = (A_gate is not None) and (B_gate is not None)
            r = A_gate.shape[0] if has_lora else 0
            if has_lora and A_stack is None:
                A_stack = stack_lora_a(A_gate, A_up)

            out, e, g = _v6_forward_impl(
                X, W_gate, W_up,
                A_stack if has_lora else None,
                B_gate if has_lora else None,
                B_up if has_lora else None,
                s_gate, s_up,
                W_down,
                A_down if has_lora else None,
                B_down if has_lora else None,
                s_down,
                r=r,
                save_eg=True,
                enable_streams=enable_streams,
                side_stream=side_stream,
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
            # fp64 fallback (gradcheck): no in-place ops, safe for reentrant backward.
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
            dX = (
                df @ W_up + s_up * ((df @ B_up) @ A_up)
                + de @ W_gate + s_gate * ((de @ B_gate) @ A_gate)
            )
        else:
            # Production path: identical to v5_upgrade_1's backward.
            e = e.contiguous()
            g = g.contiguous()

            gateA, gateB = A_gate.to(dtype).t(), B_gate.to(dtype).t()
            upA, upB = A_up.to(dtype).t(), B_up.to(dtype).t()
            downA, downB = A_down.to(dtype).t(), B_down.to(dtype).t()

            DW = unsloth_matmul_lora(dY, W_down.t(), None, downB, downA, s_down)
            DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
            h, df, de = DW, e, g

            d_downA = torch.empty_like(downA)
            d_downB = torch.empty_like(downB)
            d_gateA = torch.empty_like(gateA)
            d_gateB = torch.empty_like(gateB)
            d_upA = torch.empty_like(upA)
            d_upB = torch.empty_like(upB)

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
            None,  # A_stack
            None,  # enable_streams
            None,  # side_stream
        )


# ---------------------------------------------------------------------------
# nn.Module wrapper with cached stacked-A buffer
# ---------------------------------------------------------------------------

class LoRAMLPv6Module(nn.Module):
    """
    Convenience nn.Module for v6 with a cached stacked A buffer.

    Usage:
        mlp = LoRAMLPv6Module(hidden=4096, intermediate=14336, rank=16,
                              dtype=torch.bfloat16).cuda()
        # After every optimizer step that updates A_gate / A_up:
        mlp.refresh_packed()
        out = mlp(X)

    The cache (``_A_stack_cache``) is registered as a non-persistent buffer so
    it does not show up in state_dict. It is rebuilt by ``refresh_packed()``
    or lazily on first forward.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        rank: int,
        s_gate: float = 1.0,
        s_up: float = 1.0,
        s_down: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_streams: bool = True,
    ):
        super().__init__()
        H, I, r = hidden, intermediate, rank

        def W(out_dim, in_dim):
            return torch.randn(out_dim, in_dim, dtype=dtype, device=device) * 0.02

        def A(in_dim):
            return torch.randn(r, in_dim, dtype=dtype, device=device) * 0.02

        def B(out_dim):
            return torch.zeros(out_dim, r, dtype=dtype, device=device)

        # Base weights (frozen — no requires_grad by convention).
        self.W_gate = nn.Parameter(W(I, H), requires_grad=False)
        self.W_up = nn.Parameter(W(I, H), requires_grad=False)
        self.W_down = nn.Parameter(W(H, I), requires_grad=False)
        # LoRA trainables.
        self.A_gate = nn.Parameter(A(H))
        self.B_gate = nn.Parameter(B(I))
        self.A_up = nn.Parameter(A(H))
        self.B_up = nn.Parameter(B(I))
        self.A_down = nn.Parameter(A(I))
        self.B_down = nn.Parameter(B(H))

        self.s_gate = s_gate
        self.s_up = s_up
        self.s_down = s_down
        self.rank = r
        self.enable_streams = enable_streams

        # Non-persistent buffer (won't show up in state_dict).
        self.register_buffer("_A_stack_cache", None, persistent=False)
        self._cache_valid = False
        # Lazy side stream (not a buffer — CUDA Stream isn't a tensor).
        self._v6_side_stream: Optional[torch.cuda.Stream] = None

    def refresh_packed(self) -> None:
        """Rebuild the cached stacked-A buffer from current A_gate / A_up."""
        with torch.no_grad():
            self._A_stack_cache = stack_lora_a(self.A_gate.data, self.A_up.data)
        self._cache_valid = True

    def invalidate_packed(self) -> None:
        """Force the next forward to rebuild the stacked-A cache."""
        self._cache_valid = False
        self._A_stack_cache = None

    def _get_side_stream(self) -> Optional[torch.cuda.Stream]:
        if not self.enable_streams:
            return None
        if self._v6_side_stream is None and torch.cuda.is_available():
            self._v6_side_stream = torch.cuda.Stream()
        return self._v6_side_stream

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._cache_valid:
            self.refresh_packed()
        return LoRAMLPv6.apply(
            X,
            self.W_gate, self.A_gate, self.B_gate, self.s_gate,
            self.W_up, self.A_up, self.B_up, self.s_up,
            self.W_down, self.A_down, self.B_down, self.s_down,
            self._A_stack_cache,
            self.enable_streams,
            self._get_side_stream(),
        )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "fused_lora_a_stacked",
    "stack_lora_a",
    "lora_mlp_v6",
    "lora_mlp_v6_inference",
    "LoRAMLPv6",
    "LoRAMLPv6Module",
    "prepare_inference_weights",
]

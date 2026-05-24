"""
v5_upgrade_1 — v5 with two cuBLAS-tile-alignment fixes (training path)

Background
----------
The v5 packing experiment (see ``experiments/v5/lora_mlp_kernel_v5.py``)
combined v3's 8 cuBLAS calls into 4 by packing W_gate/W_up/A_gate/A_up into a
single mega-matrix and W_down/A_down into a second one. The microbench in
``benchmarks/microbench_v5_packing.py`` and the diagnosis at
``docs/analysis/v5_packing_diagnosis.md`` showed that:

  * Gate+up packing wins ~0.8 ms in isolation (mostly by absorbing the two
    skinny LoRA-A launches), BUT the resulting mega-matrix has
    N = 2*I + 2*r = 28704, which is *not* a multiple of 128 and pushes
    cuBLAS off its best tile (79.6% peak vs 82.7% peak for N=28672).
  * Down packing actually LOSES ~0.22 ms vs v3, for two reasons:
      - N = H + r = 4112 is off the clean N=4096 tile (~0.18 ms penalty).
      - The H slice of the [M, H+r] mega-output is non-contiguous, so
        ``addmm_`` requires a fresh ``.contiguous()`` copy (~0.15 ms).

This upgrade applies both fixes inside the same algorithmic family ("packed
cuBLAS + Triton SwiGLU+LoRA epilogue"):

Change 1 — Drop down-phase packing
   Revert the down phase to v3's pattern (two separate cuBLAS calls plus
   ``addmm_`` on the contiguous out_base buffer). This recovers the wasted
   ``.contiguous()`` and lets cuBLAS pick its best N=H tile.

Change 2 — Pad gate+up mega N to a multiple of 128
   Append zero-padding rows at the end of the gate+up mega-matrix so the
   output column count is divisible by 128 (cuBLAS's preferred tile width
   on A100 bf16). The padded columns of the result are garbage (zeros from
   zero @ X), and we simply ignore them when slicing.

Inference path is unchanged — pre-merging LoRA into the base weights already
sidesteps both pain points (``W_down_eff`` is [H, I], not [H+r, I], and the
gate/up matmuls have N=I=14336 which is already a multiple of 128). v5's
inference path is re-exported here as ``lora_mlp_v5_upgrade_1_inference`` for
convenience.

Per-project convention (see ``.cursor/rules/lora-mlp-kernel-research.mdc``),
upgrades within the same algorithmic approach live in a separate file with
``_upgrade_<N>`` suffix; the previous version is never modified.
"""

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reference.unsloth_baseline import (
    matmul_lora as unsloth_matmul_lora,
    swiglu_DWf_DW_dfg_kernel,
)

from experiments.v5.lora_mlp_kernel_v5 import (
    fused_lora_swiglu,
    merge_lora_weights,
    prepare_inference_weights,
    lora_mlp_v5_inference,
)


# ---------------------------------------------------------------------------
# Re-exports of unchanged v5 components
# ---------------------------------------------------------------------------

# Inference path didn't need either fix, so just alias v5's implementation.
lora_mlp_v5_upgrade_1_inference = lora_mlp_v5_inference


# ---------------------------------------------------------------------------
# Weight packing utilities
# ---------------------------------------------------------------------------

def pack_gate_up_weights_padded(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    A_gate: Optional[torch.Tensor] = None,
    A_up: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Pack W_gate, W_up, A_gate, A_up into a single mega-matrix with zero
    rows appended so the resulting output dimension N is a multiple of 128
    (cuBLAS's preferred tile width on A100 bf16).

    Layout (rows of the returned tensor, top to bottom):
        W_gate (I rows)
        W_up   (I rows)
        A_gate (r rows)   [if has_lora]
        A_up   (r rows)   [if has_lora]
        zeros  (pad_rows) [if pad_rows > 0]

    The padded tail of the mega-matmul output is zero @ X = 0; the caller
    must simply ignore those columns when slicing.

    Args:
        W_gate, W_up: [I, H]
        A_gate, A_up: [r, H] (or None for no-LoRA)

    Returns:
        (W_mega_padded, pad_rows):
          W_mega_padded — [N_padded, H], contiguous, where
                          N_padded = ceil((2*I + 2*r) / 128) * 128.
          pad_rows      — number of zero rows appended at the bottom.
    """
    I = W_gate.shape[0]
    H = W_gate.shape[1]
    has_lora = A_gate is not None and A_up is not None
    r = A_gate.shape[0] if has_lora else 0

    n_unpadded = 2 * I + 2 * r
    n_padded = ((n_unpadded + 127) // 128) * 128
    pad_rows = n_padded - n_unpadded

    parts = [W_gate, W_up]
    if has_lora:
        parts.extend([A_gate, A_up])
    if pad_rows > 0:
        parts.append(
            torch.zeros(pad_rows, H, dtype=W_gate.dtype, device=W_gate.device)
        )

    return torch.cat(parts, dim=0).contiguous(), pad_rows


# ---------------------------------------------------------------------------
# Training forward (packed gate+up only; v3-style down)
# ---------------------------------------------------------------------------

def _v5_upgrade_1_forward_impl(
    X: torch.Tensor,
    W_mega_padded: torch.Tensor,
    B_gate: Optional[torch.Tensor], B_up: Optional[torch.Tensor],
    s_gate: float, s_up: float,
    W_down: torch.Tensor,
    A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
    I: int, r: int,
    save_eg: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward pass with the two upgrades applied:

      Phase 1 (cuBLAS) — one mega-GEMM ``X @ W_mega_padded^T`` with N padded
                         to a multiple of 128. Slice off the first 2*I + 2*r
                         columns; ignore the pad tail.
      Phase 2 (Triton) — fused LoRA addition + SwiGLU (same kernel as v5).
      Phase 3 (cuBLAS) — v3-style down: ``out = h @ W_down^T`` (clean N=H tile),
                         then optional skinny ``h @ A_down^T`` and an
                         ``addmm_`` for the LoRA-B term. ``out`` is contiguous
                         straight from cuBLAS — no ``.contiguous()`` copy.
    """
    orig_shape = X.shape
    X_flat = X.view(-1, X.shape[-1]) if X.dim() == 3 else X
    has_lora = (B_gate is not None) and (r > 0)

    # ── Phase 1: padded mega-GEMM (gate base + up base + LoRA-A_gate + LoRA-A_up)
    # The last `pad_rows` columns are zeros and are deliberately ignored.
    result = torch.matmul(X_flat, W_mega_padded.t())
    e_base = result[:, :I]
    g_base = result[:, I:2 * I]
    if has_lora:
        xa_gate = result[:, 2 * I: 2 * I + r]
        xa_up = result[:, 2 * I + r: 2 * I + 2 * r]
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

    # ── Phase 3: v3-style down (no packing)
    # `out` is contiguous straight from cuBLAS, so addmm_ doesn't need any
    # extra copy. cuBLAS gets a clean N=H tile (multiple of 128 for typical H).
    out = torch.matmul(h, W_down.t())  # [M, H]
    if has_lora:
        xa_down = torch.matmul(h, A_down.t())  # [M, r]
        out.addmm_(xa_down, B_down.t(), alpha=s_down)

    if len(orig_shape) == 3:
        out = out.view(orig_shape[0], orig_shape[1], -1)
        if e_full is not None:
            e_full = e_full.view(orig_shape[0], orig_shape[1], -1)
            g_full = g_full.view(orig_shape[0], orig_shape[1], -1)

    return out, e_full, g_full


def lora_mlp_v5_upgrade_1(
    X: torch.Tensor,
    W_gate: torch.Tensor, A_gate: Optional[torch.Tensor], B_gate: Optional[torch.Tensor], s_gate: float,
    W_up: torch.Tensor, A_up: Optional[torch.Tensor], B_up: Optional[torch.Tensor], s_up: float,
    W_down: torch.Tensor, A_down: Optional[torch.Tensor], B_down: Optional[torch.Tensor], s_down: float,
    W_mega_padded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full LoRA MLP v5_upgrade_1 forward (inference convenience, no autograd).

    If ``W_mega_padded`` is not provided, it is packed on the fly. For repeated
    forwards, pack it once and reuse.

    The down weights are used as-is (no packing in v5_upgrade_1).
    """
    has_lora = (A_gate is not None) and (B_gate is not None)
    I = W_gate.shape[0]
    r = A_gate.shape[0] if has_lora else 0

    if W_mega_padded is None:
        W_mega_padded, _pad = pack_gate_up_weights_padded(
            W_gate, W_up,
            A_gate if has_lora else None,
            A_up if has_lora else None,
        )

    out, _, _ = _v5_upgrade_1_forward_impl(
        X, W_mega_padded,
        B_gate if has_lora else None,
        B_up if has_lora else None,
        s_gate, s_up,
        W_down,
        A_down if has_lora else None,
        B_down if has_lora else None,
        s_down,
        I, r,
        save_eg=False,
    )
    return out


# ---------------------------------------------------------------------------
# Autograd Function (training)
# ---------------------------------------------------------------------------

class LoRAMLPv5_upgrade_1(torch.autograd.Function):
    """
    v5_upgrade_1 LoRA MLP with full backward pass.

    Forward:
      Padded gate+up mega-cuBLAS (N % 128 == 0)
      + Triton fused LoRA+SwiGLU
      + v3-style down (separate W_down and A_down cuBLAS calls + addmm_).
      Down packing is dropped per the v5 packing diagnosis (it loses
      ~0.22 ms net vs v3 because of the awkward N=H+r tile and the
      forced ``.contiguous()`` copy).

    Backward:
      Identical math to v3/v5 — packing only helps the forward path because
      each backward matmul has a different left-hand side (dY, df, de, X)
      and no shared input.
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

            W_mega_padded, _pad = pack_gate_up_weights_padded(
                W_gate, W_up,
                A_gate if has_lora else None,
                A_up if has_lora else None,
            )

            out, e, g = _v5_upgrade_1_forward_impl(
                X, W_mega_padded,
                B_gate if has_lora else None,
                B_up if has_lora else None,
                s_gate, s_up,
                W_down,
                A_down if has_lora else None,
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
            # Production path: same as v3/v5 — Unsloth's optimized in-place
            # buffer-reuse pattern. Backward doesn't benefit from packing.
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

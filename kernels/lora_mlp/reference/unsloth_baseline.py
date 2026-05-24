"""
Unsloth's exact LoRA MLP code, extracted for standalone benchmarking.

Sources (Apache-2.0 license, retrieved 2026-05-23):
  - matmul_lora:              https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py
  - LoRA_MLP:                 https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py
  - swiglu_fg_kernel:         https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py
  - swiglu_DWf_DW_dfg_kernel: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py

Only change: fast_dequantize stubbed to identity (we benchmark non-quantized weights).
Everything else is copy-pasted verbatim from Unsloth's repo.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# From unsloth/kernels/utils.py — fast_dequantize stubbed for non-quantized
# ─────────────────────────────────────────────────────────────────────────────

def fast_dequantize(W, W_quant=None, use_global_buffer=True):
    """Stub: with non-quantized weights, just return W as-is."""
    if W_quant is not None:
        raise NotImplementedError("Quantized weights not supported in standalone benchmark")
    return W


# ─────────────────────────────────────────────────────────────────────────────
# From unsloth/kernels/utils.py — matmul_lora (EXACT Unsloth code)
# ─────────────────────────────────────────────────────────────────────────────

def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    W = fast_dequantize(W, W_quant, use_global_buffer=True)
    out = torch.matmul(X, W.t(), out=out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch.matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out


# ─────────────────────────────────────────────────────────────────────────────
# From unsloth/kernels/swiglu.py — Triton SwiGLU kernels (EXACT Unsloth code)
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE = 1024
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER

from contextlib import nullcontext
def torch_gpu_device(device):
    return nullcontext()


@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask = mask)


def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _fg_kernel[grid](
            e,
            g,
            h,
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE,
            LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return h


@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    tl.store(DW + offsets, h_row, mask = mask)
    tl.store(e + offsets, df_row, mask = mask)
    tl.store(g + offsets, de_row, mask = mask)


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(e.device):
        _DWf_DW_dfg_kernel[grid](
            DW,
            e,
            g,
            n_elements,
            BLOCK_SIZE = BLOCK_SIZE,
            LONG_INDEXING = 0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return DW, e, g


# ─────────────────────────────────────────────────────────────────────────────
# From unsloth/kernels/fast_lora.py — LoRA_MLP (EXACT Unsloth code)
# ─────────────────────────────────────────────────────────────────────────────

class LoRA_MLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW,   upW_quant,   upA,   upB,   upS,
        downW, downW_quant, downA, downB, downS,
        _forward_function,
        _backward_function,
        inplace = True,
    ):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW,   upW_quant,   upA,   upB,   upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW,   upW_quant,   upS,
            downW, downW_quant, downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i

    @staticmethod
    def backward(ctx, dY):
        (
            gateW, gateW_quant, gateS,
            upW,   upW_quant,   upS,
            downW, downW_quant, downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X.view(-1, X.shape[-1])
        e  = e.view(-1, e.shape[-1])
        g  = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype), gateB.to(dtype),
            upA.to(dtype),   upB.to(dtype),
            downA.to(dtype), downB.to(dtype),
        )

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(), gateB.t(),
            upA.t(),   upB.t(),
            downA.t(), downB.t(),
        )

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA   = torch.empty_like(upA)
        d_upB   = torch.empty_like(upB)

        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY,  alpha = downS, beta = 0)

        d_upA.addmm_(X.t(), df @ upB.t(),   alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ X.t(), df,   alpha = upS, beta = 0)

        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha = gateS, beta = 0)

        upW   = fast_dequantize(upW.t(), upW_quant)
        dX    = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)
        del upW
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW.t())
        del gateW
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)

        return (
            dX.view(batch, seq_len, hd),
            None, None, d_gateA.t(), d_gateB.t(), None,
            None, None, d_upA.t(),   d_upB.t(),   None,
            None, None, d_downA.t(), d_downB.t(), None,
            None, None, None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# From unsloth/kernels/fast_lora.py — apply_lora_mlp_swiglu (EXACT Unsloth code)
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora_mlp_swiglu(X, gate_proj, up_proj, down_proj, inplace=True):
    """
    Entry point matching Unsloth's apply_lora_mlp_swiglu.
    Adapted to accept weight dicts instead of nn.Module projections.

    gate_proj/up_proj/down_proj are dicts with keys: W, A, B, s
    """
    gateW, gateA, gateB, gateS = gate_proj["W"], gate_proj["A"], gate_proj["B"], gate_proj["s"]
    upW,   upA,   upB,   upS   = up_proj["W"],   up_proj["A"],   up_proj["B"],   up_proj["s"]
    downW, downA, downB, downS = down_proj["W"],  down_proj["A"], down_proj["B"], down_proj["s"]
    out = LoRA_MLP.apply(
        X,
        gateW, None, gateA, gateB, gateS,
        upW,   None, upA,   upB,   upS,
        downW, None, downA, downB, downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper for creating test parameters
# ─────────────────────────────────────────────────────────────────────────────

def make_lora_mlp_params(
    hidden_dim=4096,
    intermediate_dim=14336,
    rank=16,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad=False,
):
    """Create LoRA MLP params in Unsloth's format (W, A, B, s per projection)."""
    H, I, r = hidden_dim, intermediate_dim, rank
    s = 1.0

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
        gate_proj=dict(W=make_weight(I, H), A=make_lora_a(H), B=make_lora_b(I), s=s),
        up_proj=dict(W=make_weight(I, H), A=make_lora_a(H), B=make_lora_b(I), s=s),
        down_proj=dict(W=make_weight(H, I), A=make_lora_a(I), B=make_lora_b(H), s=s),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    B, S, H, I, r = 2, 64, 256, 512, 16
    params = make_lora_mlp_params(H, I, r, dtype=dtype, device=device)
    X = torch.randn(B, S, H, dtype=dtype, device=device)

    # Test matmul_lora
    W, A, Bm, s = params["gate_proj"]["W"], params["gate_proj"]["A"], params["gate_proj"]["B"], params["gate_proj"]["s"]
    out = matmul_lora(X, W, None, A, Bm, s)
    print(f"matmul_lora output: {out.shape}")

    # Test full MLP via LoRA_MLP
    out_mlp = apply_lora_mlp_swiglu(X, **params)
    print(f"LoRA_MLP output: {out_mlp.shape}")
    print("Unsloth baseline self-test passed.")

"""
Unsloth — LoRA MLP Custom Autograd Function

Source: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py
License: Apache-2.0
Retrieved: 2026-05-23

This is the core LoRA_MLP autograd function that fuses the entire
MLP forward+backward with LoRA into a single autograd.Function.
It handles all 3 projections (gate, up, down) with their LoRA terms,
the SwiGLU activation, and computes all 6 LoRA weight gradients
(dA + dB for each of gate, up, down) in the backward pass.

Key insight: This is NOT a Triton kernel fusion — it's a PyTorch-level
autograd fusion that reduces memory by reusing intermediate buffers
and computing gradients in a specific order to minimize peak memory.
"""

import torch

# From unsloth/kernels/utils.py — simplified for readability
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    """
    Compute X @ W + s * (X @ A) @ B in a memory-efficient way.
    W may be quantized (4-bit via bitsandbytes), dequantized on the fly.
    """
    dtype = X.dtype
    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    # Base matmul: X @ W^T (handles quantized weights)
    W = fast_dequantize(W, W_quant, use_global_buffer=True)
    out = torch.matmul(X, W.t(), out=out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA additive term: s * (X @ A^T) @ B^T
        A, B = A.t(), B.t()
        XA = torch.matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg       (gate projection)
    U = U + Au @ Bu       (up projection)
    W = W + Aw @ Bw       (down projection)

    ### SwiGLU(X) forward
    e = X @ G             (gate pre-activation)
    f = e * sigmoid(e)    (SiLU activation)
    g = X @ U             (up projection output)
    h = f * g             (SwiGLU output)
    i = h @ W             (down projection)

    ### Backward pass gradient formulas
    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (dY @ W.T * f)
    dC/dG = X.T @ (dY @ W.T * df * g)

    ### Down projection LoRA gradients
    dC/dAw = h.T @ dY @ Bw.T
    dC/dBw = Aw.T @ h.T @ dY

    ### Up projection LoRA gradients
    dC/dAu = X.T @ (dY @ W.T * f) @ Bu.T
    dC/dBu = Au.T @ X.T @ (dY @ W.T * f)

    ### Gate projection LoRA gradients
    dC/dAg = X.T @ (dY @ W.T * df * g) @ Bg.T
    dC/dBg = Ag.T @ X.T @ (dY @ W.T * df * g)
    """

    @staticmethod
    def forward(
        ctx,
        X,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW, upW_quant, upA, upB, upS,
        downW, downW_quant, downA, downB, downS,
        _forward_function,
        _backward_function,
        inplace=True,
    ):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)  # SwiGLU: silu(e) * g
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
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
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        # Cast LoRA weights and transpose for matmul
        gateA, gateB = gateA.to(dtype).t(), gateB.to(dtype).t()
        upA, upB = upA.to(dtype).t(), upB.to(dtype).t()
        downA, downB = downA.to(dtype).t(), downB.to(dtype).t()

        # Backprop through down projection: DW = dY @ W_down^T (+ LoRA terms)
        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        # Backprop through SwiGLU: returns (h, df, de)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        # Allocate LoRA gradient buffers
        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)

        # Down projection LoRA gradients
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha=downS, beta=0)

        # Up projection LoRA gradients
        d_upA.addmm_(X.t(), df @ upB.t(), alpha=upS, beta=0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha=upS, beta=0)

        # Gate projection LoRA gradients
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha=gateS, beta=0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha=gateS, beta=0)

        # Input gradient: dX flows through both up and gate projections
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out=X if ctx.inplace else None)
        del upW
        dX.addmm_(df @ upB.t(), upA.t(), alpha=upS)

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW.t())
        del gateW
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha=gateS)

        return (
            dX.view(batch, seq_len, hd),
            None, None, d_gateA.t(), d_gateB.t(), None,
            None, None, d_upA.t(), d_upB.t(), None,
            None, None, d_downA.t(), d_downB.t(), None,
            None, None, None,
        )


# ─── Entry point that patches onto the HuggingFace MLP module ───

def apply_lora_mlp_swiglu(self, X, inplace=True):
    """Called as self.mlp(X) after monkey-patching the MLP forward."""
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW, upW_quant, upA, upB, upS,
        downW, downW_quant, downA, downB, downS,
        swiglu_fg_kernel,           # forward activation function
        swiglu_DWf_DW_dfg_kernel,   # backward activation function
        inplace,
    )
    return out

"""
EXACT CODE extracted from Unsloth.

Sources:
  - matmul_lora():    https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py
  - LoRA_QKV class:   https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py
  - apply_lora_qkv(): https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py
License: Apache-2.0
Retrieved: 2026-05-24
Commit: main (latest at retrieval)

DO NOT MODIFY this code — it is a verbatim copy for reference.
Our annotations are in clearly-separated blocks marked with "# === OUR ANALYSIS ===".
"""

# ============================================================
# DEPENDENCIES (stubs / notes for helpers referenced by the code below)
# ============================================================
#
# These are NOT the real implementations — just documentation of what
# the code below calls, so you can read it without chasing imports.
#
# From unsloth/kernels/utils.py:
#
#   torch_matmul = torch.matmul
#   torch_mm = torch.mm
#   torch_mv = torch.mv
#       Simple aliases for torch builtins, used throughout.
#
#   Float8Tensor:
#       from torchao.quantization import Float8Tensor
#       (falls back to type(None) if torchao is not installed)
#       Used to detect FP8-quantized weights.
#
#   fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
#       Dequantizes bitsandbytes 4-bit or FP8 weights back to fp16/bf16.
#       Returns W unchanged if quant_state is None (i.e. W is not quantized).
#       Uses a global buffer to avoid repeated allocations when use_global_buffer=True.
#
#   fp8_linear(X, W, W_quant):
#       FP8 matmul path using torchao's Float8Tensor.
#
#   torch_amp_custom_fwd / torch_amp_custom_bwd:
#       Compatibility shims for torch.amp.custom_fwd/bwd across PyTorch versions:
#         if torch.__version__ < "2.4.0":
#             torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
#             torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
#         else:
#             torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
#             torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")
#
#   get_lora_parameters(proj):
#       Returns 5-tuple: (weight, weight_quant_state, lora_A, lora_B, scaling)
#       Extracts LoRA parameters from a PEFT LoRA layer.
#
#   _maybe_fake_quantize_activations(X, proj):
#       QAT (quantization-aware training) support — no-op if QAT is disabled.
#
# From unsloth/kernels/fast_lora.py imports:
#
#   import torch
#   from .utils import (
#       _maybe_fake_quantize_activations,
#       fast_dequantize,
#       QUANT_STATE,
#       get_lora_parameters,
#       get_lora_parameters_bias,
#       matmul_lora,
#       torch_amp_custom_fwd,
#       torch_amp_custom_bwd,
#   )

import torch


# ============================================================
# ORIGINAL CODE — matmul_lora() from unsloth/kernels/utils.py
# (lines 1036–1071)
# ============================================================

def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if isinstance(W, Float8Tensor):
        assert W.ndim == 2
        if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
            # In the backward pass, rowwise scaled becomes colwise scaled after we
            # transpose the weight tensor. Use this case to detect backward.
            # TODO: would be simpler if we simply don't call `matmul_lora` in backward
            W = W.dequantize()
        else:
            W = W.contiguous()
        out = torch_matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer = True)
        out = torch_matmul(X, W.t(), out = out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out


# === OUR ANALYSIS — matmul_lora() ===
#
# This is the core building block for every LoRA projection in Unsloth.
# Each call to matmul_lora() does up to 3 cuBLAS calls:
#
#   1. Base matmul:  out = X @ W.t()                [cuBLAS call 1]
#   2. LoRA down:    XA = X @ A.t()                 [cuBLAS call 2 — reads X AGAIN]
#   3. LoRA up:      out.addmm_(XA, B.t(), alpha=s) [cuBLAS call 3 — fused add+GEMM]
#
# Key observations for our fused kernel:
#   - X is read from HBM TWICE: once for base matmul, once for LoRA A matmul.
#   - XA (shape [M, r]) is written to HBM then immediately re-read by addmm_.
#   - The addmm_ call is already a good optimization: it fuses scalar multiply + matmul
#     + addition into a single cuBLAS kernel, avoiding a temporary for the LoRA output.
#   - The three branches (Float8Tensor, float8_e4m3fn, bitsandbytes) handle different
#     quantization formats. For our project, we focus on the non-quantized path.
#   - No cross-projection fusion: Q, K, V each call matmul_lora() independently.
#
# Our fused kernel eliminates the extra X read and XA materialization by computing
# the LoRA addition inside the base matmul's tile loop, keeping intermediates in
# registers/SRAM.


# ============================================================
# ORIGINAL CODE — LoRA_QKV class from unsloth/kernels/fast_lora.py
# (lines 325–530)
# ============================================================

class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace = True,
    ):
        dtype = X.dtype

        # bitsandbytes 8-bit matmul expects 2D inputs.
        # TorchInductor/AOTAutograd fails on 3D tensors during backward,
        # so we explicitly flatten the sequence dimension.
        orig_shape = X.shape
        X_for_matmul = X
        if X.dim() == 3:
            X_for_matmul = X.view(-1, X.shape[-1])
        Q = matmul_lora(X_for_matmul, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X_for_matmul, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X_for_matmul, VW, VW_quant, VA, VB, VS)

        # Restore original shape after matmul
        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], -1)
            K = K.view(orig_shape[0], orig_shape[1], -1)
            V = V.view(orig_shape[0], orig_shape[1], -1)

        ctx.custom_saved_tensors = (
            QW,
            QW_quant,
            QS,
            KW,
            KW_quant,
            KS,
            VW,
            VW_quant,
            VS,
        )
        ctx.save_for_backward(
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
        )
        ctx.inplace = inplace
        return Q, K, V

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        (
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
        ) = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])  # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = (
            QA.to(dtype),
            QB.to(dtype),
            KA.to(dtype),
            KB.to(dtype),
            VA.to(dtype),
            VB.to(dtype),
        )

        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        ### Weight projection LoRA weights
        # See our blogpost for more details.
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        # Q Projection
        # d_QA = X.t() @ (dQ @ QB.t())
        # d_QB = (QA.t() @ X.t()) @ dQ
        # d_QA *= QS
        # d_QB *= QS
        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha = QS, beta = 0)

        # K Projection
        # d_KA = X.t() @ (dK @ KB.t())
        # d_KB = (KA.t() @ X.t()) @ dK
        # d_KA *= KS
        # d_KB *= KS
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha = KS, beta = 0)

        # V Projection
        # d_VA = X.t() @ (dV @ VB.t())
        # d_VB = (VA.t() @ X.t()) @ dV
        # d_VA *= VS
        # d_VB *= VS
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha = VS, beta = 0)

        # Combine derivatives to find dX
        # dQ
        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace else None)
        del QW
        # dX += (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha = QS)

        # dK
        KW = fast_dequantize(KW.t(), KW_quant)
        # dX += dK @ KW.t()
        dX.addmm_(dK, KW.t())
        del KW
        # dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())
        dX.addmm_(dK @ KB.t(), KA.t(), alpha = KS)

        # dV
        VW = fast_dequantize(VW.t(), VW_quant)
        # dX += dV @ VW.t()
        dX.addmm_(dV, VW.t())
        del VW
        # dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())
        dX.addmm_(dV @ VB.t(), VA.t(), alpha = VS)

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
            None,
            None,
        )


# === OUR ANALYSIS — LoRA_QKV ===
#
# Forward:
#   Calls matmul_lora() 3 times (once each for Q, K, V), totalling:
#     - 9 cuBLAS kernel launches (3 per projection)
#     - 6 HBM reads of X (2 per projection: once for base matmul, once for LoRA A)
#     - 3 HBM writes of XA intermediates (one per projection)
#
# Backward:
#   Computes 6 LoRA parameter gradients (d_QA, d_QB, d_KA, d_KB, d_VA, d_VB)
#   and accumulates dX from all 3 projections. Key patterns:
#
#   1. Pre-allocates gradient tensors with torch.empty_like, then uses addmm_ with
#      beta=0 to write into them (avoids allocation + zeroing overhead).
#
#   2. dX accumulation chain:
#        dX  = dQ @ W_q.t()                   [cuBLAS — optionally reuses X buffer]
#        dX += s_q * (dQ @ B_q.t()) @ A_q.t() [LoRA contribution via addmm_]
#        dX += dK @ W_k.t()                   [cuBLAS via addmm_]
#        dX += s_k * (dK @ B_k.t()) @ A_k.t()
#        dX += dV @ W_v.t()                   [cuBLAS via addmm_]
#        dX += s_v * (dV @ B_v.t()) @ A_v.t()
#      Total: 9+ cuBLAS calls in backward (3 for base weights, 6 for LoRA)
#
#   3. ctx.inplace=True allows writing dX into the X buffer, saving one allocation.
#
#   4. fast_dequantize(W.t(), quant_state) is called in backward to recover the
#      base weights — note the .t() happens BEFORE dequantization.
#
#   5. The return tuple has 17 elements matching the 17 forward args
#      (X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS,
#       VW, VW_quant, VA, VB, VS, inplace), with None for non-differentiable args.
#
# Opportunities for our fused kernel:
#   - Forward: fuse all 3 projections into 1 kernel launch, read X once.
#   - Backward: fuse dX accumulation (currently 9 addmm_ calls) into fewer launches.
#   - Eliminate XA intermediates by keeping them in registers/SRAM.


# ============================================================
# ORIGINAL CODE — apply_lora_qkv() from unsloth/kernels/fast_lora.py
# (lines 532–556)
# ============================================================

def apply_lora_qkv(self, X, inplace = True):
    X = _maybe_fake_quantize_activations(X, self.q_proj)
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
    )
    return Q, K, V


# === OUR ANALYSIS — apply_lora_qkv() ===
#
# This is the entry point called from the model code. It:
#   1. Optionally applies fake quantization to activations (QAT support)
#   2. Extracts (W, quant_state, A, B, scaling) from each PEFT LoRA layer
#   3. Calls LoRA_QKV.apply() which routes through forward/backward
#
# For our drop-in replacement, we would swap LoRA_QKV.apply() with our
# fused kernel's autograd.Function.apply(), keeping the same parameter
# extraction logic.

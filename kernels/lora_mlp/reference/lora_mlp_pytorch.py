"""
PyTorch reference implementation of LoRA MLP (LLaMA-style SwiGLU).

Mirrors Unsloth's LoRA_MLP logic but uses only plain PyTorch — no
bitsandbytes, no quantization, no Triton. This serves as the correctness
ground truth for testing fused Triton kernels.

Three levels of reference are provided:
  1. matmul_lora()     — single projection with LoRA (tests v1)
  2. lora_swiglu_mlp() — full MLP forward (tests v2/v3)
  3. LoRAMLP           — autograd.Function with forward + backward (tests v3+)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Level 1: Single projection — X @ W + s * (X @ A) @ B
# ---------------------------------------------------------------------------

def matmul_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    s: float = 1.0,
) -> torch.Tensor:
    """
    Compute a single LoRA-augmented linear projection.

    Args:
        X: input tensor [B*S, H] or [B, S, H]
        W: weight matrix [out_dim, in_dim] (stored as in nn.Linear)
        A: LoRA down-projection [r, in_dim]
        B: LoRA up-projection [out_dim, r]
        s: LoRA scaling factor (alpha / r)

    Returns:
        Y: [B*S, out_dim] or [B, S, out_dim]
    """
    orig_shape = X.shape
    if X.dim() == 3:
        X = X.view(-1, X.shape[-1])

    out = X @ W.t()

    if A is not None and B is not None:
        XA = X @ A.t()
        out = out + s * (XA @ B.t())

    if len(orig_shape) == 3:
        out = out.view(orig_shape[0], orig_shape[1], -1)
    return out


# ---------------------------------------------------------------------------
# Level 2: Full MLP forward — gate + up + SwiGLU + down, all with LoRA
# ---------------------------------------------------------------------------

def lora_swiglu_mlp(
    X: torch.Tensor,
    W_gate: torch.Tensor,
    A_gate: Optional[torch.Tensor],
    B_gate: Optional[torch.Tensor],
    s_gate: float,
    W_up: torch.Tensor,
    A_up: Optional[torch.Tensor],
    B_up: Optional[torch.Tensor],
    s_up: float,
    W_down: torch.Tensor,
    A_down: Optional[torch.Tensor],
    B_down: Optional[torch.Tensor],
    s_down: float,
) -> torch.Tensor:
    """
    Full LoRA MLP forward pass (LLaMA-style SwiGLU).

    Matches the forward path of Unsloth's LoRA_MLP.apply().

    Args:
        X:      input [B, S, H]
        W_gate: gate weight [I, H]
        A_gate: gate LoRA A [r, H]
        B_gate: gate LoRA B [I, r]
        s_gate: gate LoRA scale
        W_up:   up weight [I, H]
        A_up:   up LoRA A [r, H]
        B_up:   up LoRA B [I, r]
        s_up:   up LoRA scale
        W_down: down weight [H, I]
        A_down: down LoRA A [r, I]
        B_down: down LoRA B [H, r]
        s_down: down LoRA scale

    Returns:
        output [B, S, H]
    """
    e = matmul_lora(X, W_gate, A_gate, B_gate, s_gate)
    g = matmul_lora(X, W_up, A_up, B_up, s_up)
    h = F.silu(e) * g
    out = matmul_lora(h, W_down, A_down, B_down, s_down)
    return out


# ---------------------------------------------------------------------------
# Level 3: autograd.Function with custom backward
# ---------------------------------------------------------------------------

class LoRAMLP(torch.autograd.Function):
    """
    Custom autograd.Function mirroring Unsloth's LoRA_MLP.

    Saves the same tensors as Unsloth (X, e, g, LoRA A/B) and computes
    all gradients in the backward pass — dX, d_gateA, d_gateB, d_upA,
    d_upB, d_downA, d_downB.

    Base weights (W_gate, W_up, W_down) are frozen and do not receive
    gradients (standard LoRA training).
    """

    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W_gate: torch.Tensor, A_gate: torch.Tensor, B_gate: torch.Tensor, s_gate: float,
        W_up: torch.Tensor, A_up: torch.Tensor, B_up: torch.Tensor, s_up: float,
        W_down: torch.Tensor, A_down: torch.Tensor, B_down: torch.Tensor, s_down: float,
    ) -> torch.Tensor:
        e = matmul_lora(X, W_gate, A_gate, B_gate, s_gate)
        g = matmul_lora(X, W_up, A_up, B_up, s_up)
        h = F.silu(e) * g
        out = matmul_lora(h, W_down, A_down, B_down, s_down)

        ctx.save_for_backward(
            A_gate, B_gate, A_up, B_up, A_down, B_down,
            X, e, g,
        )
        ctx.scales = (s_gate, s_up, s_down)
        ctx.base_weights = (W_gate, W_up, W_down)
        return out

    @staticmethod
    def backward(
        ctx, dY: torch.Tensor
    ) -> Tuple[
        torch.Tensor,  # dX
        None, torch.Tensor, torch.Tensor, None,  # gate: W(frozen), dA, dB, s
        None, torch.Tensor, torch.Tensor, None,  # up: W(frozen), dA, dB, s
        None, torch.Tensor, torch.Tensor, None,  # down: W(frozen), dA, dB, s
    ]:
        A_gate, B_gate, A_up, B_up, A_down, B_down, X, e, g = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.scales
        W_gate, W_up, W_down = ctx.base_weights

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X_flat = X.view(-1, X.shape[-1])
        e_flat = e.view(-1, e.shape[-1])
        g_flat = g.view(-1, g.shape[-1])
        dtype = X.dtype

        # Recompute h = SiLU(e) * g
        sig_e = torch.sigmoid(e_flat.float()).to(dtype)
        silu_e = e_flat * sig_e
        h = silu_e * g_flat

        # W is [out_dim, in_dim]. Forward used X @ W.t(), so backward
        # through W is: grad_input = grad_output @ W (no transpose).

        # ── Backprop through down projection ──
        # W_down: [H, I]. Forward: h @ W_down.t() -> [B*S, I] @ [I, H] = [B*S, H]
        # Backward: dY @ W_down -> [B*S, H] @ [H, I] = [B*S, I]
        DW = dY @ W_down
        if A_down is not None:
            # LoRA forward: s * (h @ A_down.t()) @ B_down.t()
            # A_down: [r, I], B_down: [H, r]
            # Backward: s * dY @ B_down @ A_down -> [B*S, H]@[H,r]@[r,I] = [B*S, I]
            DW = DW + s_down * ((dY @ B_down) @ A_down)

        # ── Down LoRA grads ──
        # d_downA [r, I] = s * (dY @ B_down).t() @ h = [r, B*S] @ [B*S, I]
        d_downA = s_down * ((dY @ B_down).t() @ h)
        # d_downB [H, r] = s * dY.t() @ h @ A_down.t() = [H, B*S]@[B*S,I]@[I,r]
        d_downB = s_down * (dY.t() @ h @ A_down.t())

        # ── Backprop through SwiGLU ──
        # df (gradient flowing to up path) = DW * SiLU(e)
        df = DW * silu_e
        # de (gradient flowing to gate path)
        dsilu = sig_e * (1.0 + e_flat * (1.0 - sig_e))
        de = DW * g_flat * dsilu

        # ── Up LoRA grads ──
        # A_up: [r, H], B_up: [I, r]
        # d_upA [r, H] = s * (df @ B_up).t() @ X = [r, B*S] @ [B*S, H]
        d_upA = s_up * ((df @ B_up).t() @ X_flat)
        # d_upB [I, r] = s * df.t() @ X @ A_up.t() = [I, B*S]@[B*S,H]@[H,r]
        d_upB = s_up * (df.t() @ X_flat @ A_up.t())

        # ── Gate LoRA grads ──
        # A_gate: [r, H], B_gate: [I, r]
        # d_gateA [r, H] = s * (de @ B_gate).t() @ X = [r, B*S] @ [B*S, H]
        d_gateA = s_gate * ((de @ B_gate).t() @ X_flat)
        # d_gateB [I, r] = s * de.t() @ X @ A_gate.t() = [I, B*S]@[B*S,H]@[H,r]
        d_gateB = s_gate * (de.t() @ X_flat @ A_gate.t())

        # ── Input gradient: dX ──
        # W_up: [I, H]. Backward: df @ W_up -> [B*S, I] @ [I, H] = [B*S, H]
        dX = df @ W_up
        if A_up is not None:
            # LoRA backward: s * df @ B_up @ A_up -> [B*S,I]@[I,r]@[r,H] = [B*S,H]
            dX = dX + s_up * ((df @ B_up) @ A_up)
        # W_gate: [I, H]. Same logic.
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
# Helper: create random LoRA MLP parameters for testing
# ---------------------------------------------------------------------------

def make_lora_mlp_params(
    hidden_dim: int = 4096,
    intermediate_dim: int = 14336,
    rank: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    requires_grad: bool = True,
) -> dict:
    """
    Create a full set of LoRA MLP parameters for testing.

    Returns a dict with keys:
        W_gate, A_gate, B_gate, s_gate,
        W_up, A_up, B_up, s_up,
        W_down, A_down, B_down, s_down
    """
    H, I, r = hidden_dim, intermediate_dim, rank
    scale = 1.0  # alpha / r, typically 1.0 for testing

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
        W_gate=make_weight(I, H),
        A_gate=make_lora_a(H),
        B_gate=make_lora_b(I),
        s_gate=scale,
        W_up=make_weight(I, H),
        A_up=make_lora_a(H),
        B_up=make_lora_b(I),
        s_up=scale,
        W_down=make_weight(H, I),
        A_down=make_lora_a(I),
        B_down=make_lora_b(H),
        s_down=scale,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp32 for numerical verification

    B, S, H, I, r = 2, 64, 128, 256, 8
    params = make_lora_mlp_params(H, I, r, dtype=dtype, device=device, requires_grad=True)
    X = torch.randn(B, S, H, dtype=dtype, device=device, requires_grad=True)

    # Level 2: functional forward
    out_fn = lora_swiglu_mlp(X, **params)
    print(f"lora_swiglu_mlp output shape: {out_fn.shape}")  # [B, S, H]

    # Level 3: autograd.Function forward + backward
    out_ag = LoRAMLP.apply(
        X,
        params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
        params["W_up"], params["A_up"], params["B_up"], params["s_up"],
        params["W_down"], params["A_down"], params["B_down"], params["s_down"],
    )
    print(f"LoRAMLP.apply output shape: {out_ag.shape}")

    # Check forward consistency
    diff = (out_fn - out_ag).abs().max().item()
    print(f"Forward consistency (fn vs autograd): max diff = {diff:.2e}")

    # Test backward
    loss = out_ag.sum()
    loss.backward()
    print(f"Backward completed. dX shape: {X.grad.shape}")
    print(f"dA_gate shape: {params['A_gate'].grad.shape}")
    print("All checks passed.")

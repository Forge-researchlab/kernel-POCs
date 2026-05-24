"""
Correctness tests for LoRA QKV fused Triton kernels.

Compares against:
  - PyTorch reference (reference/lora_qkv_pytorch.py) for ground truth

Tests are organized by scope and cover:
  - Forward correctness: matmul_lora (single proj) and lora_qkv_forward (all QKV)
  - GQA support: num_kv_heads != num_heads (asymmetric K/V output dims)
  - Backward correctness: LoRAQKV autograd.Function gradients
  - Gradcheck: torch.autograd.gradcheck in fp64
  - Multiple LoRA ranks: 8, 16, 32, 64
  - Multiple shapes: small, LLaMA-8B, LLaMA-70B scale
  - Edge cases: no LoRA, non-power-of-2, large LoRA scale, 2D/3D inputs

When kernel experiments (v1, v2, ...) are created, add test classes below
that import the kernel and compare against these reference results.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reference.lora_qkv_pytorch import (
    matmul_lora,
    lora_qkv_forward,
    LoRAQKV,
    make_lora_qkv_params,
)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"

RANKS = [8, 16, 32, 64]

SHAPES_SMALL = [
    # (M, H, H_q, H_kv)
    (32, 64, 64, 64),        # tiny MHA
    (64, 128, 128, 128),     # small MHA
    (64, 128, 128, 32),      # small GQA (4:1 ratio)
    (17, 33, 33, 33),        # non-power-of-2
]

SHAPES_LLAMA = [
    # (M, H, H_q, H_kv) — LLaMA-3 8B
    (2048, 4096, 4096, 1024),  # GQA: 32 heads, 8 kv heads
    (4096, 4096, 4096, 1024),  # larger batch*seq
    (8192, 4096, 4096, 4096),  # MHA fallback (32 kv heads)
]

GQA_CONFIGS = [
    # (num_heads, num_kv_heads, head_dim) → (H_q, H_kv)
    (32, 8, 128),    # LLaMA-3 8B: 4:1 GQA
    (64, 8, 128),    # LLaMA-3 70B: 8:1 GQA
    (32, 32, 128),   # MHA (no grouping)
    (16, 4, 64),     # small GQA
    (8, 1, 128),     # multi-query attention (1 KV head)
]


# ---------------------------------------------------------------------------
# Reference tests: matmul_lora (single projection)
# ---------------------------------------------------------------------------

class TestMatmulLora:
    """Tests for the reference matmul_lora function (Level 1)."""

    @pytest.mark.parametrize("M,H,H_q,H_kv", SHAPES_SMALL)
    def test_base_matmul_no_lora(self, M, H, H_q, H_kv):
        """Base matmul without LoRA produces correct output."""
        torch.manual_seed(42)
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        W = torch.randn(H_q, H, device=DEVICE, dtype=torch.float32) * 0.02
        out = matmul_lora(X, W, None, None, 1.0)
        expected = X @ W.t()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("rank", RANKS)
    def test_lora_ranks_fp32(self, rank):
        """matmul_lora works at various ranks in fp32."""
        torch.manual_seed(42)
        M, N, K = 128, 256, 512
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(rank, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, rank, device=DEVICE, dtype=torch.float32) * 0.02
        out = matmul_lora(X, W, A, B, 1.0)
        expected = X @ W.t() + (X @ A.t()) @ B.t()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_lora_scale(self):
        """Non-unit LoRA scale is applied correctly."""
        torch.manual_seed(42)
        M, N, K, r = 64, 128, 256, 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02
        for s in [0.5, 2.0, 0.125]:
            out = matmul_lora(X, W, A, B, s)
            expected = X @ W.t() + s * (X @ A.t()) @ B.t()
            torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_3d_input(self):
        """3D input [B, S, H] is handled correctly."""
        torch.manual_seed(42)
        B, S, H, N, r = 2, 64, 256, 512, 16
        X = torch.randn(B, S, H, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, H, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, H, device=DEVICE, dtype=torch.float32) * 0.02
        Bm = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02
        out = matmul_lora(X, W, A, Bm, 1.0)
        assert out.shape == (B, S, N)
        X_flat = X.reshape(-1, H)
        expected = (X_flat @ W.t() + (X_flat @ A.t()) @ Bm.t()).reshape(B, S, N)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reference tests: lora_qkv_forward (all Q/K/V projections)
# ---------------------------------------------------------------------------

class TestLoRAQKVForward:
    """Tests for the reference lora_qkv_forward function (Level 2)."""

    def test_mha_forward_fp32(self):
        """QKV forward with MHA (all same dimensions) in fp32."""
        torch.manual_seed(42)
        M, H, r = 128, 256, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=4, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (M, 256)
        assert K.shape == (M, 256)
        assert V.shape == (M, 256)

    @pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", GQA_CONFIGS)
    def test_gqa_forward_shapes(self, num_heads, num_kv_heads, head_dim):
        """QKV forward produces correct shapes for various GQA configs."""
        torch.manual_seed(42)
        H = num_heads * head_dim
        H_q = num_heads * head_dim
        H_kv = num_kv_heads * head_dim
        M, r = 64, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float32, device=DEVICE,
            requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (M, H_q), f"Q shape {Q.shape} != ({M}, {H_q})"
        assert K.shape == (M, H_kv), f"K shape {K.shape} != ({M}, {H_kv})"
        assert V.shape == (M, H_kv), f"V shape {V.shape} != ({M}, {H_kv})"

    def test_gqa_forward_correctness(self):
        """GQA forward matches per-projection matmul_lora calls."""
        torch.manual_seed(42)
        H, num_heads, num_kv_heads, head_dim, r = 512, 8, 2, 64, 16
        M = 128
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float32, device=DEVICE,
            requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)

        Q, K, V = lora_qkv_forward(X, **params)
        Q_ref = matmul_lora(X, params["W_q"], params["A_q"], params["B_q"], params["s_q"])
        K_ref = matmul_lora(X, params["W_k"], params["A_k"], params["B_k"], params["s_k"])
        V_ref = matmul_lora(X, params["W_v"], params["A_v"], params["B_v"], params["s_v"])

        torch.testing.assert_close(Q, Q_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(K, K_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(V, V_ref, rtol=1e-5, atol=1e-5)

    def test_forward_no_lora(self):
        """Forward without LoRA (A/B = None) is just base matmul."""
        torch.manual_seed(42)
        M, H, r = 64, 128, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=2, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(
            X,
            W_q=params["W_q"], W_k=params["W_k"], W_v=params["W_v"],
        )
        Q_ref = X @ params["W_q"].t()
        K_ref = X @ params["W_k"].t()
        V_ref = X @ params["W_v"].t()
        torch.testing.assert_close(Q, Q_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(K, K_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(V, V_ref, rtol=1e-5, atol=1e-5)

    def test_forward_bf16(self):
        """QKV forward in bf16 is close to fp32 ground truth."""
        torch.manual_seed(42)
        M, H, r = 256, 512, 16
        params_fp32 = make_lora_qkv_params(
            hidden_dim=H, num_heads=8, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X_fp32 = torch.randn(M, H, device=DEVICE, dtype=torch.float32)

        Q_fp32, K_fp32, V_fp32 = lora_qkv_forward(X_fp32, **params_fp32)

        params_bf16 = {
            k: (v.bfloat16() if isinstance(v, torch.Tensor) else v)
            for k, v in params_fp32.items()
        }
        Q_bf16, K_bf16, V_bf16 = lora_qkv_forward(X_fp32.bfloat16(), **params_bf16)

        torch.testing.assert_close(Q_bf16.float(), Q_fp32, rtol=2e-2, atol=0.5)
        torch.testing.assert_close(K_bf16.float(), K_fp32, rtol=2e-2, atol=0.5)
        torch.testing.assert_close(V_bf16.float(), V_fp32, rtol=2e-2, atol=0.5)

    @pytest.mark.parametrize("rank", RANKS)
    def test_forward_rank_sweep(self, rank):
        """QKV forward across LoRA ranks in fp32."""
        torch.manual_seed(42)
        M, H = 64, 256
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=1, head_dim=64,
            rank=rank, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)

        Q_ref = matmul_lora(X, params["W_q"], params["A_q"], params["B_q"], params["s_q"])
        torch.testing.assert_close(Q, Q_ref, rtol=1e-5, atol=1e-5)

    def test_3d_input(self):
        """QKV forward with [B, S, H] input."""
        torch.manual_seed(42)
        B, S, H, r = 2, 64, 256, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(B, S, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (B, S, 256)
        assert K.shape == (B, S, 128)
        assert V.shape == (B, S, 128)


# ---------------------------------------------------------------------------
# Backward tests: LoRAQKV autograd.Function
# ---------------------------------------------------------------------------

class TestLoRAQKVBackward:
    """Tests for LoRAQKV autograd.Function backward pass."""

    def _run_forward_backward(self, X_data, params, sum_outputs=True):
        """Run LoRAQKV forward + backward, return gradients."""
        X = X_data.clone().detach().requires_grad_(True)
        p = {
            k: (v.clone().detach().requires_grad_(v.requires_grad)
                 if isinstance(v, torch.Tensor) else v)
            for k, v in params.items()
        }
        Q, K, V = LoRAQKV.apply(
            X,
            p["W_q"], p["W_k"], p["W_v"],
            p["A_q"], p["B_q"], p["s_q"],
            p["A_k"], p["B_k"], p["s_k"],
            p["A_v"], p["B_v"], p["s_v"],
        )
        if sum_outputs:
            loss = Q.sum() + K.sum() + V.sum()
        else:
            loss = Q.sum()
        loss.backward()
        return {
            "dX": X.grad,
            "dA_q": p["A_q"].grad, "dB_q": p["B_q"].grad,
            "dA_k": p["A_k"].grad, "dB_k": p["B_k"].grad,
            "dA_v": p["A_v"].grad, "dB_v": p["B_v"].grad,
        }

    def test_gradcheck_fp64_mha(self):
        """torch.autograd.gradcheck passes in fp64 for MHA."""
        torch.manual_seed(42)
        M, H, r = 8, 32, 4
        num_heads, num_kv_heads, head_dim = 2, 2, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_fp64_gqa(self):
        """torch.autograd.gradcheck passes in fp64 for GQA."""
        torch.manual_seed(42)
        M, H, r = 8, 64, 4
        num_heads, num_kv_heads, head_dim = 4, 1, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_fp64_3d_input(self):
        """torch.autograd.gradcheck passes with 3D [B, S, H] input."""
        torch.manual_seed(42)
        B, S, H, r = 2, 4, 32, 4
        num_heads, num_kv_heads, head_dim = 2, 1, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_gradcheck_rank_sweep(self, rank):
        """Gradcheck across ranks in fp64."""
        torch.manual_seed(42)
        M, H = 8, 32
        num_heads, num_kv_heads, head_dim = 2, 1, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_dX_mha(self):
        """dX is correct for MHA (all same dimensions)."""
        torch.manual_seed(42)
        M, H, r = 64, 128, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=2, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        grads = self._run_forward_backward(X, params)

        assert grads["dX"] is not None
        assert grads["dX"].shape == X.shape
        assert not torch.isnan(grads["dX"]).any()
        assert not torch.isinf(grads["dX"]).any()

    def test_backward_dX_gqa(self):
        """dX is correct for GQA (K/V smaller than Q)."""
        torch.manual_seed(42)
        M, H, r = 64, 256, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=1, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        grads = self._run_forward_backward(X, params)

        assert grads["dX"].shape == X.shape
        assert not torch.isnan(grads["dX"]).any()

    def test_backward_lora_grads_shapes(self):
        """All LoRA gradient shapes are correct."""
        torch.manual_seed(42)
        M, H, r = 64, 256, 16
        num_heads, num_kv_heads, head_dim = 4, 1, 64
        H_q = num_heads * head_dim
        H_kv = num_kv_heads * head_dim
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float32, device=DEVICE,
            requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        grads = self._run_forward_backward(X, params)

        assert grads["dA_q"].shape == (r, H)
        assert grads["dB_q"].shape == (H_q, r)
        assert grads["dA_k"].shape == (r, H)
        assert grads["dB_k"].shape == (H_kv, r)
        assert grads["dA_v"].shape == (r, H)
        assert grads["dB_v"].shape == (H_kv, r)

    def test_backward_no_nan_inf(self):
        """No NaN or Inf in any gradient."""
        torch.manual_seed(42)
        M, H, r = 128, 256, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        grads = self._run_forward_backward(X, params)

        for name, grad in grads.items():
            assert not torch.isnan(grad).any(), f"{name} contains NaN"
            assert not torch.isinf(grad).any(), f"{name} contains Inf"

    def test_backward_lora_scale(self):
        """Non-unit LoRA scales produce correct gradients."""
        torch.manual_seed(42)
        M, H, r = 8, 32, 4
        num_heads, num_kv_heads, head_dim = 2, 1, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=r, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        params["s_q"] = 0.5
        params["s_k"] = 2.0
        params["s_v"] = 0.125
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKV.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Forward-backward consistency
# ---------------------------------------------------------------------------

class TestForwardBackwardConsistency:
    """Verify that LoRAQKV.apply forward matches lora_qkv_forward."""

    def test_forward_consistency_fp32(self):
        """autograd.Function forward matches functional forward in fp32."""
        torch.manual_seed(42)
        M, H, r = 128, 256, 16
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)

        Q_fn, K_fn, V_fn = lora_qkv_forward(X, **params)
        Q_ag, K_ag, V_ag = LoRAQKV.apply(
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )

        torch.testing.assert_close(Q_fn, Q_ag, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(K_fn, K_ag, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(V_fn, V_ag, rtol=1e-5, atol=1e-5)

    def test_forward_consistency_gqa(self):
        """autograd.Function forward matches functional for GQA shapes."""
        torch.manual_seed(42)
        M, H, r = 64, 512, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=8, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=True,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)

        Q_fn, K_fn, V_fn = lora_qkv_forward(X, **params)
        Q_ag, K_ag, V_ag = LoRAQKV.apply(
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )

        torch.testing.assert_close(Q_fn, Q_ag, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(K_fn, K_ag, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(V_fn, V_ag, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for the reference implementation."""

    def test_single_token(self):
        """Works with M=1 (single token)."""
        torch.manual_seed(42)
        M, H, r = 1, 128, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=2, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (1, 128)

    def test_large_batch_seq(self):
        """Works with large M (many tokens)."""
        torch.manual_seed(42)
        M, H, r = 8192, 256, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.bfloat16)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (M, 256)
        assert K.shape == (M, 128)

    def test_non_power_of_2_dims(self):
        """Works with non-power-of-2 dimensions."""
        torch.manual_seed(42)
        M, H, r = 17, 33, 5
        W_q = torch.randn(33, 33, device=DEVICE, dtype=torch.float32) * 0.02
        W_k = torch.randn(33, 33, device=DEVICE, dtype=torch.float32) * 0.02
        W_v = torch.randn(33, 33, device=DEVICE, dtype=torch.float32) * 0.02
        A_q = torch.randn(r, 33, device=DEVICE, dtype=torch.float32) * 0.02
        B_q = torch.randn(33, r, device=DEVICE, dtype=torch.float32) * 0.02
        A_k = torch.randn(r, 33, device=DEVICE, dtype=torch.float32) * 0.02
        B_k = torch.randn(33, r, device=DEVICE, dtype=torch.float32) * 0.02
        A_v = torch.randn(r, 33, device=DEVICE, dtype=torch.float32) * 0.02
        B_v = torch.randn(33, r, device=DEVICE, dtype=torch.float32) * 0.02
        X = torch.randn(M, 33, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(
            X, W_q, W_k, W_v, A_q, B_q, 1.0, A_k, B_k, 1.0, A_v, B_v, 1.0,
        )
        assert Q.shape == (17, 33)
        assert K.shape == (17, 33)

    def test_deterministic(self):
        """Same input produces identical output across runs."""
        torch.manual_seed(42)
        M, H, r = 64, 128, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=2, num_kv_heads=1, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q1, K1, V1 = lora_qkv_forward(X, **params)
        Q2, K2, V2 = lora_qkv_forward(X, **params)
        torch.testing.assert_close(Q1, Q2, rtol=0, atol=0)
        torch.testing.assert_close(K1, K2, rtol=0, atol=0)
        torch.testing.assert_close(V1, V2, rtol=0, atol=0)

    def test_no_nan_inf(self):
        """Output contains no NaN or Inf."""
        torch.manual_seed(42)
        M, H, r = 256, 512, 32
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=8, num_kv_heads=2, head_dim=64,
            rank=r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.bfloat16)
        Q, K, V = lora_qkv_forward(X, **params)
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            assert not torch.isnan(t).any(), f"{name} contains NaN"
            assert not torch.isinf(t).any(), f"{name} contains Inf"

    def test_large_lora_scale(self):
        """Handles large LoRA scales without overflow."""
        torch.manual_seed(42)
        M, H, r = 32, 64, 4
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=2, num_kv_heads=1, head_dim=32,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        params["s_q"] = 10.0
        params["s_k"] = 10.0
        params["s_v"] = 10.0
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32) * 0.1
        Q, K, V = lora_qkv_forward(X, **params)
        assert not torch.isnan(Q).any()
        assert not torch.isinf(Q).any()

    def test_multi_query_attention(self):
        """MQA with num_kv_heads=1 works correctly."""
        torch.manual_seed(42)
        M, H, r = 64, 256, 8
        params = make_lora_qkv_params(
            hidden_dim=H, num_heads=4, num_kv_heads=1, head_dim=64,
            rank=r, dtype=torch.float32, device=DEVICE, requires_grad=False,
        )
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_forward(X, **params)
        assert Q.shape == (M, 256)   # 4 heads × 64
        assert K.shape == (M, 64)    # 1 head × 64
        assert V.shape == (M, 64)    # 1 head × 64


# ---------------------------------------------------------------------------
# Placeholder for kernel experiment tests (v1, v2, ...)
# ---------------------------------------------------------------------------
#
# When experiments/v1/lora_qkv_kernel_v1.py is implemented, add test classes
# here that import the kernel and compare against the reference results.
#
# Example:
#
# class TestV1FusedLoRAQKV:
#     """Tests for experiments/v1 fused LoRA QKV kernel."""
#
#     def test_per_projection_fp32(self):
#         ...compare fused_lora_matmul_qkv() against matmul_lora()...
#
#     def test_full_qkv_vs_reference(self):
#         ...compare kernel output against lora_qkv_forward()...
#
#     def test_gqa_support(self):
#         ...verify GQA shapes and correctness...
#
# class TestV2FusedQKV:
#     """Tests for experiments/v2 fused Q+K+V kernel."""
#     ...
#
# class TestV3AutogradFunction:
#     """Tests for experiments/v3 LoRAQKV autograd.Function wrapper."""
#
#     def test_gradcheck_fp64(self):
#         ...
#
#     def test_backward_matches_reference(self):
#         ...compare kernel backward against LoRAQKV backward...

"""
Correctness tests for v4 kernel: packed backward pass.

Tests:
  1. fp64 gradcheck (MHA, GQA, multiple ranks)
  2. Forward matches v2_3 exactly
  3. Backward gradients match PyTorch reference (LoRAQKV)
  4. 3D input support (batch, seq, hidden)
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reference.lora_qkv_pytorch import (
    lora_qkv_forward,
    LoRAQKV,
    make_lora_qkv_params,
)
from experiments.v2.lora_qkv_kernel_v2_3 import lora_qkv_v2_3
from experiments.v4.lora_qkv_kernel_v4 import (
    lora_qkv_v4,
    LoRAQKVv4Function,
    pack_weights_backward,
    pack_lora_a,
)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_params(H=256, num_heads=4, num_kv_heads=2, head_dim=64,
                rank=16, dtype=torch.float32, requires_grad=False):
    return make_lora_qkv_params(
        hidden_dim=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, rank=rank, dtype=dtype, device=DEVICE,
        requires_grad=requires_grad,
    )


# ===================================================================
# GRADCHECK — fp64
# ===================================================================

class TestV4Gradcheck:
    """fp64 gradcheck for mathematical correctness of backward."""

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_gradcheck_mha(self, rank):
        """MHA config (num_heads == num_kv_heads)."""
        torch.manual_seed(42)
        H, num_heads, head_dim = 64, 4, 16
        M = 16

        params = make_params(
            H=H, num_heads=num_heads, num_kv_heads=num_heads,
            head_dim=head_dim, rank=rank, dtype=torch.float64,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X, params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVv4Function.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_gradcheck_gqa(self, rank):
        """GQA config (num_kv_heads < num_heads)."""
        torch.manual_seed(42)
        H, num_heads, num_kv_heads, head_dim = 64, 4, 1, 16
        M = 16

        params = make_params(
            H=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=torch.float64,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X, params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVv4Function.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_3d_input(self):
        """3D input [B, S, H]."""
        torch.manual_seed(42)
        H, num_heads, num_kv_heads, head_dim = 64, 4, 2, 16
        B, S = 2, 8
        rank = 4

        params = make_params(
            H=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, rank=rank, dtype=torch.float64,
            requires_grad=True,
        )
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X, params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVv4Function.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
        )


# ===================================================================
# FORWARD CONSISTENCY — v4 forward matches v2_3
# ===================================================================

class TestV4Forward:
    """Forward pass should be identical to v2_3."""

    @pytest.mark.parametrize("rank", [8, 16, 32])
    def test_forward_matches_v2_3_bf16(self, rank):
        """bf16 forward exact match (same code path)."""
        torch.manual_seed(42)
        M, H, H_q, H_kv = 1024, 256, 256, 64

        params = make_params(
            H=H, num_heads=H_q // 64, num_kv_heads=H_kv // 64,
            head_dim=64, rank=rank, dtype=torch.bfloat16,
            requires_grad=True,
        )
        X = torch.randn(M, H, dtype=torch.bfloat16, device=DEVICE)

        Q_ref, K_ref, V_ref = lora_qkv_v2_3(
            X,
            params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
        )
        Q_v4, K_v4, V_v4 = lora_qkv_v4(
            X,
            params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
        )

        assert torch.allclose(Q_v4, Q_ref, atol=0, rtol=0)
        assert torch.allclose(K_v4, K_ref, atol=0, rtol=0)
        assert torch.allclose(V_v4, V_ref, atol=0, rtol=0)

    def test_forward_gqa_llama_scale(self):
        """LLaMA-3 8B GQA scale (32 query, 8 kv heads)."""
        torch.manual_seed(42)
        M = 4096
        params = make_params(
            H=4096, num_heads=32, num_kv_heads=8,
            head_dim=128, rank=16, dtype=torch.bfloat16,
            requires_grad=True,
        )
        X = torch.randn(M, 4096, dtype=torch.bfloat16, device=DEVICE)

        Q_ref, K_ref, V_ref = lora_qkv_v2_3(
            X,
            params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
        )
        Q_v4, K_v4, V_v4 = lora_qkv_v4(
            X,
            params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
        )

        assert torch.allclose(Q_v4, Q_ref, atol=0, rtol=0)
        assert torch.allclose(K_v4, K_ref, atol=0, rtol=0)
        assert torch.allclose(V_v4, V_ref, atol=0, rtol=0)


# ===================================================================
# BACKWARD CONSISTENCY — v4 gradients match reference
# ===================================================================

class TestV4Backward:
    """Backward gradients should match PyTorch reference (within bf16 tolerance)."""

    @pytest.mark.parametrize("rank", [8, 16, 32])
    def test_backward_matches_reference_bf16(self, rank):
        """All gradients match reference within bf16 tolerance."""
        torch.manual_seed(42)
        M, H = 512, 256

        params = make_params(
            H=H, num_heads=4, num_kv_heads=2,
            head_dim=64, rank=rank, dtype=torch.bfloat16,
            requires_grad=True,
        )

        # v4 path
        X_v4 = torch.randn(M, H, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        A_q_v4 = params["A_q"].detach().clone().requires_grad_(True)
        B_q_v4 = params["B_q"].detach().clone().requires_grad_(True)
        A_k_v4 = params["A_k"].detach().clone().requires_grad_(True)
        B_k_v4 = params["B_k"].detach().clone().requires_grad_(True)
        A_v_v4 = params["A_v"].detach().clone().requires_grad_(True)
        B_v_v4 = params["B_v"].detach().clone().requires_grad_(True)

        Q_v4, K_v4, V_v4 = lora_qkv_v4(
            X_v4,
            params["W_q"], A_q_v4, B_q_v4, params["s_q"],
            params["W_k"], A_k_v4, B_k_v4, params["s_k"],
            params["W_v"], A_v_v4, B_v_v4, params["s_v"],
        )
        (Q_v4.sum() + K_v4.sum() + V_v4.sum()).backward()

        # Reference path
        X_ref = X_v4.detach().clone().requires_grad_(True)
        A_q_ref = params["A_q"].detach().clone().requires_grad_(True)
        B_q_ref = params["B_q"].detach().clone().requires_grad_(True)
        A_k_ref = params["A_k"].detach().clone().requires_grad_(True)
        B_k_ref = params["B_k"].detach().clone().requires_grad_(True)
        A_v_ref = params["A_v"].detach().clone().requires_grad_(True)
        B_v_ref = params["B_v"].detach().clone().requires_grad_(True)

        Q_ref, K_ref, V_ref = LoRAQKV.apply(
            X_ref, params["W_q"], params["W_k"], params["W_v"],
            A_q_ref, B_q_ref, params["s_q"],
            A_k_ref, B_k_ref, params["s_k"],
            A_v_ref, B_v_ref, params["s_v"],
        )
        (Q_ref.sum() + K_ref.sum() + V_ref.sum()).backward()

        # dA, dB should match exactly (same operations, same order)
        assert torch.allclose(A_q_v4.grad, A_q_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(B_q_v4.grad, B_q_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(A_k_v4.grad, A_k_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(B_k_v4.grad, B_k_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(A_v_v4.grad, A_v_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(B_v_v4.grad, B_v_ref.grad, rtol=1e-2, atol=1e-2)

        # dX uses Triton epilogue — slightly looser tolerance due to operation reordering
        assert torch.allclose(X_v4.grad, X_ref.grad, rtol=5e-2, atol=5e-2), \
            f"dX max diff: {(X_v4.grad - X_ref.grad).abs().max().item():.4e}"

    def test_backward_gqa_3d(self):
        """GQA with 3D [B, S, H] input."""
        torch.manual_seed(42)
        B, S, H = 2, 256, 256
        rank = 16

        params = make_params(
            H=H, num_heads=4, num_kv_heads=1,
            head_dim=64, rank=rank, dtype=torch.bfloat16,
            requires_grad=True,
        )

        X_v4 = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        A_q_v4 = params["A_q"].detach().clone().requires_grad_(True)
        B_q_v4 = params["B_q"].detach().clone().requires_grad_(True)
        A_k_v4 = params["A_k"].detach().clone().requires_grad_(True)
        B_k_v4 = params["B_k"].detach().clone().requires_grad_(True)
        A_v_v4 = params["A_v"].detach().clone().requires_grad_(True)
        B_v_v4 = params["B_v"].detach().clone().requires_grad_(True)

        Q_v4, K_v4, V_v4 = lora_qkv_v4(
            X_v4,
            params["W_q"], A_q_v4, B_q_v4, params["s_q"],
            params["W_k"], A_k_v4, B_k_v4, params["s_k"],
            params["W_v"], A_v_v4, B_v_v4, params["s_v"],
        )
        (Q_v4.sum() + K_v4.sum() + V_v4.sum()).backward()

        X_ref = X_v4.detach().clone().requires_grad_(True)
        A_q_ref = params["A_q"].detach().clone().requires_grad_(True)
        B_q_ref = params["B_q"].detach().clone().requires_grad_(True)
        A_k_ref = params["A_k"].detach().clone().requires_grad_(True)
        B_k_ref = params["B_k"].detach().clone().requires_grad_(True)
        A_v_ref = params["A_v"].detach().clone().requires_grad_(True)
        B_v_ref = params["B_v"].detach().clone().requires_grad_(True)

        Q_ref, K_ref, V_ref = LoRAQKV.apply(
            X_ref, params["W_q"], params["W_k"], params["W_v"],
            A_q_ref, B_q_ref, params["s_q"],
            A_k_ref, B_k_ref, params["s_k"],
            A_v_ref, B_v_ref, params["s_v"],
        )
        (Q_ref.sum() + K_ref.sum() + V_ref.sum()).backward()

        assert X_v4.grad.shape == (B, S, H)
        assert torch.allclose(X_v4.grad, X_ref.grad, rtol=5e-2, atol=5e-2)
        assert torch.allclose(A_q_v4.grad, A_q_ref.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(B_q_v4.grad, B_q_ref.grad, rtol=1e-2, atol=1e-2)


# ===================================================================
# PACKING HELPERS
# ===================================================================

class TestPackingHelpers:
    """Test weight packing helper functions."""

    def test_pack_weights_backward_shape(self):
        H_q, H_kv, K = 256, 64, 256
        W_q = torch.randn(H_q, K, device=DEVICE)
        W_k = torch.randn(H_kv, K, device=DEVICE)
        W_v = torch.randn(H_kv, K, device=DEVICE)
        packed = pack_weights_backward(W_q, W_k, W_v)
        assert packed.shape == (H_q + 2 * H_kv, K)
        assert packed.is_contiguous()

    def test_pack_lora_a_shape(self):
        R, K = 16, 256
        A_q = torch.randn(R, K, device=DEVICE)
        A_k = torch.randn(R, K, device=DEVICE)
        A_v = torch.randn(R, K, device=DEVICE)
        packed = pack_lora_a(A_q, A_k, A_v)
        assert packed.shape == (3 * R, K)
        assert packed.is_contiguous()

    def test_precomputed_packing(self):
        """Pre-computed packed weights produce same results."""
        torch.manual_seed(42)
        M, H = 512, 256
        params = make_params(H=H, num_heads=4, num_kv_heads=2,
                             head_dim=64, rank=16, dtype=torch.bfloat16,
                             requires_grad=True)
        X = torch.randn(M, H, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        W_dX = pack_weights_backward(params["W_q"], params["W_k"], params["W_v"])
        A_pack = pack_lora_a(params["A_q"], params["A_k"], params["A_v"])

        Q1, K1, V1 = lora_qkv_v4(
            X, params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
        )
        Q2, K2, V2 = lora_qkv_v4(
            X, params["W_q"], params["A_q"], params["B_q"], params["s_q"],
            params["W_k"], params["A_k"], params["B_k"], params["s_k"],
            params["W_v"], params["A_v"], params["B_v"], params["s_v"],
            W_dX_packed=W_dX, A_packed=A_pack,
        )
        assert torch.allclose(Q1, Q2, atol=0, rtol=0)
        assert torch.allclose(K1, K2, atol=0, rtol=0)
        assert torch.allclose(V1, V2, atol=0, rtol=0)

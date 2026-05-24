"""
Correctness tests for v2_2, v2_3, and v3 kernel experiments.

Tests each version against the PyTorch reference implementation.
Covers: fp32, bf16, GQA, LLaMA-3 8B scale, and (for v3) gradcheck.
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
from experiments.v2.lora_qkv_kernel_v2_2 import (
    fused_lora_matmul_v2_2,
    lora_qkv_v2_2,
)
from experiments.v2.lora_qkv_kernel_v2_3 import (
    lora_qkv_v2_3,
    pack_weights_all,
)
from experiments.v3.lora_qkv_kernel_v3 import (
    lora_qkv_v3,
    LoRAQKVFunction,
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
# v2_2 Tests
# ===================================================================

class TestV2_2SingleProjection:
    """v2_2 per-projection tests."""

    def test_fp32_exact(self):
        torch.manual_seed(42)
        M, N, K, r = 128, 256, 512, 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02
        out = fused_lora_matmul_v2_2(X, W, A, B, 1.0)
        ref = matmul_lora(X, W, A, B, 1.0)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_bf16(self):
        torch.manual_seed(42)
        M, N, K, r = 128, 256, 512, 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.bfloat16) * 0.02
        out = fused_lora_matmul_v2_2(X, W, A, B, 1.0)
        ref = matmul_lora(X, W, A, B, 1.0)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=0.1)

    def test_no_lora(self):
        torch.manual_seed(42)
        M, N, K = 64, 128, 256
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        out = fused_lora_matmul_v2_2(X, W)
        ref = X @ W.t()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


class TestV2_2FullQKV:
    """v2_2 full QKV tests."""

    def test_fp32_gqa(self):
        torch.manual_seed(42)
        M = 128
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=8)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v2_2(X, **{k: v for k, v in params.items()})
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)

    def test_bf16_llama(self):
        torch.manual_seed(42)
        M = 8192
        params = make_params(H=4096, num_heads=32, num_kv_heads=8, head_dim=128,
                            rank=16, dtype=torch.bfloat16)
        X = torch.randn(M, 4096, device=DEVICE, dtype=torch.bfloat16)
        Q, K, V = lora_qkv_v2_2(X, **{k: v for k, v in params.items()})
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(K, K_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(V, V_ref, rtol=5e-2, atol=0.1)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_rank_sweep(self, rank):
        torch.manual_seed(42)
        M = 64
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=rank)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v2_2(X, **{k: v for k, v in params.items()})
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)


# ===================================================================
# v2_3 Tests
# ===================================================================

class TestV2_3FullQKV:
    """v2_3 full QKV tests (single cuBLAS)."""

    def test_fp32_gqa(self):
        torch.manual_seed(42)
        M = 128
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=8)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v2_3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)

    def test_bf16_llama(self):
        torch.manual_seed(42)
        M = 8192
        params = make_params(H=4096, num_heads=32, num_kv_heads=8, head_dim=128,
                            rank=16, dtype=torch.bfloat16)
        X = torch.randn(M, 4096, device=DEVICE, dtype=torch.bfloat16)
        Q, K, V = lora_qkv_v2_3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(K, K_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(V, V_ref, rtol=5e-2, atol=0.1)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_rank_sweep(self, rank):
        torch.manual_seed(42)
        M = 64
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=rank)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v2_3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)

    def test_prepacked_weights(self):
        torch.manual_seed(42)
        M = 64
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=8)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        W_all = pack_weights_all(
            params['W_q'], params['A_q'],
            params['W_k'], params['A_k'],
            params['W_v'], params['A_v'],
        )
        Q, K, V = lora_qkv_v2_3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
            W_all=W_all,
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)

    def test_no_lora(self):
        torch.manual_seed(42)
        M = 64
        H, H_q, H_kv = 256, 256, 64
        W_q = torch.randn(H_q, H, device=DEVICE, dtype=torch.float32) * 0.02
        W_k = torch.randn(H_kv, H, device=DEVICE, dtype=torch.float32) * 0.02
        W_v = torch.randn(H_kv, H, device=DEVICE, dtype=torch.float32) * 0.02
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v2_3(
            X, W_q, None, None, 1.0, W_k, None, None, 1.0, W_v, None, None, 1.0,
        )
        torch.testing.assert_close(Q, X @ W_q.t(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(K, X @ W_k.t(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(V, X @ W_v.t(), rtol=1e-5, atol=1e-5)


# ===================================================================
# v3 Tests
# ===================================================================

class TestV3Forward:
    """v3 forward pass tests (should match v2_3)."""

    def test_fp32_gqa(self):
        torch.manual_seed(42)
        M = 128
        params = make_params(H=256, num_heads=4, num_kv_heads=1, head_dim=64, rank=8)
        X = torch.randn(M, 256, device=DEVICE, dtype=torch.float32)
        Q, K, V = lora_qkv_v3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(K, K_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(V, V_ref, rtol=1e-3, atol=1e-3)

    def test_bf16_llama(self):
        torch.manual_seed(42)
        M = 8192
        params = make_params(H=4096, num_heads=32, num_kv_heads=8, head_dim=128,
                            rank=16, dtype=torch.bfloat16)
        X = torch.randn(M, 4096, device=DEVICE, dtype=torch.bfloat16)
        Q, K, V = lora_qkv_v3(
            X,
            params['W_q'], params['A_q'], params['B_q'], params['s_q'],
            params['W_k'], params['A_k'], params['B_k'], params['s_k'],
            params['W_v'], params['A_v'], params['B_v'], params['s_v'],
        )
        Q_ref, K_ref, V_ref = lora_qkv_forward(X, **{
            'W_q': params['W_q'], 'W_k': params['W_k'], 'W_v': params['W_v'],
            'A_q': params['A_q'], 'B_q': params['B_q'], 's_q': params['s_q'],
            'A_k': params['A_k'], 'B_k': params['B_k'], 's_k': params['s_k'],
            'A_v': params['A_v'], 'B_v': params['B_v'], 's_v': params['s_v'],
        })
        torch.testing.assert_close(Q, Q_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(K, K_ref, rtol=5e-2, atol=0.1)
        torch.testing.assert_close(V, V_ref, rtol=5e-2, atol=0.1)


class TestV3Backward:
    """v3 backward pass and gradcheck tests."""

    def test_gradcheck_fp64_mha(self):
        torch.manual_seed(42)
        M, H, r = 8, 32, 4
        params = make_params(H=H, num_heads=2, num_kv_heads=2, head_dim=16,
                            rank=r, dtype=torch.float64, requires_grad=True)
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVFunction.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    def test_gradcheck_fp64_gqa(self):
        torch.manual_seed(42)
        M, H, r = 8, 64, 4
        params = make_params(H=H, num_heads=4, num_kv_heads=1, head_dim=16,
                            rank=r, dtype=torch.float64, requires_grad=True)
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVFunction.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    def test_gradcheck_fp64_3d_input(self):
        torch.manual_seed(42)
        B, S, H, r = 2, 4, 32, 4
        params = make_params(H=H, num_heads=2, num_kv_heads=1, head_dim=16,
                            rank=r, dtype=torch.float64, requires_grad=True)
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVFunction.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_gradcheck_rank_sweep(self, rank):
        torch.manual_seed(42)
        M, H = 8, 32
        params = make_params(H=H, num_heads=2, num_kv_heads=1, head_dim=16,
                            rank=rank, dtype=torch.float64, requires_grad=True)
        X = torch.randn(M, H, dtype=torch.float64, device=DEVICE, requires_grad=True)
        inputs = (
            X,
            params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        assert torch.autograd.gradcheck(
            LoRAQKVFunction.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    def test_backward_matches_reference(self):
        """v3 backward gradients match PyTorch reference backward."""
        torch.manual_seed(42)
        M, H, r = 64, 128, 8
        params = make_params(H=H, num_heads=2, num_kv_heads=1, head_dim=64,
                            rank=r, dtype=torch.float32, requires_grad=True)

        # v3 backward
        X_v3 = torch.randn(M, H, device=DEVICE, dtype=torch.float32, requires_grad=True)
        p_v3 = {k: (v.clone().detach().requires_grad_(v.requires_grad)
                     if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        Q3, K3, V3 = LoRAQKVFunction.apply(
            X_v3, p_v3["W_q"], p_v3["W_k"], p_v3["W_v"],
            p_v3["A_q"], p_v3["B_q"], p_v3["s_q"],
            p_v3["A_k"], p_v3["B_k"], p_v3["s_k"],
            p_v3["A_v"], p_v3["B_v"], p_v3["s_v"],
        )
        (Q3.sum() + K3.sum() + V3.sum()).backward()

        # Reference backward
        X_ref = X_v3.detach().clone().requires_grad_(True)
        p_ref = {k: (v.clone().detach().requires_grad_(v.requires_grad)
                      if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        Qr, Kr, Vr = LoRAQKV.apply(
            X_ref, p_ref["W_q"], p_ref["W_k"], p_ref["W_v"],
            p_ref["A_q"], p_ref["B_q"], p_ref["s_q"],
            p_ref["A_k"], p_ref["B_k"], p_ref["s_k"],
            p_ref["A_v"], p_ref["B_v"], p_ref["s_v"],
        )
        (Qr.sum() + Kr.sum() + Vr.sum()).backward()

        torch.testing.assert_close(X_v3.grad, X_ref.grad, rtol=1e-3, atol=1e-3)
        for name in ["A_q", "B_q", "A_k", "B_k", "A_v", "B_v"]:
            torch.testing.assert_close(
                p_v3[name].grad, p_ref[name].grad,
                rtol=1e-3, atol=1e-3,
                msg=f"Gradient mismatch for {name}",
            )

    def test_backward_lora_scale(self):
        torch.manual_seed(42)
        M, H, r = 8, 32, 4
        params = make_params(H=H, num_heads=2, num_kv_heads=1, head_dim=16,
                            rank=r, dtype=torch.float64, requires_grad=True)
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
            LoRAQKVFunction.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    def test_gradient_shapes(self):
        torch.manual_seed(42)
        M, H, r = 64, 256, 16
        params = make_params(H=H, num_heads=4, num_kv_heads=1, head_dim=64,
                            rank=r, dtype=torch.float32, requires_grad=True)
        H_q = 4 * 64
        H_kv = 1 * 64
        X = torch.randn(M, H, device=DEVICE, dtype=torch.float32, requires_grad=True)
        Q, K, V = LoRAQKVFunction.apply(
            X, params["W_q"], params["W_k"], params["W_v"],
            params["A_q"], params["B_q"], params["s_q"],
            params["A_k"], params["B_k"], params["s_k"],
            params["A_v"], params["B_v"], params["s_v"],
        )
        (Q.sum() + K.sum() + V.sum()).backward()
        assert X.grad.shape == (M, H)
        assert params["A_q"].grad.shape == (r, H)
        assert params["B_q"].grad.shape == (H_q, r)
        assert params["A_k"].grad.shape == (r, H)
        assert params["B_k"].grad.shape == (H_kv, r)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Correctness tests for LoRA MLP fused Triton kernels.

Compares against:
  - Unsloth's exact code (reference/unsloth_baseline.py) for bf16 benchmarks
  - PyTorch reference (reference/lora_mlp_pytorch.py) for fp32 ground truth

Tests are organized by kernel version (v1, v2, ...) and cover:
  - Forward correctness (fp32 + bf16)
  - Multiple LoRA ranks (8, 16, 32, 64)
  - Multiple shapes (small, LLaMA-8B, LLaMA-13B scale)
  - 2D and 3D inputs
  - Edge cases (no LoRA, non-power-of-2 M/N/K)
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reference.lora_mlp_pytorch import matmul_lora, lora_swiglu_mlp, make_lora_mlp_params
from reference.unsloth_baseline import (
    matmul_lora as unsloth_matmul_lora,
    apply_lora_mlp_swiglu as unsloth_lora_mlp,
    make_lora_mlp_params as unsloth_make_params,
    swiglu_fg_kernel,
)
from experiments.v1.lora_mlp_kernel_v1 import fused_lora_matmul
from experiments.v2.lora_mlp_kernel_v2 import fused_gate_up_swiglu, lora_mlp_v2, LoRAMLPv2
from experiments.v3.lora_mlp_kernel_v3 import lora_mlp_v3, fused_lora_swiglu
from experiments.v5.lora_mlp_kernel_v5 import (
    lora_mlp_v5,
    LoRAMLPv5,
    lora_mlp_v5_inference,
    pack_gate_up_weights,
    pack_down_weights,
    merge_lora_weights,
    prepare_inference_weights,
)
from experiments.v5.lora_mlp_kernel_v5_upgrade_1 import (
    lora_mlp_v5_upgrade_1,
    LoRAMLPv5_upgrade_1,
    lora_mlp_v5_upgrade_1_inference,
    pack_gate_up_weights_padded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda"

SHAPES_SMALL = [
    (32, 64, 128),
    (64, 128, 256),
    (17, 33, 65),      # non-power-of-2
]

SHAPES_LLAMA = [
    (2048, 14336, 4096),   # LLaMA-8B: seq*batch=2048, I=14336, H=4096
    (4096, 14336, 4096),   # larger batch
    (2048, 17920, 5120),   # LLaMA-13B scale
]

RANKS = [8, 16, 32, 64]


def _ref_and_triton(X, W, A, B, s):
    """Run both reference and Triton, return results."""
    ref = matmul_lora(X, W, A, B, s)
    tri = fused_lora_matmul(X, W, A, B, s)
    return ref, tri


# ---------------------------------------------------------------------------
# v1: Fused LoRA Matmul — Forward Correctness
# ---------------------------------------------------------------------------

class TestV1FusedLoRAMatmul:
    """Tests for experiments/v1 fused_lora_matmul kernel."""

    # ── fp32 tests ──

    @pytest.mark.parametrize("M,N,K", SHAPES_SMALL)
    def test_base_matmul_fp32(self, M, N, K):
        """Base matmul without LoRA matches PyTorch."""
        torch.manual_seed(42)
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
        ref, tri = _ref_and_triton(X, W, None, None, 1.0)
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("rank", RANKS)
    def test_fused_lora_fp32_ranks(self, rank):
        """Fused LoRA matmul at various ranks in fp32."""
        torch.manual_seed(42)
        M, N, K = 128, 256, 512
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(rank, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, rank, device=DEVICE, dtype=torch.float32) * 0.02
        ref, tri = _ref_and_triton(X, W, A, B, 1.0)
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("M,N,K", SHAPES_SMALL)
    def test_fused_lora_fp32_shapes(self, M, N, K):
        """Fused LoRA matmul at various shapes in fp32."""
        torch.manual_seed(42)
        r = 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02
        ref, tri = _ref_and_triton(X, W, A, B, 1.0)
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)

    def test_3d_input_fp32(self):
        """3D input [B, S, H] works correctly."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16
        X = torch.randn(B, S, H, device=DEVICE, dtype=torch.float32)
        W = torch.randn(I, H, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, H, device=DEVICE, dtype=torch.float32) * 0.02
        Bm = torch.randn(I, r, device=DEVICE, dtype=torch.float32) * 0.02
        ref, tri = _ref_and_triton(X, W, A, Bm, 1.0)
        assert tri.shape == (B, S, I)
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)

    def test_lora_scale_fp32(self):
        """Non-unit LoRA scale is applied correctly."""
        torch.manual_seed(42)
        M, N, K, r = 64, 128, 256, 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02
        for s in [0.5, 2.0, 0.125]:
            ref, tri = _ref_and_triton(X, W, A, B, s)
            torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)

    # ── bf16 tests ──

    @pytest.mark.parametrize("rank", RANKS)
    def test_fused_lora_bf16_ranks(self, rank):
        """Fused LoRA matmul at various ranks in bf16.

        bf16 comparison uses relative tolerance since both implementations
        accumulate differently (both are close to the fp32 ground truth).
        """
        torch.manual_seed(42)
        M, N, K = 128, 256, 512
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(rank, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, rank, device=DEVICE, dtype=torch.float32) * 0.02

        # fp32 ground truth
        ref_fp32 = matmul_lora(X, W, A, B, 1.0)

        # Triton bf16
        tri_bf16 = fused_lora_matmul(
            X.bfloat16(), W.bfloat16(), A.bfloat16(), B.bfloat16(), 1.0
        )

        torch.testing.assert_close(
            tri_bf16.float(), ref_fp32, rtol=2e-2, atol=0.5
        )

    @pytest.mark.parametrize("M,N,K", SHAPES_SMALL)
    def test_fused_lora_bf16_shapes(self, M, N, K):
        """Fused LoRA matmul at various shapes in bf16."""
        torch.manual_seed(42)
        r = 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.float32) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.float32) * 0.02

        ref_fp32 = matmul_lora(X, W, A, B, 1.0)
        tri_bf16 = fused_lora_matmul(
            X.bfloat16(), W.bfloat16(), A.bfloat16(), B.bfloat16(), 1.0
        )
        torch.testing.assert_close(
            tri_bf16.float(), ref_fp32, rtol=2e-2, atol=0.5
        )

    # ── LLaMA-scale tests ──

    @pytest.mark.parametrize("M,N,K", SHAPES_LLAMA)
    def test_llama_scale_bf16(self, M, N, K):
        """Correctness at LLaMA-scale dimensions in bf16."""
        torch.manual_seed(42)
        r = 16
        X = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        A = torch.randn(r, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        B = torch.randn(N, r, device=DEVICE, dtype=torch.bfloat16) * 0.02

        ref_fp32 = matmul_lora(
            X.float(), W.float(), A.float(), B.float(), 1.0
        )
        tri = fused_lora_matmul(X, W, A, B, 1.0)
        torch.testing.assert_close(
            tri.float(), ref_fp32, rtol=2e-2, atol=1.0
        )


# ---------------------------------------------------------------------------
# v1: Full MLP using fused_lora_matmul for each projection
# ---------------------------------------------------------------------------

class TestV1FullMLP:
    """Full MLP forward using v1 fused_lora_matmul for each of gate/up/down."""

    def _run_mlp(self, X, params):
        """Run full MLP using fused_lora_matmul per projection + Unsloth's Triton SwiGLU."""
        e = fused_lora_matmul(X, params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"])
        g = fused_lora_matmul(X, params["W_up"], params["A_up"], params["B_up"], params["s_up"])
        h = swiglu_fg_kernel(e, g)
        out = fused_lora_matmul(h, params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        return out

    def test_full_mlp_fp32(self):
        """Full MLP forward matches reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        tri = self._run_mlp(X, params)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)

    def test_full_mlp_bf16(self):
        """Full MLP forward in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)
        X_bf16 = X_fp32.bfloat16()

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        tri = self._run_mlp(X_bf16, params_bf16)
        torch.testing.assert_close(tri.float(), ref, rtol=5e-2, atol=2.0)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_full_mlp_rank_sweep(self, rank):
        """Full MLP forward across LoRA ranks in fp32."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        params = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        tri = self._run_mlp(X, params)
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# v1: Direct comparison against Unsloth's exact code
# ---------------------------------------------------------------------------

class TestV1VsUnsloth:
    """Compare Triton v1 output directly against Unsloth's matmul_lora."""

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_matmul_lora_vs_unsloth_bf16(self, rank):
        """fused_lora_matmul matches Unsloth matmul_lora in bf16."""
        torch.manual_seed(42)
        M, N, K = 128, 256, 512
        X = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        W = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        A = torch.randn(rank, K, device=DEVICE, dtype=torch.bfloat16) * 0.02
        B = torch.randn(N, rank, device=DEVICE, dtype=torch.bfloat16) * 0.02

        unsloth_out = unsloth_matmul_lora(X, W, None, A, B, 1.0)
        triton_out = fused_lora_matmul(X, W, A, B, 1.0)

        # Both should be close to fp32 truth
        ref_fp32 = X.float() @ W.float().t() + (X.float() @ A.float().t()) @ B.float().t()
        torch.testing.assert_close(triton_out.float(), ref_fp32, rtol=2e-2, atol=0.5)
        torch.testing.assert_close(unsloth_out.float(), ref_fp32, rtol=2e-2, atol=0.5)

    def test_full_mlp_vs_unsloth_bf16(self):
        """Full MLP via Triton v1 matches Unsloth LoRA_MLP in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16

        # Unsloth params format
        u_params = unsloth_make_params(H, I, r, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        # Unsloth output
        unsloth_out = unsloth_lora_mlp(X, **u_params)

        # Triton v1 output (same weights)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        e = fused_lora_matmul(X, gp["W"], gp["A"], gp["B"], gp["s"])
        g = fused_lora_matmul(X, up["W"], up["A"], up["B"], up["s"])
        h = swiglu_fg_kernel(e, g)
        triton_out = fused_lora_matmul(h, dp["W"], dp["A"], dp["B"], dp["s"])

        # Both should produce same-magnitude results
        torch.testing.assert_close(
            triton_out.float(), unsloth_out.float(), rtol=5e-2, atol=2.0
        )


# ---------------------------------------------------------------------------
# v2: Fused Gate+Up+SwiGLU + cuBLAS Down
# ---------------------------------------------------------------------------

class TestV2FusedGateUpSwiGLU:
    """Tests for experiments/v2 fused gate+up+SwiGLU kernel."""

    def test_gate_up_swiglu_fp32(self):
        """Fused gate+up+SwiGLU matches separate ops in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        # Reference: separate gate + up + SiLU*up
        import torch.nn.functional as F
        e = matmul_lora(X, params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"])
        g = matmul_lora(X, params["W_up"], params["A_up"], params["B_up"], params["s_up"])
        h_ref = F.silu(e) * g

        h_tri, _, _ = fused_gate_up_swiglu(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
        )
        torch.testing.assert_close(h_tri, h_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("rank", [8, 16, 32])
    def test_gate_up_swiglu_ranks_bf16(self, rank):
        """Fused gate+up+SwiGLU across ranks in bf16 matches fp32 truth."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        params_fp32 = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        import torch.nn.functional as F
        e = matmul_lora(X_fp32, params_fp32["W_gate"], params_fp32["A_gate"], params_fp32["B_gate"], params_fp32["s_gate"])
        g = matmul_lora(X_fp32, params_fp32["W_up"], params_fp32["A_up"], params_fp32["B_up"], params_fp32["s_up"])
        h_truth = F.silu(e) * g

        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        h_tri, _, _ = fused_gate_up_swiglu(
            X_fp32.bfloat16(),
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
        )
        torch.testing.assert_close(h_tri.float(), h_truth, rtol=5e-2, atol=2.0)

    def test_full_mlp_v2_fp32(self):
        """Full MLP v2 matches reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v2(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_full_mlp_v2_bf16(self):
        """Full MLP v2 in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        out = lora_mlp_v2(
            X_fp32.bfloat16(),
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"],
        )
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    def test_full_mlp_v2_vs_unsloth(self):
        """v2 MLP matches Unsloth LoRA_MLP output in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16
        u_params = unsloth_make_params(H, I, r, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        unsloth_out = unsloth_lora_mlp(X, **u_params)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        v2_out = lora_mlp_v2(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )

        torch.testing.assert_close(v2_out.float(), unsloth_out.float(), rtol=5e-2, atol=2.0)


# ---------------------------------------------------------------------------
# v2: Backward Pass Tests
# ---------------------------------------------------------------------------

class TestV2Backward:
    """Backward pass tests for LoRAMLPv2."""

    def test_gradcheck_fp64(self):
        """torch.autograd.gradcheck passes in fp64 (numerical backward verification)."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 4, 16, 32, 4
        params = make_lora_mlp_params(H, I, r, dtype=torch.float64, device=DEVICE, requires_grad=True)
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert torch.autograd.gradcheck(LoRAMLPv2.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_matches_reference_fp32(self):
        """v2 gradients match the PyTorch reference backward in fp32."""
        from reference.lora_mlp_pytorch import LoRAMLP

        torch.manual_seed(42)
        B, S, H, I, r = 2, 32, 64, 128, 8
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=True)
        X_data = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        def _run_backward(fn_class, params_dict):
            X = X_data.clone().detach().requires_grad_(True)
            p = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v)
                 for k, v in params_dict.items()}
            out = fn_class.apply(
                X, p["W_gate"], p["A_gate"], p["B_gate"], p["s_gate"],
                p["W_up"], p["A_up"], p["B_up"], p["s_up"],
                p["W_down"], p["A_down"], p["B_down"], p["s_down"],
            )
            out.sum().backward()
            return X.grad, p["A_gate"].grad, p["B_gate"].grad, p["A_up"].grad, p["B_up"].grad, p["A_down"].grad, p["B_down"].grad

        ref_grads = _run_backward(LoRAMLP, params)
        v2_grads = _run_backward(LoRAMLPv2, params)

        names = ["dX", "dA_gate", "dB_gate", "dA_up", "dB_up", "dA_down", "dB_down"]
        for name, rg, vg in zip(names, ref_grads, v2_grads):
            # v2 forward uses Triton (tf32 for fp32), so backward grads differ slightly
            torch.testing.assert_close(vg, rg, rtol=5e-3, atol=5e-3, msg=f"{name} mismatch")

    @pytest.mark.parametrize("rank", [8, 16, 32])
    def test_backward_rank_sweep_fp32(self, rank):
        """Backward passes at various LoRA ranks in fp32."""
        from reference.lora_mlp_pytorch import LoRAMLP

        torch.manual_seed(42)
        B, S, H, I = 2, 16, 64, 128
        params = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=True)

        X_ref = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
        X_v2 = X_ref.clone().detach().requires_grad_(True)
        p_ref = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        p_v2 = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}

        out_ref = LoRAMLP.apply(X_ref, p_ref["W_gate"], p_ref["A_gate"], p_ref["B_gate"], p_ref["s_gate"],
            p_ref["W_up"], p_ref["A_up"], p_ref["B_up"], p_ref["s_up"],
            p_ref["W_down"], p_ref["A_down"], p_ref["B_down"], p_ref["s_down"])
        out_ref.sum().backward()

        out_v2 = LoRAMLPv2.apply(X_v2, p_v2["W_gate"], p_v2["A_gate"], p_v2["B_gate"], p_v2["s_gate"],
            p_v2["W_up"], p_v2["A_up"], p_v2["B_up"], p_v2["s_up"],
            p_v2["W_down"], p_v2["A_down"], p_v2["B_down"], p_v2["s_down"])
        out_v2.sum().backward()

        torch.testing.assert_close(X_v2.grad, X_ref.grad, rtol=1e-3, atol=1e-3, msg=f"dX mismatch at rank={rank}")
        torch.testing.assert_close(p_v2["A_gate"].grad, p_ref["A_gate"].grad, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# v3: cuBLAS matmuls + Triton fused LoRA-SwiGLU epilogue
# ---------------------------------------------------------------------------

class TestV3LoRAMLPv3:
    """Tests for v3 hybrid cuBLAS + Triton LoRA-SwiGLU kernel."""

    def test_full_mlp_fp32(self):
        """v3 full MLP matches reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_full_mlp_bf16(self):
        """v3 full MLP in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        out = lora_mlp_v3(X_fp32.bfloat16(),
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"])
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_full_mlp_rank_sweep(self, rank):
        """v3 works across LoRA ranks including r=64."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        params = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_vs_unsloth_bf16(self):
        """v3 matches Unsloth LoRA_MLP output in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16
        u_params = unsloth_make_params(H, I, r, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        unsloth_out = unsloth_lora_mlp(X, **u_params)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        v3_out = lora_mlp_v3(X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"])
        torch.testing.assert_close(v3_out.float(), unsloth_out.float(), rtol=5e-2, atol=2.0)

    # ── Fused LoRA-SwiGLU kernel (the Triton epilogue) in isolation ──

    def test_fused_lora_swiglu_fp32(self):
        """The Triton LoRA+SwiGLU epilogue kernel matches PyTorch ops."""
        torch.manual_seed(42)
        M, N, r = 256, 512, 16
        e = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        g = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        xa_gate = torch.randn(M, r, device=DEVICE, dtype=torch.float32)
        xa_up = torch.randn(M, r, device=DEVICE, dtype=torch.float32)
        B_gate = torch.randn(N, r, device=DEVICE, dtype=torch.float32)
        B_up = torch.randn(N, r, device=DEVICE, dtype=torch.float32)
        s = 1.0

        ref = torch.nn.functional.silu(e + s * (xa_gate @ B_gate.t())) * (g + s * (xa_up @ B_up.t()))
        out, _, _ = fused_lora_swiglu(e, g, xa_gate, xa_up, B_gate, B_up, s, s)
        # tl.dot uses tf32 for fp32 inputs (standard Triton behavior), so ~1e-2 precision
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=0.5)

    def test_fused_lora_swiglu_no_lora(self):
        """Epilogue kernel without LoRA is just SiLU(e) * g."""
        torch.manual_seed(42)
        M, N = 256, 512
        e = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        g = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

        ref = torch.nn.functional.silu(e) * g
        out, _, _ = fused_lora_swiglu(e, g, None, None, None, None)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 0.125])
    def test_fused_lora_swiglu_scales(self, scale):
        """LoRA scaling is applied correctly."""
        torch.manual_seed(42)
        M, N, r = 128, 256, 8
        e = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        g = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        xa_gate = torch.randn(M, r, device=DEVICE, dtype=torch.float32)
        xa_up = torch.randn(M, r, device=DEVICE, dtype=torch.float32)
        B_gate = torch.randn(N, r, device=DEVICE, dtype=torch.float32)
        B_up = torch.randn(N, r, device=DEVICE, dtype=torch.float32)

        ref = torch.nn.functional.silu(e + scale * (xa_gate @ B_gate.t())) * (g + scale * (xa_up @ B_up.t()))
        out, _, _ = fused_lora_swiglu(e, g, xa_gate, xa_up, B_gate, B_up, scale, scale)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=1.0)

    # ── Sequence length sweep ──

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_v3_seq_len_sweep_bf16(self, seq_len):
        """v3 works across different sequence lengths."""
        torch.manual_seed(42)
        B, H, I, r = 1, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, seq_len, H, dtype=torch.bfloat16, device=DEVICE)

        ref_fp32 = lora_swiglu_mlp(
            X.float(),
            **{k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        )
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out.float(), ref_fp32, rtol=5e-2, atol=2.0)

    # ── Batch size sweep ──

    @pytest.mark.parametrize("batch", [1, 2, 4, 8])
    def test_v3_batch_sweep_bf16(self, batch):
        """v3 works across different batch sizes."""
        torch.manual_seed(42)
        S, H, I, r = 64, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(batch, S, H, dtype=torch.bfloat16, device=DEVICE)

        ref_fp32 = lora_swiglu_mlp(
            X.float(),
            **{k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        )
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out.float(), ref_fp32, rtol=5e-2, atol=2.0)

    # ── LLaMA-scale shapes ──

    @pytest.mark.parametrize("H,I", [(4096, 14336), (5120, 17920)])
    def test_v3_llama_shapes_bf16(self, H, I):
        """v3 works at LLaMA-8B and LLaMA-13B dimensions."""
        torch.manual_seed(42)
        B, S, r = 1, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        ref_fp32 = lora_swiglu_mlp(
            X.float(),
            **{k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        )
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out.float(), ref_fp32, rtol=5e-2, atol=2.0)

    # ── Non-power-of-2 dimensions ──

    def test_v3_non_power_of_2(self):
        """v3 handles non-power-of-2 hidden/intermediate dims."""
        torch.manual_seed(42)
        B, S, H, I, r = 3, 37, 97, 193, 7
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    # ── 2D input (no batch dimension) ──

    def test_v3_2d_input(self):
        """v3 works with 2D input [M, H] (no batch/seq dims)."""
        torch.manual_seed(42)
        M, H, I, r = 128, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(M, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X.unsqueeze(0), **params).squeeze(0)
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    # ── Determinism: same input → same output ──

    def test_v3_deterministic(self):
        """v3 produces identical output on repeated calls."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        def run():
            return lora_mlp_v3(X,
                params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
                params["W_up"], params["A_up"], params["B_up"], params["s_up"],
                params["W_down"], params["A_down"], params["B_down"], params["s_down"])

        out1 = run()
        out2 = run()
        out3 = run()
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)
        torch.testing.assert_close(out2, out3, rtol=0, atol=0)

    # ── No NaN / Inf in output ──

    def test_v3_no_nan_inf(self):
        """v3 output contains no NaN or Inf values."""
        torch.manual_seed(42)
        B, S, H, I, r = 4, 256, 256, 512, 32
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    # ── Large LoRA scale stress test ──

    def test_v3_large_lora_scale(self):
        """v3 handles large LoRA scales without overflow."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 32, 128, 256, 8
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params["s_gate"] = 10.0
        params["s_up"] = 10.0
        params["s_down"] = 10.0
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE) * 0.1

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v3(X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    # ── v3 vs Unsloth at LLaMA-8B scale ──

    def test_v3_vs_unsloth_llama_scale(self):
        """v3 matches Unsloth at full LLaMA-8B scale."""
        torch.manual_seed(42)
        u_params = unsloth_make_params(4096, 14336, 16, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(1, 512, 4096, dtype=torch.bfloat16, device=DEVICE)

        unsloth_out = unsloth_lora_mlp(X, **u_params)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        v3_out = lora_mlp_v3(X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"])

        rel_err = ((v3_out.float() - unsloth_out.float()).abs() / (unsloth_out.float().abs() + 1e-8)).mean()
        assert rel_err < 0.15, f"Mean relative error vs Unsloth too high: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# v5: Packed cuBLAS Matmuls + Triton LoRA-SwiGLU Epilogue (Training & Inference)
# ---------------------------------------------------------------------------

class TestV5:
    """Tests for v5 packed cuBLAS LoRA MLP (training + inference paths)."""

    # ── Training forward (packed mega-GEMM + Triton epilogue + packed down) ──

    def test_packed_forward_fp32(self):
        """v5 packed forward matches PyTorch reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_packed_forward_bf16(self):
        """v5 packed forward in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        out = lora_mlp_v5(
            X_fp32.bfloat16(),
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"],
        )
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_packed_rank_sweep(self, rank):
        """v5 packed forward across LoRA ranks in fp32."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        params = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_packed_vs_unsloth_bf16(self):
        """v5 packed forward matches Unsloth apply_lora_mlp_swiglu in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16
        u_params = unsloth_make_params(H, I, r, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        unsloth_out = unsloth_lora_mlp(X, **u_params)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        v5_out = lora_mlp_v5(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )
        torch.testing.assert_close(v5_out.float(), unsloth_out.float(), rtol=5e-2, atol=2.0)

    # ── Inference forward (pre-merged weights + cublasLt SWISH or Triton fallback) ──

    def test_inference_fp32(self):
        """v5 inference path with pre-merged weights matches reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        W_gate_eff_T, W_up_eff_T, W_down_eff_T = prepare_inference_weights(
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        out = lora_mlp_v5_inference(X, W_gate_eff_T, W_up_eff_T, W_down_eff_T)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_inference_bf16(self):
        """v5 inference path with pre-merged weights in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        W_gate_eff_T, W_up_eff_T, W_down_eff_T = prepare_inference_weights(
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"],
        )
        out = lora_mlp_v5_inference(X_fp32.bfloat16(), W_gate_eff_T, W_up_eff_T, W_down_eff_T)
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    # ── Backward correctness (fp64 gradcheck + bf16 vs reference) ──

    def test_backward_gradcheck(self):
        """torch.autograd.gradcheck passes in fp64 for LoRAMLPv5 on small inputs."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 4, 16, 32, 4
        params = make_lora_mlp_params(H, I, r, dtype=torch.float64, device=DEVICE, requires_grad=True)
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert torch.autograd.gradcheck(LoRAMLPv5.apply, inputs, eps=1e-6, atol=1e-4)

    def test_backward_bf16(self):
        """v5 LoRA gradients match the PyTorch reference backward in bf16 (loose tolerances)."""
        from reference.lora_mlp_pytorch import LoRAMLP

        torch.manual_seed(42)
        B, S, H, I, r = 2, 32, 128, 256, 8
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        X_data = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        def _run_backward(fn_class, params_dict):
            X = X_data.clone().detach().requires_grad_(True)
            p = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v)
                 for k, v in params_dict.items()}
            out = fn_class.apply(
                X, p["W_gate"], p["A_gate"], p["B_gate"], p["s_gate"],
                p["W_up"], p["A_up"], p["B_up"], p["s_up"],
                p["W_down"], p["A_down"], p["B_down"], p["s_down"],
            )
            out.sum().backward()
            return X.grad, p["A_gate"].grad, p["B_gate"].grad, p["A_up"].grad, p["B_up"].grad, p["A_down"].grad, p["B_down"].grad

        ref_grads = _run_backward(LoRAMLP, params)
        v5_grads = _run_backward(LoRAMLPv5, params)

        names = ["dX", "dA_gate", "dB_gate", "dA_up", "dB_up", "dA_down", "dB_down"]
        for name, rg, vg in zip(names, ref_grads, v5_grads):
            torch.testing.assert_close(vg.float(), rg.float(), rtol=5e-2, atol=2.0, msg=f"{name} mismatch")

    # ── Edge cases ──

    def test_v5_no_lora(self):
        """v5 with A/B = None (no LoRA) still works."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)
        W_gate = torch.randn(I, H, dtype=torch.float32, device=DEVICE) * 0.02
        W_up = torch.randn(I, H, dtype=torch.float32, device=DEVICE) * 0.02
        W_down = torch.randn(H, I, dtype=torch.float32, device=DEVICE) * 0.02

        ref = lora_swiglu_mlp(X, W_gate, None, None, 1.0, W_up, None, None, 1.0, W_down, None, None, 1.0)
        out = lora_mlp_v5(
            X,
            W_gate, None, None, 1.0,
            W_up, None, None, 1.0,
            W_down, None, None, 1.0,
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_non_power_of_2(self):
        """v5 handles non-power-of-2 dimensions."""
        torch.manual_seed(42)
        B, S, H, I, r = 3, 37, 97, 193, 7
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_3d_input(self):
        """v5 works with 3D input [B, S, H]."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert out.shape == (B, S, H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_2d_input(self):
        """v5 works with 2D input [M, H]."""
        torch.manual_seed(42)
        M, H, I, r = 128, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(M, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X.unsqueeze(0), **params).squeeze(0)
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert out.shape == (M, H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("H,I", [(4096, 14336), (5120, 17920)])
    def test_v5_llama_shapes_bf16(self, H, I):
        """v5 packed forward at LLaMA-8B and LLaMA-13B dimensions in bf16."""
        torch.manual_seed(42)
        B, S, r = 1, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        ref_fp32 = lora_swiglu_mlp(
            X.float(),
            **{k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in params.items()},
        )
        out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out.float(), ref_fp32, rtol=5e-2, atol=2.0)


# ---------------------------------------------------------------------------
# v5_upgrade_1: padded gate+up packing + dropped down packing
# ---------------------------------------------------------------------------

class TestV5Upgrade1:
    """Tests for v5_upgrade_1 (padded gate+up mega-GEMM, v3-style down)."""

    # ── Training forward ──

    def test_packed_forward_fp32(self):
        """v5_upgrade_1 forward matches PyTorch reference in fp32."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_packed_forward_bf16(self):
        """v5_upgrade_1 forward in bf16 close to fp32 ground truth."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        out = lora_mlp_v5_upgrade_1(
            X_fp32.bfloat16(),
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"],
        )
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_packed_rank_sweep(self, rank):
        """v5_upgrade_1 forward across LoRA ranks in fp32."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        params = make_lora_mlp_params(H, I, rank, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_packed_vs_unsloth_bf16(self):
        """v5_upgrade_1 forward matches Unsloth apply_lora_mlp_swiglu in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 256, 512, 16
        u_params = unsloth_make_params(H, I, r, dtype=torch.bfloat16, device=DEVICE)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        unsloth_out = unsloth_lora_mlp(X, **u_params)
        gp, up, dp = u_params["gate_proj"], u_params["up_proj"], u_params["down_proj"]
        out = lora_mlp_v5_upgrade_1(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )
        torch.testing.assert_close(out.float(), unsloth_out.float(), rtol=5e-2, atol=2.0)

    # ── Inference (re-export of v5; should match v5 byte-for-byte) ──

    def test_inference_bf16(self):
        """v5_upgrade_1 inference (re-exported from v5) matches reference in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 128, 256, 512, 16
        params_fp32 = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        params_bf16 = {k: (v.bfloat16() if isinstance(v, torch.Tensor) else v) for k, v in params_fp32.items()}
        X_fp32 = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X_fp32, **params_fp32)
        W_gate_eff_T, W_up_eff_T, W_down_eff_T = prepare_inference_weights(
            params_bf16["W_gate"], params_bf16["A_gate"], params_bf16["B_gate"], params_bf16["s_gate"],
            params_bf16["W_up"], params_bf16["A_up"], params_bf16["B_up"], params_bf16["s_up"],
            params_bf16["W_down"], params_bf16["A_down"], params_bf16["B_down"], params_bf16["s_down"],
        )
        out = lora_mlp_v5_upgrade_1_inference(X_fp32.bfloat16(), W_gate_eff_T, W_up_eff_T, W_down_eff_T)
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=2.0)

    # ── Backward correctness (fp64 gradcheck + bf16 vs reference) ──

    def test_backward_gradcheck(self):
        """torch.autograd.gradcheck passes in fp64 for LoRAMLPv5_upgrade_1 on small inputs."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 4, 16, 32, 4
        params = make_lora_mlp_params(H, I, r, dtype=torch.float64, device=DEVICE, requires_grad=True)
        X = torch.randn(B, S, H, dtype=torch.float64, device=DEVICE, requires_grad=True)

        inputs = (
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert torch.autograd.gradcheck(LoRAMLPv5_upgrade_1.apply, inputs, eps=1e-6, atol=1e-4)

    def test_backward_bf16(self):
        """v5_upgrade_1 LoRA gradients match the PyTorch reference backward in bf16."""
        from reference.lora_mlp_pytorch import LoRAMLP

        torch.manual_seed(42)
        B, S, H, I, r = 2, 32, 128, 256, 8
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        X_data = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        def _run_backward(fn_class, params_dict):
            X = X_data.clone().detach().requires_grad_(True)
            p = {k: (v.clone().detach().requires_grad_(v.requires_grad) if isinstance(v, torch.Tensor) else v)
                 for k, v in params_dict.items()}
            out = fn_class.apply(
                X, p["W_gate"], p["A_gate"], p["B_gate"], p["s_gate"],
                p["W_up"], p["A_up"], p["B_up"], p["s_up"],
                p["W_down"], p["A_down"], p["B_down"], p["s_down"],
            )
            out.sum().backward()
            return X.grad, p["A_gate"].grad, p["B_gate"].grad, p["A_up"].grad, p["B_up"].grad, p["A_down"].grad, p["B_down"].grad

        ref_grads = _run_backward(LoRAMLP, params)
        up1_grads = _run_backward(LoRAMLPv5_upgrade_1, params)

        names = ["dX", "dA_gate", "dB_gate", "dA_up", "dB_up", "dA_down", "dB_down"]
        for name, rg, ug in zip(names, ref_grads, up1_grads):
            torch.testing.assert_close(ug.float(), rg.float(), rtol=5e-2, atol=2.0, msg=f"{name} mismatch")

    # ── Edge cases ──

    def test_v5_upgrade_1_no_lora(self):
        """v5_upgrade_1 with A/B = None (no LoRA) still works."""
        torch.manual_seed(42)
        B, S, H, I = 2, 64, 128, 256
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)
        W_gate = torch.randn(I, H, dtype=torch.float32, device=DEVICE) * 0.02
        W_up = torch.randn(I, H, dtype=torch.float32, device=DEVICE) * 0.02
        W_down = torch.randn(H, I, dtype=torch.float32, device=DEVICE) * 0.02

        ref = lora_swiglu_mlp(X, W_gate, None, None, 1.0, W_up, None, None, 1.0, W_down, None, None, 1.0)
        out = lora_mlp_v5_upgrade_1(
            X,
            W_gate, None, None, 1.0,
            W_up, None, None, 1.0,
            W_down, None, None, 1.0,
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_upgrade_1_non_power_of_2(self):
        """v5_upgrade_1 handles non-power-of-2 dimensions (and arbitrary pad amounts)."""
        torch.manual_seed(42)
        B, S, H, I, r = 3, 37, 97, 193, 7
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_upgrade_1_3d_input(self):
        """v5_upgrade_1 works with 3D input [B, S, H]."""
        torch.manual_seed(42)
        B, S, H, I, r = 2, 64, 128, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X, **params)
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert out.shape == (B, S, H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_v5_upgrade_1_2d_input(self):
        """v5_upgrade_1 works with 2D input [M, H]."""
        torch.manual_seed(42)
        M, H, I, r = 128, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.float32, device=DEVICE, requires_grad=False)
        X = torch.randn(M, H, dtype=torch.float32, device=DEVICE)

        ref = lora_swiglu_mlp(X.unsqueeze(0), **params).squeeze(0)
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        assert out.shape == (M, H)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("H,I", [(4096, 14336), (5120, 17920)])
    def test_v5_upgrade_1_llama_shapes_bf16(self, H, I):
        """v5_upgrade_1 forward at LLaMA-8B and LLaMA-13B dimensions in bf16."""
        torch.manual_seed(42)
        B, S, r = 1, 256, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        ref_fp32 = lora_swiglu_mlp(
            X.float(),
            **{k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in params.items()},
        )
        out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        torch.testing.assert_close(out.float(), ref_fp32, rtol=5e-2, atol=2.0)

    # ── v5_upgrade_1-specific: padding alignment ──

    @pytest.mark.parametrize("rank", [8, 16, 32, 64])
    def test_padded_alignment(self, rank):
        """The packed mega-matrix's row count (= output N) is divisible by 128."""
        H, I = 4096, 14336
        W_gate = torch.zeros(I, H, dtype=torch.bfloat16, device=DEVICE)
        W_up = torch.zeros(I, H, dtype=torch.bfloat16, device=DEVICE)
        A_gate = torch.zeros(rank, H, dtype=torch.bfloat16, device=DEVICE)
        A_up = torch.zeros(rank, H, dtype=torch.bfloat16, device=DEVICE)

        W_mega_padded, pad_rows = pack_gate_up_weights_padded(W_gate, W_up, A_gate, A_up)

        n_unpadded = 2 * I + 2 * rank
        assert W_mega_padded.shape[0] % 128 == 0, (
            f"Padded N={W_mega_padded.shape[0]} not divisible by 128"
        )
        assert W_mega_padded.shape[0] - pad_rows == n_unpadded
        assert pad_rows < 128
        # Padding rows must be all zeros
        if pad_rows > 0:
            tail = W_mega_padded[-pad_rows:]
            assert tail.abs().sum().item() == 0.0

    def test_padded_correctness_matches_v5(self):
        """Padded mega doesn't corrupt the output: v5 and v5_upgrade_1 match in bf16."""
        torch.manual_seed(42)
        B, S, H, I, r = 1, 512, 256, 512, 16
        params = make_lora_mlp_params(H, I, r, dtype=torch.bfloat16, device=DEVICE, requires_grad=False)
        X = torch.randn(B, S, H, dtype=torch.bfloat16, device=DEVICE)

        v5_out = lora_mlp_v5(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        up1_out = lora_mlp_v5_upgrade_1(
            X,
            params["W_gate"], params["A_gate"], params["B_gate"], params["s_gate"],
            params["W_up"], params["A_up"], params["B_up"], params["s_up"],
            params["W_down"], params["A_down"], params["B_down"], params["s_down"],
        )
        # bf16 tolerance: both should hit the same fp32 ground truth, but
        # accumulation order differs slightly between addmm_ on a packed slice
        # vs addmm_ on a fresh contig buffer. Same tolerance band as
        # `test_packed_forward_bf16`.
        torch.testing.assert_close(up1_out.float(), v5_out.float(), rtol=5e-2, atol=2.0)

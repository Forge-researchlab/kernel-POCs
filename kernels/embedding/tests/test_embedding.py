"""
Embedding Kernel Correctness Tests
====================================

Tests for all Forge embedding kernel variants against PyTorch reference.

Run:
    pytest tests/test_embedding.py -v
    pytest tests/test_embedding.py -v -k "backward"   # just backward tests
"""

import pytest
import torch
import sys
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v1.embedding_kernel_v1_upgrade_1 import ForgeEmbeddingFunction


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[(32000, 768), (64000, 4096)], ids=["32k_768", "64k_4096"])
def vocab_dim(request):
    return request.param


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------
class TestForward:
    def test_basic_lookup(self, dtype, vocab_dim):
        vocab_size, emb_dim = vocab_dim
        weight = torch.randn(vocab_size, emb_dim, device=DEVICE, dtype=dtype)
        indices = torch.randint(0, vocab_size, (4, 128), device=DEVICE)

        expected = torch.nn.functional.embedding(indices, weight)
        result = ForgeEmbeddingFunction.apply(weight.clone(), indices)

        rtol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_single_token(self):
        weight = torch.randn(100, 64, device=DEVICE)
        indices = torch.tensor([42], device=DEVICE)

        expected = weight[42].unsqueeze(0)
        result = ForgeEmbeddingFunction.apply(weight, indices)
        torch.testing.assert_close(result, expected)

    def test_all_same_index(self):
        weight = torch.randn(100, 128, device=DEVICE)
        indices = torch.full((512,), 7, device=DEVICE, dtype=torch.long)

        expected = weight[7].unsqueeze(0).expand(512, -1)
        result = ForgeEmbeddingFunction.apply(weight, indices)
        torch.testing.assert_close(result, expected)

    def test_2d_indices(self):
        weight = torch.randn(1000, 256, device=DEVICE)
        indices = torch.randint(0, 1000, (8, 64), device=DEVICE)

        expected = torch.nn.functional.embedding(indices, weight)
        result = ForgeEmbeddingFunction.apply(weight, indices)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------
class TestBackward:
    def test_gradient_matches_pytorch(self, dtype, vocab_dim):
        vocab_size, emb_dim = vocab_dim
        weight_ref = torch.randn(vocab_size, emb_dim, device=DEVICE, dtype=dtype, requires_grad=True)
        weight_forge = weight_ref.detach().clone().requires_grad_(True)
        indices = torch.randint(0, vocab_size, (4, 128), device=DEVICE)

        out_ref = torch.nn.functional.embedding(indices, weight_ref)
        out_ref.sum().backward()

        out_forge = ForgeEmbeddingFunction.apply(weight_forge, indices)
        out_forge.sum().backward()

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(weight_forge.grad, weight_ref.grad, rtol=rtol, atol=atol)

    def test_backward_high_duplicates(self):
        """Stress test: 500 unique tokens, 65 duplicates each -> cooperative path."""
        vocab_size, emb_dim = 32000, 768
        n_unique = 500
        dups_per = 65
        seq_len = n_unique * dups_per

        unique_ids = torch.randint(0, vocab_size, (n_unique,), device=DEVICE)
        indices = unique_ids.repeat_interleave(dups_per)

        weight_ref = torch.randn(vocab_size, emb_dim, device=DEVICE, requires_grad=True)
        weight_forge = weight_ref.detach().clone().requires_grad_(True)

        out_ref = torch.nn.functional.embedding(indices, weight_ref)
        out_ref.sum().backward()

        out_forge = ForgeEmbeddingFunction.apply(weight_forge, indices)
        out_forge.sum().backward()

        torch.testing.assert_close(weight_forge.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)

    def test_backward_all_unique(self):
        """All unique tokens — no duplicate accumulation needed."""
        vocab_size, emb_dim = 32000, 768
        indices = torch.arange(0, 1024, device=DEVICE)

        weight_ref = torch.randn(vocab_size, emb_dim, device=DEVICE, requires_grad=True)
        weight_forge = weight_ref.detach().clone().requires_grad_(True)

        out_ref = torch.nn.functional.embedding(indices, weight_ref)
        out_ref.sum().backward()

        out_forge = ForgeEmbeddingFunction.apply(weight_forge, indices)
        out_forge.sum().backward()

        torch.testing.assert_close(weight_forge.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)

    def test_backward_small_input_index_add_path(self):
        """< SORT_BACKWARD_THRESHOLD elements — uses index_add_ fallback."""
        vocab_size, emb_dim = 1000, 128
        indices = torch.randint(0, vocab_size, (64,), device=DEVICE)

        weight_ref = torch.randn(vocab_size, emb_dim, device=DEVICE, requires_grad=True)
        weight_forge = weight_ref.detach().clone().requires_grad_(True)

        out_ref = torch.nn.functional.embedding(indices, weight_ref)
        out_ref.sum().backward()

        out_forge = ForgeEmbeddingFunction.apply(weight_forge, indices)
        out_forge.sum().backward()

        torch.testing.assert_close(weight_forge.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Gradcheck (fp64 numerical gradient verification)
# ---------------------------------------------------------------------------
class TestGradcheck:
    @pytest.mark.slow
    def test_gradcheck_small(self):
        vocab_size, emb_dim = 16, 8
        weight = torch.randn(vocab_size, emb_dim, device=DEVICE, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([0, 3, 3, 7, 15], device=DEVICE)

        def fn(w):
            return ForgeEmbeddingFunction.apply(w, indices)

        assert torch.autograd.gradcheck(fn, (weight,), eps=1e-6, atol=1e-4)

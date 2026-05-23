from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v1 import forge_cross_entropy  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _supports_bfloat16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 5e-2, 1e-2
    if dtype == torch.float16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _make_case(
    batch: int,
    seq_len: int,
    vocab: int,
    dtype: torch.dtype,
    scalar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Liger's tests use B, T, V shapes but pass cross entropy flattened as
    # (B*T, V). Forge follows that same convention because language-model logits
    # are usually flattened before the loss kernel.
    torch.manual_seed(0)
    logits = torch.randn(batch * seq_len, vocab, device=DEVICE, dtype=dtype) * scalar
    logits_ref = logits.detach().clone().requires_grad_(True)
    logits_forge = logits.detach().clone().requires_grad_(True)
    target = torch.randint(0, vocab, (batch * seq_len,), device=DEVICE, dtype=torch.long)
    return logits_ref, logits_forge, target


def _apply_ignore_index(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    # Deterministic ignore positions make failures reproducible while still
    # exercising the branch Liger optimizes by exiting early for ignored rows.
    target = target.clone()
    target[::3] = ignore_index
    return target


def _assert_forward_backward_match(
    logits_ref: torch.Tensor,
    logits_forge: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    weight: torch.Tensor | None = None,
    upstream: torch.Tensor | None = None,
) -> None:
    expected = F.cross_entropy(
        logits_ref,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    actual = forge_cross_entropy(
        logits_forge,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    rtol, atol = _tolerances(logits_ref.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    grad = upstream if upstream is not None else torch.ones_like(expected)
    expected.backward(gradient=grad)
    actual.backward(gradient=grad)
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch,seq_len,vocab",
    [
        (2, 8, 128),
        (3, 7, 257),  # odd shape to catch power-of-two block and mask bugs
        pytest.param(2, 64, 4096, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not _supports_bfloat16(), reason="bfloat16 is not supported on this GPU"),
        ),
    ],
)
def test_correctness_matches_torch(batch: int, seq_len: int, vocab: int, reduction: str, dtype: torch.dtype) -> None:
    logits_ref, logits_forge, target = _make_case(batch, seq_len, vocab, dtype)
    _assert_forward_backward_match(logits_ref, logits_forge, target, reduction=reduction)


@pytest.mark.parametrize(
    "batch,seq_len,vocab,ignore_index",
    [
        (2, 8, 128, -100),
        (3, 7, 257, -123),
        (2, 16, 512, 2),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ignore_index_matches_torch(batch: int, seq_len: int, vocab: int, ignore_index: int, reduction: str) -> None:
    logits_ref, logits_forge, target = _make_case(batch, seq_len, vocab, torch.float32)
    target = _apply_ignore_index(target, ignore_index)
    _assert_forward_backward_match(
        logits_ref,
        logits_forge,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
    )


@pytest.mark.parametrize("label_smoothing", [0.1, 0.2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not _supports_bfloat16(), reason="bfloat16 is not supported on this GPU"),
        ),
    ],
)
def test_label_smoothing_matches_torch(label_smoothing: float, reduction: str, dtype: torch.dtype) -> None:
    logits_ref, logits_forge, target = _make_case(2, 8, 129, dtype)
    _assert_forward_backward_match(
        logits_ref,
        logits_forge,
        target,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


@pytest.mark.parametrize("ignore_index,label_smoothing", [(-100, 0.1), (3, 0.2)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_label_smoothing_with_ignore_index_matches_torch(
    ignore_index: int,
    label_smoothing: float,
    reduction: str,
) -> None:
    logits_ref, logits_forge, target = _make_case(3, 5, 97, torch.float32)
    target = _apply_ignore_index(target, ignore_index)
    _assert_forward_backward_match(
        logits_ref,
        logits_forge,
        target,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_not_last_layer_grad_output_matches_torch(reduction: str) -> None:
    logits_ref, logits_forge, target = _make_case(2, 6, 113, torch.float32)

    expected = F.cross_entropy(logits_ref, target, reduction=reduction)
    actual = forge_cross_entropy(logits_forge, target, reduction=reduction)

    loss_ref = expected * 3.0
    loss_forge = actual * 3.0
    upstream = torch.rand_like(loss_ref)
    loss_ref.backward(gradient=upstream)
    loss_forge.backward(gradient=upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward_only_no_grad_matches_torch(reduction: str, dtype: torch.dtype) -> None:
    logits_ref, logits_forge, target = _make_case(2, 5, 64, dtype)
    logits_ref = logits_ref.detach()
    logits_forge = logits_forge.detach()

    with torch.no_grad():
        expected = F.cross_entropy(logits_ref, target, reduction=reduction)
        actual = forge_cross_entropy(logits_forge, target, reduction=reduction)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert not actual.requires_grad

    with pytest.raises(RuntimeError, match="does not require grad"):
        actual.backward(gradient=torch.ones_like(actual))


@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_all_ignored_rows_have_zero_gradient(reduction: str) -> None:
    logits_ref, logits_forge, target = _make_case(2, 4, 32, torch.float32)
    target.fill_(-100)
    _assert_forward_backward_match(logits_ref, logits_forge, target, reduction=reduction, ignore_index=-100)


def test_all_ignored_mean_matches_torch_nan_value_and_zero_gradient() -> None:
    logits_ref, logits_forge, target = _make_case(2, 4, 32, torch.float32)
    target.fill_(-100)

    expected = F.cross_entropy(logits_ref, target, ignore_index=-100, reduction="mean")
    actual = forge_cross_entropy(logits_forge, target, ignore_index=-100, reduction="mean")

    assert torch.isnan(expected)
    assert torch.isnan(actual)

    expected.backward()
    actual.backward()
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad_target", [-1, 128])
def test_out_of_bounds_target_raises(bad_target: int) -> None:
    logits = torch.randn(8, 128, device=DEVICE, requires_grad=True)
    target = torch.randint(0, 128, (8,), device=DEVICE)
    target[2] = bad_target

    with pytest.raises(IndexError, match="out of bounds"):
        forge_cross_entropy(logits, target)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_weight_fallback_matches_torch(reduction: str) -> None:
    logits_ref, logits_forge, target = _make_case(2, 4, 73, torch.float32)
    weight = torch.rand(73, device=DEVICE)
    _assert_forward_backward_match(
        logits_ref,
        logits_forge,
        target,
        reduction=reduction,
        weight=weight,
    )


def test_cpu_path_uses_torch_compatibility_fallback() -> None:
    logits_ref = torch.randn(5, 19, requires_grad=True)
    logits_forge = logits_ref.detach().clone().requires_grad_(True)
    target = torch.randint(0, 19, (5,))

    _assert_forward_backward_match(logits_ref, logits_forge, target)

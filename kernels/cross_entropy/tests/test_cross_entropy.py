from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v2 import CrossEntropyOutput  # noqa: E402
from experiments.v2 import forge_cross_entropy  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Forge cross entropy v2 requires CUDA")


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


def test_all_ignored_mean_matches_liger_zero_value_and_zero_gradient() -> None:
    logits_ref, logits_forge, target = _make_case(2, 4, 32, torch.float32)
    target.fill_(-100)

    expected = torch.zeros((), device=DEVICE, dtype=logits_ref.dtype)
    actual = forge_cross_entropy(logits_forge, target, ignore_index=-100, reduction="mean")

    torch.testing.assert_close(actual, expected)

    actual.backward()
    torch.testing.assert_close(logits_forge.grad, torch.zeros_like(logits_forge), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad_target", [-1, 128])
def test_out_of_bounds_target_raises(bad_target: int) -> None:
    logits = torch.randn(8, 128, device=DEVICE, requires_grad=True)
    target = torch.randint(0, 128, (8,), device=DEVICE)
    target[2] = bad_target

    with pytest.raises((AssertionError, IndexError), match="out of bounds"):
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


def _softcap_reference(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits.to(torch.float32) / softcap)


def _cross_entropy_with_z_loss_reference(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_for_loss = _softcap_reference(logits, softcap)
    ce_loss = F.cross_entropy(
        logits_for_loss,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    target_mask = target != ignore_index
    z_loss = torch.where(
        target_mask,
        lse_square_scale * torch.logsumexp(logits_for_loss, dim=-1) ** 2,
        0.0,
    ).to(logits.dtype)
    if reduction == "mean":
        z_loss = z_loss.sum() / target_mask.sum()
    elif reduction == "sum":
        z_loss = z_loss.sum()

    return ce_loss.to(logits.dtype) + z_loss, z_loss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("return_z_loss", [False, True])
def test_z_loss_matches_torch_reference(reduction: str, return_z_loss: bool) -> None:
    logits_ref, logits_forge, target = _make_case(2, 5, 67, torch.float32)
    expected, expected_z = _cross_entropy_with_z_loss_reference(
        logits_ref,
        target,
        lse_square_scale=1e-4,
        reduction=reduction,
    )
    actual = forge_cross_entropy(
        logits_forge,
        target,
        lse_square_scale=1e-4,
        reduction=reduction,
        return_z_loss=return_z_loss,
    )

    actual_loss = actual.loss if isinstance(actual, CrossEntropyOutput) else actual
    torch.testing.assert_close(actual_loss, expected, rtol=1e-5, atol=1e-5)
    if return_z_loss:
        assert isinstance(actual, CrossEntropyOutput)
        torch.testing.assert_close(actual.z_loss, expected_z, rtol=1e-5, atol=1e-5)

    actual_loss.backward(gradient=torch.ones_like(actual_loss))
    expected.backward(gradient=torch.ones_like(expected))
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("softcap", [30.0, 40.0])
def test_softcap_matches_torch_reference(reduction: str, softcap: float) -> None:
    logits_ref, logits_forge, target = _make_case(2, 5, 67, torch.float32, scalar=3.0)
    expected, _ = _cross_entropy_with_z_loss_reference(
        logits_ref,
        target,
        reduction=reduction,
        softcap=softcap,
    )
    actual = forge_cross_entropy(logits_forge, target, reduction=reduction, softcap=softcap)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.backward(gradient=torch.ones_like(actual))
    expected.backward(gradient=torch.ones_like(expected))
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=1e-5, atol=1e-5)


def test_weight_with_label_smoothing_and_ignore_index_matches_reference() -> None:
    logits_ref, logits_forge, target = _make_case(2, 5, 83, torch.float32)
    target = _apply_ignore_index(target, -100)
    weight = torch.rand(83, device=DEVICE)
    expected, _ = _cross_entropy_with_z_loss_reference(
        logits_ref,
        target,
        weight=weight,
        ignore_index=-100,
        label_smoothing=0.1,
        reduction="mean",
    )
    actual = forge_cross_entropy(
        logits_forge,
        target,
        weight=weight,
        ignore_index=-100,
        label_smoothing=0.1,
        reduction="mean",
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits_forge.grad, logits_ref.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("return_z_loss", [False, True])
def test_token_accuracy_and_predicted_tokens_match_torch(return_z_loss: bool) -> None:
    logits = torch.randn(13, 41, device=DEVICE, requires_grad=True)
    target = torch.randint(0, 41, (13,), device=DEVICE)
    target[::4] = -100
    predicted = logits.detach().argmax(dim=-1)
    predicted = torch.where(target == -100, torch.full_like(predicted, -1), predicted)
    expected_accuracy = ((predicted == target) & (target != -100)).float().sum() / (target != -100).sum()

    actual = forge_cross_entropy(
        logits,
        target,
        ignore_index=-100,
        lse_square_scale=1e-4,
        return_z_loss=return_z_loss,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )

    assert isinstance(actual, CrossEntropyOutput)
    torch.testing.assert_close(actual.token_accuracy, expected_accuracy)
    torch.testing.assert_close(actual.predicted_tokens, predicted)


@pytest.mark.parametrize("return_z_loss", [False, True])
@pytest.mark.parametrize("return_token_accuracy", [False, True])
@pytest.mark.parametrize("return_predicted_tokens", [False, True])
def test_full_feature_surface_matches_liger_public_wrapper(
    return_z_loss: bool,
    return_token_accuracy: bool,
    return_predicted_tokens: bool,
) -> None:
    liger_ce_module = pytest.importorskip("liger_kernel.transformers.cross_entropy")
    liger_ce_cls = liger_ce_module.LigerCrossEntropyLoss

    logits = torch.randn(17, 89, device=DEVICE, requires_grad=True)
    logits_forge = logits.detach().clone().requires_grad_(True)
    logits_liger = logits.detach().clone().requires_grad_(True)
    target = torch.randint(0, 89, (17,), device=DEVICE)
    target[::5] = -100
    weight = torch.rand(89, device=DEVICE)

    kwargs = {
        "weight": weight,
        "ignore_index": -100,
        "lse_square_scale": 1e-4,
        "label_smoothing": 0.1,
        "reduction": "mean",
        "softcap": 30.0,
        "return_z_loss": return_z_loss,
        "return_token_accuracy": return_token_accuracy,
        "return_predicted_tokens": return_predicted_tokens,
    }
    forge_out = forge_cross_entropy(logits_forge, target, **kwargs)
    liger_out = liger_ce_cls(**kwargs)(logits_liger, target)

    forge_loss = forge_out.loss if isinstance(forge_out, CrossEntropyOutput) else forge_out
    liger_loss = liger_out.loss if hasattr(liger_out, "loss") else liger_out
    torch.testing.assert_close(forge_loss, liger_loss, rtol=1e-5, atol=1e-5)

    if return_z_loss:
        torch.testing.assert_close(forge_out.z_loss, liger_out.z_loss, rtol=1e-5, atol=1e-5)
    if return_token_accuracy:
        torch.testing.assert_close(forge_out.token_accuracy, liger_out.token_accuracy, rtol=1e-5, atol=1e-5)
    if return_predicted_tokens:
        torch.testing.assert_close(forge_out.predicted_tokens, liger_out.predicted_tokens)

    forge_loss.backward()
    liger_loss.backward()
    torch.testing.assert_close(logits_forge.grad, logits_liger.grad, rtol=1e-5, atol=1e-5)

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL_ROOT))

from experiments.v2 import CrossEntropyOutput  # noqa: E402
from experiments.v2 import ForgeFusedLinearCrossEntropyLoss  # noqa: E402
from experiments.v2 import forge_fused_linear_cross_entropy  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Forge fused linear CE requires CUDA")


def _supports_bfloat16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 5e-2, 2e-2
    if dtype == torch.float16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _make_case(
    bt: int,
    hidden: int,
    vocab: int,
    dtype: torch.dtype = torch.float32,
    *,
    bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(123)
    base_input = torch.randn(bt, hidden, device=DEVICE, dtype=dtype)
    base_weight = (torch.randn(vocab, hidden, device=DEVICE, dtype=dtype) / hidden**0.5).contiguous()
    base_bias = torch.randn(vocab, device=DEVICE, dtype=dtype) if bias else None
    target = torch.randint(0, vocab, (bt,), device=DEVICE)

    input_ref = base_input.detach().clone().requires_grad_(True)
    weight_ref = base_weight.detach().clone().requires_grad_(True)
    bias_ref = base_bias.detach().clone().requires_grad_(True) if base_bias is not None else None
    input_forge = base_input.detach().clone().requires_grad_(True)
    weight_forge = base_weight.detach().clone().requires_grad_(True)
    bias_forge = base_bias.detach().clone().requires_grad_(True) if base_bias is not None else None
    return input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge


def _softcap_reference(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits.to(torch.float32) / softcap)


def _loss_reference(
    _input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    target: torch.Tensor,
    *,
    ce_weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits = _softcap_reference(F.linear(_input, weight, bias), softcap)
    loss = F.cross_entropy(
        logits,
        target,
        weight=ce_weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    z_loss = None
    if lse_square_scale:
        target_mask = target != ignore_index
        z_loss = torch.where(
            target_mask,
            lse_square_scale * torch.logsumexp(logits, dim=-1) ** 2,
            0.0,
        ).to(logits.dtype)
        if reduction == "mean":
            z_loss = z_loss.sum() / target_mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        loss = loss + z_loss
    return loss, z_loss


def _assert_grads_close(
    input_forge: torch.Tensor,
    input_ref: torch.Tensor,
    weight_forge: torch.Tensor,
    weight_ref: torch.Tensor,
    bias_forge: torch.Tensor | None,
    bias_ref: torch.Tensor | None,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(input_forge.grad, input_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(weight_forge.grad, weight_ref.grad, rtol=rtol, atol=atol)
    if bias_ref is not None:
        assert bias_forge is not None
        torch.testing.assert_close(bias_forge.grad, bias_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(10, 32, 73), (21, 48, 129)])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_linear_cross_entropy_matches_torch(shape: tuple[int, int, int], reduction: str, bias: bool) -> None:
    bt, hidden, vocab = shape
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        bt, hidden, vocab, bias=bias
    )
    target[::4] = -100

    expected, _ = _loss_reference(input_ref, weight_ref, bias_ref, target, ignore_index=-100, reduction=reduction)
    actual = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        ignore_index=-100,
        reduction=reduction,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref)


@pytest.mark.parametrize("label_smoothing", [0.1, 0.2])
def test_fused_linear_cross_entropy_label_smoothing_matches_torch(label_smoothing: float) -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        17, 40, 67, bias=True
    )

    expected, _ = _loss_reference(
        input_ref,
        weight_ref,
        bias_ref,
        target,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    actual = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        label_smoothing=label_smoothing,
        reduction="mean",
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref)


def test_fused_linear_cross_entropy_class_weight_matches_torch() -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        13, 36, 59, bias=True
    )
    target[::3] = -100
    ce_weight = torch.rand(59, device=DEVICE)

    expected, _ = _loss_reference(
        input_ref,
        weight_ref,
        bias_ref,
        target,
        ce_weight=ce_weight,
        ignore_index=-100,
        reduction="mean",
    )
    actual = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        ce_weight=ce_weight,
        ignore_index=-100,
        reduction="mean",
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref)


@pytest.mark.parametrize("return_z_loss", [False, True])
def test_fused_linear_cross_entropy_softcap_z_loss_matches_reference(return_z_loss: bool) -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        11, 28, 53, bias=True
    )
    expected, expected_z = _loss_reference(
        input_ref,
        weight_ref,
        bias_ref,
        target,
        lse_square_scale=1e-4,
        softcap=30.0,
        reduction="mean",
    )
    actual = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        lse_square_scale=1e-4,
        softcap=30.0,
        reduction="mean",
        return_z_loss=return_z_loss,
    )

    actual_loss = actual.loss if isinstance(actual, CrossEntropyOutput) else actual
    torch.testing.assert_close(actual_loss, expected, rtol=1e-5, atol=1e-5)
    if return_z_loss:
        assert isinstance(actual, CrossEntropyOutput)
        torch.testing.assert_close(actual.z_loss, expected_z, rtol=1e-5, atol=1e-5)

    expected.backward()
    actual_loss.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref)


def test_fused_linear_cross_entropy_metrics_match_torch_argmax() -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        13, 24, 41, bias=True
    )
    target[::5] = -100
    logits = F.linear(input_ref.detach(), weight_ref.detach(), bias_ref.detach())
    predicted = logits.argmax(dim=-1)
    predicted = torch.where(target == -100, torch.full_like(predicted, -1), predicted)
    expected_accuracy = ((predicted == target) & (target != -100)).float().sum() / (target != -100).sum()

    actual = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )

    assert isinstance(actual, CrossEntropyOutput)
    torch.testing.assert_close(actual.token_accuracy, expected_accuracy)
    torch.testing.assert_close(actual.predicted_tokens, predicted)
    actual.loss.backward()
    assert input_forge.grad is not None
    assert weight_forge.grad is not None
    assert bias_forge is not None and bias_forge.grad is not None


def test_fused_linear_cross_entropy_module_liger_call_order() -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        9, 20, 37, bias=True
    )
    module = ForgeFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")

    expected, _ = _loss_reference(input_ref, weight_ref, bias_ref, target, ignore_index=-100, reduction="mean")
    actual = module(weight_forge, input_forge, target, bias_forge)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref)


def test_fused_linear_cross_entropy_forward_only_no_grad_matches_torch() -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        8, 24, 47, bias=True
    )
    input_ref = input_ref.detach()
    weight_ref = weight_ref.detach()
    bias_ref = bias_ref.detach() if bias_ref is not None else None
    input_forge = input_forge.detach()
    weight_forge = weight_forge.detach()
    bias_forge = bias_forge.detach() if bias_forge is not None else None

    with torch.no_grad():
        expected, _ = _loss_reference(input_ref, weight_ref, bias_ref, target, reduction="mean")
        actual = forge_fused_linear_cross_entropy(input_forge, weight_forge, target, bias=bias_forge, reduction="mean")

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not actual.requires_grad


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
def test_fused_linear_cross_entropy_dtype_smoke(dtype: torch.dtype) -> None:
    input_ref, weight_ref, bias_ref, target, input_forge, weight_forge, bias_forge = _make_case(
        7, 32, 61, dtype=dtype, bias=True
    )

    expected, _ = _loss_reference(input_ref, weight_ref, bias_ref, target, reduction="mean")
    actual = forge_fused_linear_cross_entropy(input_forge, weight_forge, target, bias=bias_forge, reduction="mean")

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, check_dtype=False)
    expected.backward()
    actual.backward()
    _assert_grads_close(input_forge, input_ref, weight_forge, weight_ref, bias_forge, bias_ref, dtype=dtype)


def test_fused_linear_cross_entropy_matches_liger_public_wrapper_when_available() -> None:
    liger_module = pytest.importorskip("liger_kernel.transformers.fused_linear_cross_entropy")
    liger_cls = liger_module.LigerFusedLinearCrossEntropyLoss

    _, _, _, target, input_forge, weight_forge, bias_forge = _make_case(15, 32, 71, bias=True)
    input_liger = input_forge.detach().clone().requires_grad_(True)
    weight_liger = weight_forge.detach().clone().requires_grad_(True)
    bias_liger = bias_forge.detach().clone().requires_grad_(True)
    target[::4] = -100

    kwargs = {
        "ignore_index": -100,
        "label_smoothing": 0.1,
        "reduction": "mean",
        "return_token_accuracy": True,
        "return_predicted_tokens": True,
    }
    forge_out = forge_fused_linear_cross_entropy(
        input_forge,
        weight_forge,
        target,
        bias=bias_forge,
        **kwargs,
    )
    liger_out = liger_cls(**kwargs)(weight_liger, input_liger, target, bias_liger)

    assert isinstance(forge_out, CrossEntropyOutput)
    forge_loss = forge_out.loss
    liger_loss = liger_out.loss if hasattr(liger_out, "loss") else liger_out
    torch.testing.assert_close(forge_loss, liger_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(forge_out.token_accuracy, liger_out.token_accuracy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(forge_out.predicted_tokens, liger_out.predicted_tokens)

    forge_loss.backward()
    liger_loss.backward()
    torch.testing.assert_close(input_forge.grad, input_liger.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight_forge.grad, weight_liger.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(bias_forge.grad, bias_liger.grad, rtol=1e-5, atol=1e-5)


def test_fused_linear_cross_entropy_rejects_none_reduction() -> None:
    _, _, _, target, input_forge, weight_forge, _ = _make_case(5, 12, 19)
    with pytest.raises(AssertionError, match="reduction='mean' or 'sum'"):
        forge_fused_linear_cross_entropy(input_forge, weight_forge, target, reduction="none")

"""Forge fused linear + cross entropy v2.

Chunked ``linear + cross_entropy`` implementation that mirrors Liger's fused
linear CE algorithm while reusing Forge's CE v2 Triton kernel. It computes
``F.cross_entropy(F.linear(hidden, weight, bias), target)`` without
materializing the full ``B*T*V`` logits tensor at once.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from .cross_entropy_kernel_v2 import CrossEntropyOutput
from .cross_entropy_kernel_v2 import MAX_FUSED_SIZE
from .cross_entropy_kernel_v2 import _element_mul_kernel
from .cross_entropy_kernel_v2 import forge_cross_entropy_kernel


__all__ = [
    "ForgeFusedLinearCrossEntropyFunction",
    "ForgeFusedLinearCrossEntropyLoss",
    "fused_linear_cross_entropy_forward",
    "fused_linear_cross_entropy_backward",
    "forge_fused_linear_cross_entropy",
]


def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    """Chunked ``linear + cross_entropy`` without materializing full logits.

    ``_input`` is flattened hidden states with shape ``(B*T, H)`` and
    ``weight`` is the LM-head/classifier matrix with shape ``(V, H)``. Each
    chunk materializes only ``chunk_size x V`` logits, immediately reuses the
    CE Triton kernel to turn that chunk into ``dLoss/dLogits`` in-place, and
    accumulates gradients for input/weight/bias. This mirrors Liger's fused
    linear CE algorithm while reusing Forge's CE v2 kernel.
    """
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )

    if _input.ndim != 2:
        raise ValueError(f"_input must be 2D (B*T, H); got shape={tuple(_input.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D (V, H); got shape={tuple(weight.shape)}")
    if target.ndim != 1:
        raise ValueError(f"target must be 1D (B*T,); got shape={tuple(target.shape)}")

    device = _input.device
    input_requires_grad = _input.requires_grad

    BT, H = _input.shape
    V, weight_h = weight.shape
    if weight_h != H:
        raise ValueError(f"weight.shape[1] must match _input.shape[1]; got {weight_h} != {H}")
    if target.shape[0] != BT:
        raise ValueError(f"target length must match _input rows; got {target.shape[0]} != {BT}")
    if bias is not None and bias.shape != (V,):
        raise ValueError(f"bias must have shape ({V},); got {tuple(bias.shape)}")

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    if n_non_ignore:
        target_values = target.masked_select(target_mask)
        assert target_values.max() < V, f"Target {target_values.max()} is out of bounds. Expected < {V}"
        assert target_values.min() >= 0, f"Target {target_values.min()} is out of bounds. Expected >= 0"

    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, ce_weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, ce_weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        sum_non_ignore_weight = torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()
    if bias is not None and bias.stride(-1) != 1:
        bias = bias.contiguous()

    grad_input = torch.zeros_like(_input, device=device)
    if input_requires_grad:
        if accum_dtype is None:
            grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
        else:
            grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None

    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    # Choose a token chunk so the temporary logits chunk is roughly comparable
    # to the hidden-state chunk instead of the full B*T*V logits tensor.
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        input_chunk = _input[start_idx:end_idx]
        target_chunk = target[start_idx:end_idx]

        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias

        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)
            probs = torch.softmax(logits_for_softmax, dim=-1)
            valid_mask = target_chunk != ignore_index
            scaling_factors = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)
            if valid_mask.any():
                valid_targets = target_chunk[valid_mask]
                scaling_factors[valid_mask] = torch.gather(
                    probs[valid_mask], -1, valid_targets.unsqueeze(-1)
                ).squeeze(-1)
            scaling_factors = scaling_factors.detach()

        n_rows = logits_chunk.shape[0]
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()
        loss_1d_slice = loss_1d[start_idx:end_idx]
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        forge_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),
            token_accuracy_ptr=token_accuracy_1d_slice,
            token_accuracy_stride=token_accuracy_1d_slice.stride(-1) if return_token_accuracy else 0,
            predicted_tokens_ptr=predicted_tokens_1d_slice,
            predicted_tokens_stride=predicted_tokens_1d_slice.stride(-1) if return_predicted_tokens else 0,
            n_cols=V,
            n_non_ignore=n_non_ignore,
            sum_non_ignore_weight=sum_non_ignore_weight,
            weight_sum=weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            RETURN_PREDICTED_TOKENS=return_predicted_tokens,
            BLOCK_SIZE=block_size,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            HAS_GRADIENTS=input_requires_grad,
            num_warps=32,
        )

        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            loss_1d[start_idx:end_idx] = loss_1d_slice
            if return_z_loss:
                z_loss_1d_slice = z_loss_1d_slice * scaling_factors
                z_loss_1d[start_idx:end_idx] = z_loss_1d_slice

        grad_logits_chunk = logits_chunk
        if use_token_scaling:
            grad_logits_chunk = grad_logits_chunk * scaling_factors.unsqueeze(-1)

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
        if grad_weight is not None and input_requires_grad:
            grad_weight += torch.mm(grad_logits_chunk.t(), input_chunk).float()
        if grad_bias is not None and input_requires_grad:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / n_non_ignore if return_token_accuracy else None
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None
    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    grad_bias: Optional[torch.Tensor],
):
    if grad_output.ndim > 0:
        raise NotImplementedError("Forge fused linear CE backward supports scalar losses only; use mean/sum reduction.")

    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device, dtype=grad_output.dtype)):
        tensors = [t for t in (grad_input, grad_weight) if t is not None]
        for tensor in tensors:
            n_rows, n_cols = tensor.shape
            block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
            _element_mul_kernel[(n_rows,)](
                tensor,
                tensor.stride(-2),
                grad_output,
                n_cols,
                BLOCK_SIZE=block_size,
                num_warps=32,
            )
        if grad_bias is not None:
            block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(grad_bias.shape[0]))
            _element_mul_kernel[(1,)](
                grad_bias.unsqueeze(0),
                grad_bias.shape[0],
                grad_output,
                grad_bias.shape[0],
                BLOCK_SIZE=block_size,
                num_warps=32,
            )
    return grad_input, grad_weight, grad_bias


class ForgeFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ce_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                ce_weight=ce_weight,
                bias=bias,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
        )
        saved_tensors = []
        ctx.has_grad_input = grad_input is not None
        ctx.has_grad_weight = grad_weight is not None
        ctx.has_grad_bias = grad_bias is not None
        if ctx.has_grad_input:
            saved_tensors.append(grad_input.detach())
        if ctx.has_grad_weight:
            saved_tensors.append(grad_weight.detach())
        if ctx.has_grad_bias:
            saved_tensors.append(grad_bias.detach())
        ctx.save_for_backward(*saved_tensors)
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4
        saved_iter = iter(ctx.saved_tensors)
        grad_input = next(saved_iter) if ctx.has_grad_input else None
        grad_weight = next(saved_iter) if ctx.has_grad_weight else None
        grad_bias = next(saved_iter) if ctx.has_grad_bias else None
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def forge_fused_linear_cross_entropy(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    """Functional fused linear + cross entropy API.

    Equivalent to ``F.cross_entropy(F.linear(_input, weight, bias), target, ...)``
    for mean/sum reductions, but computes logits in token chunks and stores only
    precomputed gradients for the autograd backward.
    """
    loss, z_loss, token_accuracy, predicted_tokens = ForgeFusedLinearCrossEntropyFunction.apply(
        _input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    if not return_z_loss and not return_token_accuracy and not return_predicted_tokens:
        return loss

    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


class ForgeFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {"mean", "sum", "none"}, (
            f"Forge fused linear CE supports reduction='mean', 'sum', or 'none'. Got: {reduction}"
        )
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling
        self.return_token_accuracy = return_token_accuracy
        self.return_predicted_tokens = return_predicted_tokens

    def forward(
        self,
        weight: torch.Tensor,
        _input: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """Mirror Liger's module call order: ``loss(weight, input, target)``."""
        return forge_fused_linear_cross_entropy(
            _input,
            weight,
            target,
            bias=bias,
            ce_weight=self.ce_weight,
            ignore_index=self.ignore_index,
            lse_square_scale=self.lse_square_scale,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
            softcap=self.softcap,
            return_z_loss=self.return_z_loss,
            accum_dtype=self.accum_dtype,
            use_token_scaling=self.use_token_scaling,
            return_token_accuracy=self.return_token_accuracy,
            return_predicted_tokens=self.return_predicted_tokens,
        )

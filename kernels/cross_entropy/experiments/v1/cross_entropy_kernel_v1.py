"""
Forge Cross Entropy v1.

This file is intentionally close to the structure of Liger's cross entropy op,
but the names, feature surface, and comments are Forge-specific. The important
Liger idea to preserve is not the exact API; it is the execution plan:

1. Launch one Triton program per flattened token row, where each row has V
   vocabulary logits.
2. Compute logsumexp with an online softmax pass. Liger uses this because it is
   numerically stable without materializing a full softmax tensor.
3. When gradients are required, overwrite the logits buffer with dLoss/dLogits
   inside the forward Triton kernel. Liger takes this route because the usual
   PyTorch decomposition creates large intermediate tensors for logits,
   log-softmax, and gradients. Reusing the input storage is the core memory win.
4. Save that gradient buffer for autograd backward. Backward then only needs to
   multiply by the upstream grad_output, and can skip even that when the loss is
   the final scalar and grad_output is exactly 1.

Forge v1 keeps the benchmark-facing feature set small: hard-label cross entropy
with ignore_index, reduction in {mean, sum, none}, and label smoothing. Liger
also supports class weights, z-loss, softcapping, token accuracy, and predicted
tokens. Those are good follow-up extensions, but keeping them out of this first
Forge kernel makes the POC easier to validate against Torch and Liger baselines.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton can represent blocks much larger than this, but Liger caps the fused
# row block below Triton's hard tensor limit to reduce register pressure. This
# same cap is a good first Forge default: large enough for common vocab chunks,
# small enough to avoid obvious spilling on A100/H100-class GPUs.
MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def _forge_cross_entropy_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    loss_ptr,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    has_gradients: tl.constexpr,
    block_size: tl.constexpr,
):
    # Liger casts program_id to int64 before pointer arithmetic. That avoids
    # int32 overflow when BT * stride becomes large for long sequences or huge
    # vocabularies.
    row_id = tl.program_id(0).to(tl.int64)

    row_target_ptr = target_ptr + row_id * target_stride
    target = tl.load(row_target_ptr)

    row_logits_ptr = logits_ptr + row_id * logits_stride
    row_loss_ptr = loss_ptr + row_id

    # Matching PyTorch ignore_index semantics:
    # - ignored rows contribute zero unreduced loss
    # - ignored rows have zero gradient
    # Liger exits early here as well, which saves the online softmax work for
    # padding tokens that should not affect the objective.
    if target == ignore_index:
        tl.store(row_loss_ptr, 0.0)
        if has_gradients:
            for col_start in range(0, n_cols, block_size):
                offsets = col_start + tl.arange(0, block_size)
                tl.store(row_logits_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    # Online softmax state. The variables use the same names as the classic
    # online-normalizer algorithm: m is the running max, d is the running sum of
    # exp(x - m). This gives stable logsumexp while loading each row in chunks.
    m = float("-inf")
    d = 0.0

    # We need the original target logit for CE = logsumexp(x) - x[target].
    target_logit = tl.load(row_logits_ptr + target).cast(tl.float32)

    # Label smoothing adds eps * sum(-log_softmax(x_i)) across all classes.
    # Liger accumulates the x_i part during the online pass so it does not need
    # a separate sweep over the row. We do the same here.
    eps = label_smoothing / n_cols
    smoothed_negative_logit_sum = 0.0

    for col_start in range(0, n_cols, block_size):
        offsets = col_start + tl.arange(0, block_size)
        mask = offsets < n_cols
        x = tl.load(row_logits_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)

        block_max = tl.max(x)
        new_m = tl.maximum(m, block_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m))
        m = new_m

        if label_smoothing > 0.0:
            smoothed_negative_logit_sum += tl.sum(tl.where(mask, -eps * x, 0.0))

    logsumexp = m + tl.log(d)

    # The per-row hard-label CE loss is logsumexp(x) - x[target].
    loss = logsumexp - target_logit

    # PyTorch's smoothed hard-label CE can be written as:
    # (1 - s) * hard_ce + sum_i(-s/V * x_i) + s * logsumexp(x)
    # This mirrors Liger's derivation and avoids materializing log_softmax.
    if label_smoothing > 0.0:
        smooth_loss = smoothed_negative_logit_sum + label_smoothing * logsumexp
        loss = (1.0 - label_smoothing) * loss + smooth_loss

    # Liger normalizes each row before the host-side sum for mean reduction.
    # That keeps the reduction step simple: host code always sums loss_1d for
    # mean and sum; only the row values differ.
    if reduction == "mean":
        loss = loss / n_non_ignore

    # The gradient formula is also computed while the original logits are still
    # available. Once we store gradients back to logits_ptr, the caller must
    # treat that buffer as a saved grad buffer, not as logits anymore.
    if has_gradients:
        for col_start in range(0, n_cols, block_size):
            offsets = col_start + tl.arange(0, block_size)
            mask = offsets < n_cols
            x = tl.load(row_logits_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)

            # softmax(x_i) = exp(x_i - m) / d, with m/d from the online pass.
            grad = tl.exp(x - m) / d

            if label_smoothing > 0.0:
                # For smoothing, every class receives -s/V and the true class
                # receives an additional -(1 - s).
                grad = grad - eps
                grad = tl.where(offsets == target, grad - (1.0 - label_smoothing), grad)
            else:
                grad = tl.where(offsets == target, grad - 1.0, grad)

            if reduction == "mean":
                grad = grad / n_non_ignore

            tl.store(row_logits_ptr + offsets, grad, mask=mask)

    # Liger puts a barrier between writing gradients into the logits buffer and
    # later stores. It is cheap here and documents the same ordering assumption:
    # by the time forward returns, saved logits storage contains gradients.
    tl.debug_barrier()
    tl.store(row_loss_ptr, loss)


@triton.jit
def _forge_scale_kernel(x_ptr, grad_ptr, n_elements, block_size: tl.constexpr):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr)
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, values * grad, mask=mask)


def _is_triton_path_supported(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    reduction: str,
) -> bool:
    # CPU fallback keeps local tests usable. Weight fallback keeps the public
    # function aligned with torch.nn.functional.cross_entropy while Forge's
    # first Triton variant focuses on the unweighted benchmark case.
    return (
        logits.is_cuda
        and target.is_cuda
        and logits.ndim == 2
        and target.ndim == 1
        and target.numel() == logits.shape[0]
        and weight is None
        and reduction in {"mean", "sum", "none"}
        and logits.dtype in {torch.float16, torch.bfloat16, torch.float32}
    )


def _validate_targets(target: torch.Tensor, ignore_index: int, vocab_size: int) -> int:
    # Liger performs host-side target checks before launching Triton. Without
    # this, an invalid target can become an out-of-bounds load inside the kernel.
    target_mask = target != ignore_index
    n_non_ignore = int(target_mask.sum().item())
    if n_non_ignore == 0:
        return 0

    active_targets = target.masked_select(target_mask)
    max_target = int(active_targets.max().item())
    min_target = int(active_targets.min().item())
    if max_target >= vocab_size:
        raise IndexError(f"Target {max_target} is out of bounds. Expected < {vocab_size}")
    if min_target < 0:
        raise IndexError(f"Target {min_target} is out of bounds. Expected >= 0")
    return n_non_ignore


def _forward_triton(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    bt, vocab_size = logits.shape
    n_non_ignore = _validate_targets(target, ignore_index, vocab_size)

    # Liger requires contiguous row storage because the Triton program walks the
    # vocab dimension with pointer + offset. Forge does the same, making a
    # contiguous copy only when needed.
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
    loss_1d = torch.zeros(bt, dtype=logits.dtype, device=logits.device)

    _forge_cross_entropy_forward_kernel[(bt,)](
        logits,
        logits.stride(0),
        target,
        target.stride(0),
        loss_1d,
        vocab_size,
        n_non_ignore,
        ignore_index,
        label_smoothing,
        reduction,
        logits.requires_grad,
        block_size,
        # Liger notes cross entropy performance is sensitive to num_warps and
        # uses 32 on CUDA. Forge keeps that setting for an apples-to-apples POC.
        num_warps=32,
    )

    if reduction == "none":
        return loss_1d, logits if logits.requires_grad else None
    if reduction == "mean" and n_non_ignore == 0:
        # PyTorch returns NaN for this degenerate case. The kernel wrote zeros
        # and zero gradients for every ignored row, so only the scalar value
        # needs the 0/0 treatment here.
        return torch.sum(loss_1d) / n_non_ignore, logits if logits.requires_grad else None
    return torch.sum(loss_1d), logits if logits.requires_grad else None


def _scale_saved_gradient(saved_grad: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    if grad_output.ndim > 0:
        # reduction="none": grad_output is one scalar per row.
        return saved_grad * grad_output.unsqueeze(1)

    # Liger skips the multiply when CE is the final scalar loss. This is the
    # common training path and avoids a whole extra kernel launch in backward.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device, dtype=grad_output.dtype)):
        return saved_grad

    n_elements = saved_grad.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _forge_scale_kernel[grid](saved_grad, grad_output, n_elements, block_size, num_warps=4)
    return saved_grad


class ForgeCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
    ) -> torch.Tensor:
        loss, saved_grad = _forward_triton(logits, target, ignore_index, reduction, label_smoothing)
        if saved_grad is not None:
            # Detaching is important for the same reason Liger does it: the
            # saved tensor is no longer logically logits, it is the gradient
            # buffer produced during forward. Saving the detached buffer avoids
            # autograd retaining another copy of the original values.
            ctx.save_for_backward(saved_grad.detach())
        else:
            ctx.save_for_backward()
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        grad_logits = _scale_saved_gradient(saved[0], grad_output) if saved else None
        return grad_logits, None, None, None, None


def forge_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Forge cross entropy with a Triton fast path and a Torch compatibility path."""

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")

    if not _is_triton_path_supported(logits, target, weight, reduction):
        return F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    return ForgeCrossEntropyFunction.apply(logits, target, ignore_index, reduction, label_smoothing)

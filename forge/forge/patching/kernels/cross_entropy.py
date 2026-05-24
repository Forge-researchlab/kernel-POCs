"""Module-level patch adapter for cross entropy."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as torch_F

from forge.kernels.cross_entropy import forge_cross_entropy


_TORCH_CROSS_ENTROPY = torch_F.cross_entropy


def forge_cross_entropy_replacement(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    """Drop-in-ish replacement for ``torch.nn.functional.cross_entropy``.

    Forge CE expects flattened ``[N, C]`` CUDA logits. Unsupported legacy args,
    CPU tensors, or higher-rank inputs fall back to the original PyTorch CE.
    """
    if size_average is not None or reduce is not None or input.dim() != 2 or not input.is_cuda:
        return _TORCH_CROSS_ENTROPY(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    return forge_cross_entropy(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

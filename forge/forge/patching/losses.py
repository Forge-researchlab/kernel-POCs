"""Backward-compatible import path for loss patch adapters."""

from forge.patching.kernels.cross_entropy import forge_cross_entropy_replacement

__all__ = ["forge_cross_entropy_replacement"]

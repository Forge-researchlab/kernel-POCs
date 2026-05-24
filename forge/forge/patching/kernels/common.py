"""Shared helpers for Forge patch-adapter factories."""
from __future__ import annotations


class ForgeSkipPatch(RuntimeError):
    """Internal sentinel: a real kernel is not applicable to this module."""

"""Forge RMSNorm — versioned Triton kernels.

Available versions:
    v1: forge_rmsnorm_v1.py — placeholder baseline, no offset support, kept for v1-vs-v2 comparison.
    v2: forge_rmsnorm_v2.py — Liger-style offset constexpr + casting modes + SM-proportional dW partials.
    v3: forge_rmsnorm_v3.py — v2 body + @triton.autotune over (num_warps, num_stages).

Default `apply_rmsnorm` re-exports v3 (the shipping kernel). The `forge.patch`
layer imports `apply_rmsnorm` from this package via `forge/forge/kernels/rmsnorm.py`.

Unversioned legacy symbols (`ForgeRMSNormFunction`, `rmsnorm`, `rmsnorm_forward`,
`rmsnorm_backward`) alias to v1, so the pre-existing `tests/test_rmsnorm.py`
and `forge/tests/verify_patch_qwen3.py` continue to work without edits.
"""
# ----------------------------------------------------------------------------
# v1 (placeholder baseline)
# ----------------------------------------------------------------------------
from .forge_rmsnorm_v1 import (
    ForgeRMSNormv1Function,
    rmsnorm_v1,
    rmsnorm_v1_forward,
    rmsnorm_v1_backward,
    torch_rmsnorm_reference,
)

# ----------------------------------------------------------------------------
# v2 (shipping kernel) + v3 (v2 + autotune). Imported lazily so partial
# checkouts (where v2/v3 haven't landed yet) still expose v1.
# ----------------------------------------------------------------------------
try:
    from .forge_rmsnorm_v2 import (
        ForgeRMSNormv2Function,
        ForgeRMSNormv2,
        apply_rmsnorm_v2,
    )
    _V2_AVAILABLE = True
except ImportError:  # pragma: no cover — v2 not built yet
    _V2_AVAILABLE = False

try:
    from .forge_rmsnorm_v3 import (
        ForgeRMSNormv3Function,
        ForgeRMSNormv3,
        apply_rmsnorm_v3,
    )
    _V3_AVAILABLE = True
except ImportError:  # pragma: no cover — v3 not built yet
    _V3_AVAILABLE = False


# Default `apply_rmsnorm` — v3 if available, else v2, else v1.
if _V3_AVAILABLE:
    apply_rmsnorm = apply_rmsnorm_v3
elif _V2_AVAILABLE:
    apply_rmsnorm = apply_rmsnorm_v2
else:
    def apply_rmsnorm(x, weight, eps=1e-6, offset=0.0, casting_mode="llama"):
        """v1 fallback — silently drops offset/casting_mode (v1 doesn't support them).

        Emits a warning if offset != 0.0 so callers aren't surprised.
        """
        if offset != 0.0 or casting_mode == "gemma":
            import warnings
            warnings.warn(
                "kernels.rmsnorm.apply_rmsnorm is falling back to v1 which "
                "ignores offset/casting_mode. Build v2 for Gemma support.",
                RuntimeWarning,
                stacklevel=2,
            )
        return rmsnorm_v1(x, weight, eps)


# ----------------------------------------------------------------------------
# Legacy unversioned aliases — preserve the pre-rename import surface used by
# `tests/test_rmsnorm.py` and any patching-layer code that pre-dates v2.
# ----------------------------------------------------------------------------
ForgeRMSNormFunction = ForgeRMSNormv1Function
rmsnorm = rmsnorm_v1
rmsnorm_forward = rmsnorm_v1_forward
rmsnorm_backward = rmsnorm_v1_backward


__all__ = [
    # versioned
    "ForgeRMSNormv1Function",
    "rmsnorm_v1", "rmsnorm_v1_forward", "rmsnorm_v1_backward",
    "torch_rmsnorm_reference",
    "apply_rmsnorm",
    # legacy aliases
    "ForgeRMSNormFunction", "rmsnorm", "rmsnorm_forward", "rmsnorm_backward",
]
if _V2_AVAILABLE:
    __all__ += ["ForgeRMSNormv2Function", "ForgeRMSNormv2", "apply_rmsnorm_v2"]
if _V3_AVAILABLE:
    __all__ += ["ForgeRMSNormv3Function", "ForgeRMSNormv3", "apply_rmsnorm_v3"]

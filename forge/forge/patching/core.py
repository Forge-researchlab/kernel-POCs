"""forge.patching.core — the patch/unpatch loop, closure factories, and dispatch.

Design (locked, per design_details.html#patching-design and #wiring):

  * Forward replacement, NOT module replacement.
        Module replacement breaks state_dict loading and confuses FSDP2's
        transformer-layer-cls wrap policies. We monkey-patch `module.forward`
        and remember the original on `model._forge_originals` so unpatch is
        bit-exact reversible.

  * Architecture detected via `model.config.model_type`, never type(model).__name__.
        Qwen2.5 / Qwen3 report "qwen2" (and sometimes "qwen3"); Gemma-2 reports
        "gemma2". The mapping registry keys on this string.

  * Closure factory is mandatory.
        Don't write `module.forward = lambda x: forge_module(x)` inside a loop.
        Python's late binding will bind `forge_module` to whatever the loop
        variable points to at the end — silently wrong. Always use a factory
        function that closes over the right reference.

  * Closures close over `module.weight` directly, not a copy.
        LoRA adapters that update the weight at training time must be visible
        through the closure on the next forward. Copying at patch time freezes it.

  * Double-patch raises RuntimeError, not silent no-op.
        Loud failure catches the bug where a test fixture forgets to unpatch.

  * Selective patching: `forge.patch(model, kernels=["embedding"])`.
        The #1 debugging tool when training diverges — enable kernels one at a
        time to bisect which one is broken.

  * Stubs raise NotImplementedError only when explicitly requested.
        forge.patch(model) silently skips kernels not yet built; the caller can
        force-fail by passing kernels=[<unbuilt name>] to confirm the gap.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Optional, Tuple

from .qwen3 import QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES
from .gemma import GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES
from .kernels import FORWARD_MAKERS as _FORWARD_MAKERS
from .kernels import ForgeSkipPatch


# ----------------------------------------------------------------------------
# Architecture detection
# ----------------------------------------------------------------------------

_ARCH_TO_MAPPING: Dict[str, Tuple[dict, dict]] = {
    # model_type -> (per_class_mapping, module_level_patches)
    "qwen2":  (QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES),
    "qwen3":  (QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES),
    "gemma":  (GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES),
    "gemma2": (GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES),
}


def _model_config(model):
    config = getattr(model, "config", None)
    if config is not None:
        return config
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None)
    config = getattr(inner, "config", None)
    if config is not None:
        return config
    raise ValueError("forge.patch: model does not expose a Hugging Face config.")


def _detect_architecture(model) -> str:
    model_type = getattr(_model_config(model), "model_type", None)
    if model_type not in _ARCH_TO_MAPPING:
        raise ValueError(
            f"forge.patch: no Forge mapping for model_type={model_type!r}. "
            f"Supported: {sorted(_ARCH_TO_MAPPING)}."
        )
    return model_type


# ----------------------------------------------------------------------------
# Kernel-specific forward factories live in forge.patching.kernels.
# core.py owns only orchestration: validation, traversal, mutation, restoration.
# ----------------------------------------------------------------------------

def _make_embedding_forward(module, config):
    """nn.Embedding.forward -> ForgeEmbeddingFunction.apply(weight, indices, padding_idx).

    Forwards `module.padding_idx` to the kernel so the pad row's gradient is
    zeroed in backward — matching nn.Embedding's contract. Without this the
    pad embedding silently drifts under training (caught by the Gemma
    padding-stress test). Gemma/Llama/Qwen all set padding_idx by default,
    so this is the common case, not an edge case.
    """
    from forge.kernels.embedding import ForgeEmbeddingFunction

    padding_idx = module.padding_idx  # int or None (resolved by nn.Embedding init)

    def forward(input_ids):
        return ForgeEmbeddingFunction.apply(module.weight, input_ids, padding_idx)
    return forward


def _make_rmsnorm_forward(module, config):
    """{Qwen,Gemma,Llama}RMSNorm.forward -> apply_rmsnorm(x, weight, eps, offset, casting_mode, in_place).

    Reads `offset` from config (0.0 for Qwen/Llama, 1.0 for Gemma) and picks
    casting_mode by default ("gemma" when offset==1.0, else "llama"); both can
    be overridden via the mapping config. Reads `eps` from `module.variance_epsilon`
    (Qwen2/3) or falls back to `module.eps` (Gemma).

    `in_place` controls v4's backward dY → dX optimization:
      - Qwen/Llama (offset=0): default True (sequential RMSNorm pattern; dy
                               not needed after backward — safe).
      - Gemma2 (offset=1):     default False (residual-paired RMSNorm; dy is
                               consumed by another backward node — must preserve).
    Override via `{"in_place": True/False}` in the mapping config.
    """
    from forge.kernels.rmsnorm import apply_rmsnorm

    offset = float(config.get("offset", 0.0))
    casting_mode = config.get("casting_mode", "gemma" if offset == 1.0 else "llama")
    # in_place default tracks offset: safe for Qwen/Llama, unsafe for Gemma residual paths.
    in_place = bool(config.get("in_place", offset != 1.0))
    eps = float(getattr(module, "variance_epsilon",
                        getattr(module, "eps", 1e-6)))

    def forward(hidden_states):
        # Close over `module.weight` (not a copy) so LoRA updates flow live.
        return apply_rmsnorm(hidden_states, module.weight, eps=eps,
                             offset=offset, casting_mode=casting_mode,
                             in_place=in_place)
    return forward


def _make_geglu_forward(module, config):
    """Gemma/Gemma2 MLP.forward -> down_proj( geglu(gate_proj(x), up_proj(x), approximate=...) ).

    Mirrors `_make_swiglu_forward` (calls the projection submodules, not their
    .weight, so PEFT adapters keep working). Only the activation kernel differs:
    SiLU -> GELU (variant chosen per-model from the HF config).

    Approximation mode: Gemma defaults to `gelu_pytorch_tanh` (tanh-approx GELU);
    our kernel exposes `approximate="tanh" | "none"`. We read the HF config
    string (`hidden_activation` for Gemma2, `hidden_act` for Gemma) and map to
    the kernel's mode. Unknown strings raise — silently swapping an exact GELU
    for the tanh approx would shift forward outputs ~1e-3 and fail parity tests.

    Activations resolved here (HF ACT2FN names):
        gelu, gelu_python              -> "none"  (exact)
        gelu_pytorch_tanh, gelu_new    -> "tanh"  (approx; Gemma's actual default)
    """
    from forge.kernels.geglu import geglu

    # Caller can override via mapping config; otherwise sniff the HF config.
    explicit_mode = config.get("approximate")
    if explicit_mode is not None:
        approximate = explicit_mode
    else:
        hf_cfg = getattr(module, "config", None)
        act_name = None
        if hf_cfg is not None:
            act_name = getattr(hf_cfg, "hidden_activation", None) or getattr(hf_cfg, "hidden_act", None)
        _EXACT = {"gelu", "gelu_python"}
        _TANH = {"gelu_pytorch_tanh", "gelu_new"}
        if act_name in _EXACT:
            approximate = "none"
        elif act_name in _TANH:
            approximate = "tanh"
        else:
            raise NotImplementedError(
                f"forge.patch: geglu kernel cannot infer GELU variant for "
                f"hidden_activation={act_name!r}. Pass approximate='tanh' or "
                f"approximate='none' via the mapping config to override."
            )

    def forward(x):
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(geglu(gate, up, approximate=approximate))
    return forward


def _make_swiglu_forward(module, config):
    """Qwen2/3 MLP.forward -> down_proj( swiglu(gate_proj(x), up_proj(x)) ).

    The SwiGLU kernel only fuses `silu(gate) * up`; the three projections stay
    as the module's own `nn.Linear` (or PEFT-wrapped LoRA Linear) so PEFT
    adapters keep working through `forge.patch`. Calling the submodules
    (`module.gate_proj(x)`) instead of using their `.weight` directly is what
    lets adapter-wrapped Linears intercept the call.

    Rejects non-`silu` activations at build time so a Gemma-style
    `activation="gelu"` mapping fails loudly instead of silently producing
    wrong outputs through a SiLU-only kernel. Gemma's GELU path lives in the
    separate `geglu` kernel.
    """
    activation = config.get("activation", "silu")
    if activation != "silu":
        raise NotImplementedError(
            f"forge.patch: swiglu kernel only supports activation='silu', "
            f"got {activation!r}. Route this module to the 'geglu' kernel instead."
        )

    from forge.kernels.swiglu import swiglu

    def forward(x):
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(swiglu(gate, up))
    return forward


def _make_not_implemented(kernel_name: str, why: str):
    """Stub factory: forge.patch silently skips this kernel unless the caller
    requests it explicitly via kernels=[...]. Then it raises so the gap is loud.
    """
    def factory(module, config):
        def forward(*args, **kwargs):
            raise NotImplementedError(
                f"Forge kernel {kernel_name!r} is not implemented yet ({why}). "
                f"Use forge.patch(model, kernels=[<built kernels only>]) to skip it."
            )
        return forward
    factory.__forge_stub__ = True   # sentinel: patch loop skips unless explicit
    factory.__forge_kernel_name__ = kernel_name
    factory.__forge_reason__ = why
    return factory


_FORWARD_MAKERS: Dict[str, Callable] = {
    # === Wired to real kernels ===
    "embedding":     _make_embedding_forward,
    "rmsnorm":       _make_rmsnorm_forward,
    "swiglu":        _make_swiglu_forward,
    "geglu":         _make_geglu_forward,

    # === Stubs — declared so the mapping shape matches the V1 plan; flipping
    # to real is a one-line change when the team lands the kernel.
    "cross_entropy": _make_not_implemented("cross_entropy", "H3 — model/loss forward interception not wired"),
    "lora_qkv":      _make_not_implemented("lora_qkv",      "H5 — not built"),
    "lora_mlp":      _make_not_implemented("lora_mlp",      "PEFT integration deferred per Day 2 scope"),
}


# ----------------------------------------------------------------------------
# Per-module-instance forward replacement
# ----------------------------------------------------------------------------

def _replace_forward(module, kernel_name: str, config: dict, originals: dict) -> bool:
    """Replace `module.forward` via the matching closure factory.

    Returns True if the replacement happened, False if the kernel is a stub and
    we silently skipped it.
    """
    if kernel_name not in _FORWARD_MAKERS:
        raise KeyError(
            f"forge.patch: unknown kernel {kernel_name!r}. "
            f"Registered: {sorted(_FORWARD_MAKERS)}."
        )
    maker = _FORWARD_MAKERS[kernel_name]
    try:
        new_fwd = maker(module, config)
    except ForgeSkipPatch:
        return False
    originals[id(module)] = (module, module.forward)
    module.forward = new_fwd
    return True


# ----------------------------------------------------------------------------
# Module-level function replacement (RoPE)
# ----------------------------------------------------------------------------

_MODULE_LEVEL_KEY = "__forge_module_level__"


def _apply_module_level_patches(patches: List[Tuple[str, str, Callable]], originals: dict) -> None:
    """Replace module-level functions (e.g. transformers' apply_rotary_pos_emb).

    `patches` is a list of (module_path, attr_name, replacement) tuples. We
    import the module, save the original attribute, and overwrite. unpatch
    walks the saved list and restores.
    """
    saved = []
    for module_path, attr_name, replacement in patches:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            # Quietly skip if the transformers submodule isn't installed for this arch
            continue
        original = getattr(mod, attr_name, None)
        if original is None:
            continue
        saved.append((mod, attr_name, original))
        setattr(mod, attr_name, replacement)
    if saved:
        originals[_MODULE_LEVEL_KEY] = saved


def _revert_module_level_patches(originals: dict) -> None:
    for mod, attr_name, original in originals.get(_MODULE_LEVEL_KEY, []):
        setattr(mod, attr_name, original)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def patch(model, kernels: Optional[List[str]] = None):
    """Patch an HF model in place. Returns the same model object.

    Args:
        model:   HF causal LM (currently Qwen2/Qwen3 or Gemma2).
        kernels: Optional whitelist of kernel names. None enables every kernel
                 with a real implementation (stubs are silently skipped).
                 Pass an explicit list to bisect during debugging:
                     forge.patch(model, kernels=["embedding"])
                     forge.patch(model, kernels=["rope"])

    Raises:
        RuntimeError if the model is already patched (call unpatch first).
        ValueError if model.config.model_type isn't supported.
        NotImplementedError if `kernels` names a kernel that's still a stub.
    """
    if getattr(model, "_forge_patched", False):
        raise RuntimeError("Model already patched. Call forge.unpatch(model) first.")

    arch = _detect_architecture(model)
    class_mapping, module_level_patches = _ARCH_TO_MAPPING[arch]

    # --- pre-validate: fail loudly BEFORE mutating any module, so a stub in the
    # `kernels` whitelist doesn't leave the model half-patched.
    if kernels is not None:
        for kernel_name in kernels:
            maker = _FORWARD_MAKERS.get(kernel_name)
            if maker is None:
                # Could still be valid as a module-level kernel (e.g., "rope")
                if kernel_name not in module_level_patches:
                    raise KeyError(
                        f"forge.patch: unknown kernel {kernel_name!r}. "
                        f"Known per-module: {sorted(_FORWARD_MAKERS)}. "
                        f"Known module-level: {sorted(module_level_patches)}."
                    )
                continue
            if getattr(maker, "__forge_stub__", False):
                raise NotImplementedError(
                    f"forge.patch: kernel {kernel_name!r} requested but not implemented yet "
                    f"({getattr(maker, '__forge_reason__', '')}). "
                    f"Drop {kernel_name!r} from the kernels=[...] list or build it first."
                )

    originals: dict = {}

    # --- per-module-instance forward replacement ---
    patched_count: Dict[str, int] = {}
    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name not in class_mapping:
            continue
        patch_specs = class_mapping[cls_name]
        if isinstance(patch_specs, tuple):
            patch_specs = [patch_specs]

        for kernel_name, config in patch_specs:
            # Filter by `kernels` whitelist
            if kernels is not None and kernel_name not in kernels:
                continue

            maker = _FORWARD_MAKERS.get(kernel_name)
            if maker is None:
                continue
            # Stubs: silent skip when patching everything; pre-validation above already
            # caught the explicit-whitelist case.
            if getattr(maker, "__forge_stub__", False):
                continue

            if _replace_forward(module, kernel_name, config, originals):
                patched_count[kernel_name] = patched_count.get(kernel_name, 0) + 1
                break

    # --- module-level patches (RoPE et al.) ---
    selected_module_level = []
    for kernel_name, patch_spec in module_level_patches.items():
        if kernels is not None and kernel_name not in kernels:
            continue
        # A kernel can patch more than one transformers module path (e.g. qwen2
        # and qwen3 have separate modeling modules in newer transformers).
        if isinstance(patch_spec, tuple):
            selected_module_level.append(patch_spec)
        else:
            selected_module_level.extend(patch_spec)
    _apply_module_level_patches(selected_module_level, originals)

    model._forge_originals = originals
    model._forge_patched = True
    model._forge_patched_counts = patched_count
    model._forge_arch = arch
    return model


def unpatch(model):
    """Restore original forwards. Idempotent — calling on an unpatched model is a no-op."""
    if not getattr(model, "_forge_patched", False):
        return model

    originals = model._forge_originals

    # Restore module-instance forwards
    for key, payload in originals.items():
        if key == _MODULE_LEVEL_KEY:
            continue
        module, original_fwd = payload
        module.forward = original_fwd

    # Restore module-level functions
    _revert_module_level_patches(originals)

    del model._forge_originals
    del model._forge_patched
    if hasattr(model, "_forge_patched_counts"):
        del model._forge_patched_counts
    if hasattr(model, "_forge_arch"):
        del model._forge_arch
    return model

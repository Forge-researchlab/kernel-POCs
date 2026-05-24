"""Patch adapters for PEFT LoRA kernels.

These adapters translate live PEFT-wrapped Qwen modules into the tensor API used
by the Forge LoRA kernels. The compute kernels remain in ``kernels/lora_*``;
this file only owns patch-time extraction and HF forward reconstruction.
"""
from __future__ import annotations

import importlib

import torch

from .common import ForgeSkipPatch


def _linear_base_weight(linear):
    base = getattr(linear, "base_layer", linear)
    return getattr(base, "weight", None)


def _linear_base_bias(linear):
    base = getattr(linear, "base_layer", linear)
    return getattr(base, "bias", None)


def _active_lora_adapter(linear):
    lora_a = getattr(linear, "lora_A", None)
    lora_b = getattr(linear, "lora_B", None)
    if not lora_a or not lora_b:
        return None
    active = getattr(linear, "active_adapters", None)
    if active is None:
        active = getattr(linear, "active_adapter", None)
    if callable(active):
        active = active()
    if isinstance(active, str):
        active = [active]
    if not active:
        active = list(lora_a.keys())
    for adapter in active:
        if adapter in lora_a and adapter in lora_b:
            return adapter
    return None


def _is_identity_dropout(dropout) -> bool:
    if dropout is None:
        return True
    if isinstance(dropout, torch.nn.Identity):
        return True
    return float(getattr(dropout, "p", 0.0)) == 0.0


def _lora_tensors(linear, *, allow_bias: bool = False):
    adapter = _active_lora_adapter(linear)
    if adapter is None:
        raise ForgeSkipPatch("no active PEFT LoRA adapter")
    if getattr(linear, "disable_adapters", False) or getattr(linear, "merged", False):
        raise ForgeSkipPatch("LoRA adapter is disabled or merged")
    bias = _linear_base_bias(linear)
    if bias is not None and not allow_bias:
        raise ForgeSkipPatch("LoRA Forge kernels currently require bias-free base Linear modules")

    dropouts = getattr(linear, "lora_dropout", {})
    dropout = dropouts.get(adapter) if hasattr(dropouts, "get") else None
    if not _is_identity_dropout(dropout):
        raise ForgeSkipPatch("LoRA dropout must be 0.0 for exact fused patching")

    scaling = getattr(linear, "scaling", {}).get(adapter, 1.0)
    return (
        _linear_base_weight(linear),
        linear.lora_A[adapter].weight,
        linear.lora_B[adapter].weight,
        float(scaling),
        bias,
    )


def _check_lora_active(linear):
    """Same skip-patch policy as `_lora_tensors`, but without extracting any
    weight references. Used by the FSDP2-safe LoRA-QKV path which routes
    matmuls through PEFT submodule calls (so FSDP2's all-gather hooks fire)
    instead of closing over raw weight tensors at patch time.
    """
    adapter = _active_lora_adapter(linear)
    if adapter is None:
        raise ForgeSkipPatch("no active PEFT LoRA adapter")
    if getattr(linear, "disable_adapters", False) or getattr(linear, "merged", False):
        raise ForgeSkipPatch("LoRA adapter is disabled or merged")
    dropouts = getattr(linear, "lora_dropout", {})
    dropout = dropouts.get(adapter) if hasattr(dropouts, "get") else None
    if not _is_identity_dropout(dropout):
        raise ForgeSkipPatch("LoRA dropout must be 0.0 for exact fused patching")


_QWEN_MLP_ACTIVATIONS = {"silu", None}  # None → silu by default in HF Qwen MLP
_GEMMA_TANH_ACTIVATIONS = {"gelu_pytorch_tanh", "gelu_new"}
_GEMMA_EXACT_ACTIVATIONS = {"gelu", "gelu_python"}


def _detect_mlp_activation(module):
    """Sniff the activation kind from the HF config attached to the MLP.

    Qwen MLPs use SiLU (the kernel default for LoRAMLPv3); Gemma 2 MLPs use
    `gelu_pytorch_tanh` or `gelu`. Returns one of: "silu", "gelu_tanh",
    "gelu_exact". Raises ForgeSkipPatch when the activation is unrecognized —
    silently routing a GeGLU MLP through a SiLU kernel would shift outputs
    ~1e-3 (caught by the Gemma forward-parity test).
    """
    hf_cfg = getattr(module, "config", None)
    act = None
    if hf_cfg is not None:
        act = getattr(hf_cfg, "hidden_activation", None) or getattr(hf_cfg, "hidden_act", None)
    if act in _QWEN_MLP_ACTIVATIONS:
        return "silu"
    if act in _GEMMA_TANH_ACTIVATIONS:
        return "gelu_tanh"
    if act in _GEMMA_EXACT_ACTIVATIONS:
        return "gelu_exact"
    raise ForgeSkipPatch(
        f"LoRA MLP fused kernel cannot infer activation from {act!r}; "
        f"fall back to per-projection patching"
    )


def make_lora_mlp_forward(module, config):
    """PEFT LoRA MLP.forward -> Forge LoRA-MLP fused kernel.

    Two activation paths:

      * SiLU (Qwen 2/3 default) -> LoRAMLPv3 — single autograd.Function that
        fuses gate/up/down projections + LoRA + SwiGLU.
      * GeGLU (Gemma 2 default) -> manual gate/up/down through the same
        projections, with the GeGLU activation routed through the existing
        forge.kernels.geglu kernel. We can't reuse LoRAMLPv3 here because it
        hardcodes SiLU (see kernels/lora_mlp/.../lora_mlp_kernel_v3.py:133
        `silu_e = e_tile * tl.sigmoid(e_tile)`).

    The Gemma path is a thin layered approach (LoRA projections via PyTorch + the
    existing GeGLU kernel for the activation) rather than a deeper fusion — this
    matches the Day-2 scope and ships a correct LoRA path for Gemma 2 without
    requiring a new LoRA-MLP-GeGLU kernel. A future fully-fused Gemma path can
    plug in here once the kernel exists.
    """
    activation = _detect_mlp_activation(module)

    if activation == "silu":
        # FSDP2-safe path (see context/phase3_fsdp2_resume.md §7). Going
        # through PEFT submodules (gate/up/down_proj) lets FSDP2's all-gather
        # hooks fire on each weight access. The earlier LoRAMLPv3-fused path
        # closed over raw `module.*.base_layer.weight` refs which break under
        # DTensor dispatch (mixed-Tensor/DTensor RuntimeError on aten.mm).
        # Trade-off: lost the fused SwiGLU+LoRA win; activation alone stays
        # fused via the swiglu kernel. Symmetric with the GeGLU branch below.
        from forge.kernels.swiglu import swiglu

        _check_lora_active(module.gate_proj)
        _check_lora_active(module.up_proj)
        _check_lora_active(module.down_proj)

        def forward(x):
            gate = module.gate_proj(x)
            up = module.up_proj(x)
            return module.down_proj(swiglu(gate, up))
        return forward

    # GeGLU path (Gemma 2) — no fused LoRA-MLP-GeGLU kernel yet, so we lean on
    # the projection submodules (which PEFT has already wrapped to include
    # their LoRA-A/B contribution) and just fuse the GeGLU activation.
    from forge.kernels.geglu import geglu

    # Pre-validate the projections are PEFT-wrapped with extractable adapters —
    # we don't use the extracted tensors here (the projection submodules know
    # how to compute their LoRA-augmented outputs themselves), but raising
    # ForgeSkipPatch on a non-LoRA module falls through cleanly to the geglu
    # mapping that would otherwise apply to this class.
    _lora_tensors(module.gate_proj)
    _lora_tensors(module.up_proj)
    _lora_tensors(module.down_proj)

    approximate = "tanh" if activation == "gelu_tanh" else "none"

    def forward(x):
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(geglu(gate, up, approximate=approximate))

    return forward


def make_lora_qkv_forward(module, config):
    """PEFT LoRA attention Q/K/V projections — FSDP2-safe variant.

    Routes Q/K/V through PEFT's wrapped Linear submodules (`module.q_proj(x)`
    etc.) instead of closing over raw weight tensors at patch time. This is
    the locked Phase 3 design (see `context/phase3_fsdp2_resume.md` §7): when
    FSDP2 shards a Linear's weight, all-gather is triggered by the submodule
    `forward` hook chain. Bypassing it with a raw tensor pointer captured at
    patch time leaves the closure pointing at a stale shard, which crashes
    with `mixed torch.Tensor and DTensor` under `aten.mm.default`.

    Trade-off: this path is unfused (three independent LoRA matmuls instead
    of one packed-QKV matmul). The v4 fused-QKV win is recovered later when
    FSDP2 can be taught to honor raw tensor closures, or via DTensor-aware
    rewrites of the v4 packer. For now correctness > speed.

    Works for both Qwen (2/3) and Gemma 2 attention blocks. Architecture
    differences handled at call time, not closure-build time:
      * Gemma 2 sets `self.sliding_window` per-layer (None on full-attn layers).
        The `getattr(module, "sliding_window", None)` path picks this up; the
        Qwen-specific config lookup is a no-op for Gemma 2 because
        `use_sliding_window` is absent.
      * Gemma 2 has `attn_logit_softcapping` (numeric or None) that must be
        forwarded to the attention interface as `softcap=`. Qwen 3 has no
        such attribute, so we only pass softcap when present.
      * Gemma 2's HF forward takes `past_key_values` (plural); older Qwen
        signatures used `past_key_value` (singular). We accept both names.
      * Gemma 2 has no q_norm/k_norm (Qwen 3 specific). The hasattr-guards
        already in place make this safe.
    """
    _check_lora_active(module.q_proj)
    _check_lora_active(module.k_proj)
    _check_lora_active(module.v_proj)
    modeling = importlib.import_module(module.__class__.__module__)

    def forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # Accept either naming (Gemma 2 uses past_key_values plural; older Qwen
        # signatures use past_key_value singular). Whichever the outer block
        # passes is the one we use.
        kv_cache = past_key_values if past_key_values is not None else past_key_value

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)

        # PEFT submodule calls — FSDP2 sees these and triggers all-gather. The
        # PEFT Linear wrapper applies base + lora_B(lora_A(x)) * scaling + bias
        # in one go, so we don't separately add bias here.
        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if hasattr(module, "q_norm"):
            query_states = module.q_norm(query_states)
        if hasattr(module, "k_norm"):
            key_states = module.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = kv_cache.update(key_states, value_states, module.layer_idx, cache_kwargs)

        sliding_window = getattr(module, "sliding_window", None)
        if sliding_window is None and hasattr(module, "config"):
            if (
                getattr(module.config, "use_sliding_window", False)
                and getattr(module.config, "sliding_window", None) is not None
                and module.layer_idx >= getattr(module.config, "max_window_layers", 0)
            ):
                sliding_window = module.config.sliding_window

        attention_interface = modeling.eager_attention_forward
        if module.config._attn_implementation != "eager":
            if module.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                modeling.logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention."
                )
            else:
                attention_interface = modeling.ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

        # Gemma 2 expects softcap=; Qwen doesn't pass it. Only forward when
        # the attention block actually carries it (and it's non-None).
        extra_kwargs = {}
        softcap = getattr(module, "attn_logit_softcapping", None)
        if softcap is not None:
            extra_kwargs["softcap"] = softcap

        attn_output, attn_weights = attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=sliding_window,
            **extra_kwargs,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = module.o_proj(attn_output)
        return attn_output, attn_weights

    return forward

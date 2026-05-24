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


def make_lora_mlp_forward(module, config):
    """Qwen PEFT LoRA MLP.forward -> Forge LoRA-MLP v3."""
    from forge.kernels.lora_mlp import LoRAMLPv3

    W_gate, A_gate, B_gate, s_gate, _ = _lora_tensors(module.gate_proj)
    W_up, A_up, B_up, s_up, _ = _lora_tensors(module.up_proj)
    W_down, A_down, B_down, s_down, _ = _lora_tensors(module.down_proj)

    def forward(x):
        dtype = x.dtype
        return LoRAMLPv3.apply(
            x,
            W_gate,
            A_gate.to(dtype),
            B_gate.to(dtype),
            s_gate,
            W_up,
            A_up.to(dtype),
            B_up.to(dtype),
            s_up,
            W_down,
            A_down.to(dtype),
            B_down.to(dtype),
            s_down,
        )

    return forward


def make_lora_qkv_forward(module, config):
    """Qwen PEFT LoRA attention Q/K/V projections -> Forge LoRA-QKV v3."""
    from forge.kernels.lora_qkv import lora_qkv_v3

    W_q, A_q, B_q, s_q, bias_q = _lora_tensors(module.q_proj, allow_bias=True)
    W_k, A_k, B_k, s_k, bias_k = _lora_tensors(module.k_proj, allow_bias=True)
    W_v, A_v, B_v, s_v, bias_v = _lora_tensors(module.v_proj, allow_bias=True)
    modeling = importlib.import_module(module.__class__.__module__)

    def forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        dtype = hidden_states.dtype

        query_states, key_states, value_states = lora_qkv_v3(
            hidden_states,
            W_q,
            A_q.to(dtype),
            B_q.to(dtype),
            s_q,
            W_k,
            A_k.to(dtype),
            B_k.to(dtype),
            s_k,
            W_v,
            A_v.to(dtype),
            B_v.to(dtype),
            s_v,
            W_all=None,
        )
        if bias_q is not None:
            query_states = query_states + bias_q.to(dtype)
        if bias_k is not None:
            key_states = key_states + bias_k.to(dtype)
        if bias_v is not None:
            value_states = value_states + bias_v.to(dtype)

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

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)

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

        attn_output, attn_weights = attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = module.o_proj(attn_output)
        return attn_output, attn_weights

    return forward

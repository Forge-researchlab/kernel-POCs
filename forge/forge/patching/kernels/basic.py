"""Patch adapters for simple module-forward kernels."""
from __future__ import annotations


def make_embedding_forward(module, config):
    """nn.Embedding.forward -> ForgeEmbeddingFunction.apply(weight, indices, padding_idx).

    Forward ``module.padding_idx`` to the kernel so the pad row's gradient is
    zeroed in backward, matching ``nn.Embedding`` semantics.
    """
    from forge.kernels.embedding import ForgeEmbeddingFunction

    padding_idx = module.padding_idx

    def forward(input_ids):
        return ForgeEmbeddingFunction.apply(module.weight, input_ids, padding_idx)

    return forward


def make_rmsnorm_forward(module, config):
    """RMSNorm.forward -> apply_rmsnorm(x, weight, eps, offset, casting_mode, in_place).

    ``in_place`` controls the v4 backward dY→dX optimization. It is safe by
    default for Qwen/Llama-style RMSNorm (offset=0), but disabled by default for
    Gemma residual-paired RMSNorm (offset=1) where another backward consumer may
    still need dY.
    """
    from forge.kernels.rmsnorm import apply_rmsnorm

    offset = float(config.get("offset", 0.0))
    casting_mode = config.get("casting_mode", "gemma" if offset == 1.0 else "llama")
    in_place = bool(config.get("in_place", offset != 1.0))
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

    def forward(hidden_states):
        return apply_rmsnorm(
            hidden_states,
            module.weight,
            eps=eps,
            offset=offset,
            casting_mode=casting_mode,
            in_place=in_place,
        )

    return forward


def make_geglu_forward(module, config):
    """Gemma/Gemma2 MLP.forward -> down_proj(geglu(gate_proj(x), up_proj(x))).

    The GeGLU kernel fuses only ``gelu(gate) * up``; projections remain module
    calls so PEFT wrappers and live weights continue to work. The GELU variant
    is inferred from the HF config unless mapping config supplies
    ``approximate='tanh'`` or ``approximate='none'`` explicitly.
    """
    from forge.kernels.geglu import geglu

    explicit_mode = config.get("approximate")
    if explicit_mode is not None:
        approximate = explicit_mode
    else:
        hf_cfg = getattr(module, "config", None)
        act_name = None
        if hf_cfg is not None:
            act_name = getattr(hf_cfg, "hidden_activation", None) or getattr(hf_cfg, "hidden_act", None)
        exact = {"gelu", "gelu_python"}
        tanh = {"gelu_pytorch_tanh", "gelu_new"}
        if act_name in exact:
            approximate = "none"
        elif act_name in tanh:
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


def make_swiglu_forward(module, config):
    """Qwen MLP.forward -> down_proj(Forge SwiGLU(gate_proj(x), up_proj(x)))."""
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


def make_not_implemented(kernel_name: str, why: str):
    """Factory for declared-but-unwired kernels.

    ``forge.patch(model)`` skips these, while explicit ``kernels=[...]`` requests
    fail during pre-validation in core.py.
    """

    def factory(module, config):
        def forward(*args, **kwargs):
            raise NotImplementedError(
                f"Forge kernel {kernel_name!r} is not implemented yet ({why}). "
                f"Use forge.patch(model, kernels=[<built kernels only>]) to skip it."
            )

        return forward

    factory.__forge_stub__ = True
    factory.__forge_kernel_name__ = kernel_name
    factory.__forge_reason__ = why
    return factory

"""Patch adapters for simple module-forward kernels."""
from __future__ import annotations


def make_embedding_forward(module, config):
    """nn.Embedding.forward -> ForgeEmbeddingFunction.apply(weight, indices)."""
    from forge.kernels.embedding import ForgeEmbeddingFunction

    def forward(input_ids):
        return ForgeEmbeddingFunction.apply(module.weight, input_ids)

    return forward


def make_rmsnorm_forward(module, config):
    """RMSNorm.forward -> apply_rmsnorm(x, weight, eps, offset, casting_mode)."""
    from forge.kernels.rmsnorm import apply_rmsnorm

    offset = float(config.get("offset", 0.0))
    casting_mode = config.get("casting_mode", "gemma" if offset == 1.0 else "llama")
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))

    def forward(hidden_states):
        return apply_rmsnorm(
            hidden_states,
            module.weight,
            eps=eps,
            offset=offset,
            casting_mode=casting_mode,
        )

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

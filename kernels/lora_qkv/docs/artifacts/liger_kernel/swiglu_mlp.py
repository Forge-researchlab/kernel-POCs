"""
EXACT CODE extracted from Liger-Kernel.

Source: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/swiglu.py
License: BSD-2-Clause
Retrieved: 2026-05-24
Commit: main (latest at retrieval)

DO NOT MODIFY this code — it is a verbatim copy for reference.
Our annotations are in clearly-separated blocks marked with "# === OUR ANALYSIS ===".
"""

# ============================================================
# ORIGINAL CODE (verbatim from repo)
# ============================================================

import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.ops import LigerSiLUMulFunction

class LigerSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

class LigerBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))

class LigerExperts(nn.Module):
    """
    Patch MixtralExperts for transformers v5 or later to use LigerSiLUMulFunction
    https://github.com/huggingface/transformers/blob/393b4b3d28e29b4b05b19b4b7f3242a7fc893637/src/transformers/models/mixtral/modeling_mixtral.py#L63
    """

    def __init__(self, config):
        super().__init__()
        if hasattr(config, "num_experts"):
            # qwen3_moe, qwen3_next uses num_experts
            self.num_experts = config.num_experts
        else:
            self.num_experts = config.num_local_experts
        if hasattr(config, "moe_intermediate_size"):
            # qwen3_moe, qwen3_next uses moe_intermediate_size
            self.intermediate_dim = config.moe_intermediate_size
        else:
            self.intermediate_dim = config.intermediate_size

        self.hidden_dim = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Reshape to 2D if needed (e.g. batch × seq → tokens)
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_dim)

        # top_k_index / top_k_weights may come in as (batch, seq, K) or (T, K)
        top_k_index_2d = top_k_index.view(-1, top_k_index.shape[-1]).to(torch.int32)
        top_k_weights_2d = top_k_weights.view(-1, top_k_weights.shape[-1])

        out = LigerFusedMoEFunction.apply(x, self.gate_up_proj, self.down_proj, top_k_index_2d, top_k_weights_2d)
        return out.view(orig_shape)

class LigerPhi3SwiGLUMLP(nn.Module):
    """
    Patch Phi3MLP to use LigerSiLUMulFunction
    https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(LigerSiLUMulFunction.apply(gate, up_states))

class LigerQwen3MoeSwiGLUMLP(nn.Module):
    """
    Patch Qwen3MoeMLP to use LigerSiLUMulFunction.
    https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L57
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

class LigerHunyuanV1SwiGLUMLP(nn.Module):
    def __init__(self, config, layer_idx=None, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.layer_idx = layer_idx
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

class LigerFalconH1SwiGLUMLP(nn.Module):
    """
    Patch FalconH1MLP to use LigerSiLUMulFunction with gate / down multipliers.

    Falcon H1's MLP block pre-scales the gate pre-activation and post-scales the
    down projection output:

    y = down_proj(silu(gate_proj(x) * gate_mult) * up_proj(x)) * down_mult

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/modeling_falcon_h1.py
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
        gate_multiplier, down_multiplier = config.mlp_multipliers
        self.gate_multiplier = float(gate_multiplier)
        self.down_multiplier = float(down_multiplier)

    def forward(self, x):
        return self.down_proj(
            LigerSiLUMulFunction.apply(
                self.gate_proj(x),
                self.up_proj(x),
                float(self.gate_multiplier),
                float(self.down_multiplier),
            )
        )


# ============================================================
# OUR ANALYSIS — How this relates to our LoRA QKV kernel project
# ============================================================
#
# These are nn.Module wrappers that replace HuggingFace MLP implementations
# with Liger's fused SwiGLU kernel. They demonstrate the MODULE REPLACEMENT
# PATTERN for integrating Triton kernels into HuggingFace models.
#
# Key pattern (applicable to our LoRA QKV project):
#
# 1. MATCH THE HUGGINGFACE __init__ SIGNATURE
#    Each wrapper takes the same `config` object as the original HuggingFace
#    module, creating identical nn.Linear layers. This means pretrained weights
#    load directly without any conversion.
#
# 2. KEEP CUBLAS FOR MATMULS
#    The linear layers (gate_proj, up_proj, down_proj) remain standard nn.Linear.
#    cuBLAS handles the matmuls. Only the ACTIVATION (silu * multiply) is replaced
#    with a Triton kernel. This is the "cuBLAS for matmul + Triton for fusion"
#    pattern that has proven most effective.
#
# 3. AUTOGRAD.FUNCTION AS THE BRIDGE
#    LigerSiLUMulFunction.apply() is called inside forward(), and PyTorch's
#    autograd handles the backward pass automatically. No manual gradient
#    computation in the nn.Module — that's all in the Function.
#
# 4. ARCHITECTURE VARIANTS
#    Different models have different MLP structures:
#      - LLaMA/Mistral: separate gate_proj + up_proj (2 matmuls)
#      - Phi-3: packed gate_up_proj then chunk (1 matmul, then split)
#      - Mixtral MoE: expert routing + per-expert MLPs
#      - FalconH1: gate/down multiplier scaling
#    All use the SAME Triton kernel (LigerSiLUMulFunction) — only the
#    nn.Module wrapper changes.
#
# 5. RELEVANCE TO OUR PROJECT
#    When we build our LoRA QKV drop-in replacement, we'll follow a similar
#    pattern: create an nn.Module or monkey-patch function that extracts
#    LoRA parameters and calls our fused autograd.Function. The key difference
#    is that our kernel fuses the MATMUL itself (not just the activation),
#    which is more complex but also more impactful for performance.

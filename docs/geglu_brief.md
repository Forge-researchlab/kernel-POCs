# GEGLU Brief

Stage: Generate checkpoint  
Date: 2026-05-24

## Recommendation

Proceed to Stage 3 validation on the RunPod A100 with the flattened standalone GEGLU implementation.

Implemented:

- `geglu(gate, up, approximate="tanh" | "none", preserve_inputs=False)`
- `geglu_packed(gate_up, approximate=..., preserve_inputs=False)`
- forward + backward + CPU PyTorch fallback
- benchmark against PyTorch, Liger, and Unsloth

## Technical Conclusion

GEGLU is elementwise. Row boundaries are unnecessary. Liger's row-wise power-of-two launch wastes a large fraction of lanes for common intermediate sizes, while Unsloth's flattened layout is better but coupled to its LoRA MLP graph. Forge can combine the better flattened launch with a modular standalone API.

Default GELU mode should be `approximate="tanh"` for Gemma parity. Exact GELU is included for compatibility.

## Competitive Position

Versus Liger:

- Expected Forge advantage: flattened indexing, exact + tanh support, packed layout, safe input-preserving backward, no ordinary hidden-size block cap.
- Liger baseline: tanh-only row-wise GEGLU, model patching for Gemma, tiled long-sequence wrapper.

Versus Unsloth:

- Unsloth already has flattened exact/tanh GEGLU kernels, but they are shaped around Unsloth's LoRA MLP backward convention and 3D wrappers.
- Forge's first advantage is modularity and clean standalone testing; Unsloth's coupled LoRA path remains the specialized LoRA baseline.

## Next-Stage Plan

1. Run `python -m pytest tests/test_geglu.py -xvs` on the A100 environment.
2. Run the smoke benchmark for bf16 forward/backward/full modes.
3. Run the A100 benchmark suite for bf16 forward/full modes and save CSV.
4. Report failures, timing rows, device name, PyTorch/Triton versions, and any skipped providers.

Suggested commands:

```bash
cd kernel-POCs
python -m pytest tests/test_geglu.py -xvs
python kernels/geglu/benchmarks/benchmark_geglu.py --suite smoke --dtype bf16 --modes forward backward full
python kernels/geglu/benchmarks/benchmark_geglu.py --suite a100 --dtype bf16 --modes forward full --csv geglu_a100_bf16.csv
```

## Main Evidence

- HuggingFace Gemma uses GeGLU and defaults `hidden_act` to `gelu_pytorch_tanh`.
- HF PR #29402 documents that Gemma should use approximate GELU rather than exact GELU.
- Liger current GEGLU is tanh-only and row-wise.
- Unsloth current GEGLU has exact/tanh flattened kernels but is integrated into LoRA MLP, not a general Forge-style standalone autograd op.
- First principles: activation-only GEGLU is bandwidth/launch sensitive; dense full-MLP fusion is matmul dominated and has to compete with cuBLAS.

## Risks

- Flattened Forge may trail Unsloth's flattened kernels on raw activation speed.
- Exact GELU uses heavier special functions and may be slower; it should be compatibility mode.
- In-place backward saves memory but has higher-order-gradient and tensor-aliasing limitations.
- Full dense MLP fusion may be a poor first target without a narrow shape regime.
- Local validation has not run because the current shell lacks `torch`; A100 validation is the hard gate.

## Rejected Alternatives

- Tanh-only Liger clone: not enough delta.
- Exact-only kernel: wrong default for Gemma.
- Full dense MLP fusion first: high complexity and uncertain speedup.
- PyTorch-only wrapper: not a Forge kernel contribution.

## Go/No-Go

Go to Stage 3 validation with the current implementation.

# GEGLU Brief

Stage: Generate checkpoint, post A100 smoke
Date: 2026-05-24

## Recommendation

Proceed with packed gate+up GEGLU MLP as the next implementation track.

Implemented:

- `geglu(gate, up, approximate="tanh" | "none", preserve_inputs=False)`
- `geglu_packed(gate_up, approximate=..., preserve_inputs=False)`
- `geglu_mlp(...)` with separate or packed gate+up weights
- `pack_geglu_gate_up_weight(...)` and `pack_geglu_gate_up_bias(...)`
- forward + backward for the activation primitive
- benchmarks against PyTorch and Liger; Unsloth isolated runner exists but package import failed on RunPod

## Technical Conclusion

GEGLU is elementwise. Row boundaries are unnecessary. The standalone Forge primitive is mainly a memory/support baseline. The measured performance delta comes from packing gate and up projection into one GEMM, then using packed GEGLU.

Default GELU mode should be `approximate="tanh"` for Gemma parity. Exact GELU is included for compatibility.

## A100 Smoke Result

- Activation-only Forge saves memory but is not a consistent latency win.
- Packed gate+up wins strongly on small-token/odd-intermediate shapes.
- `B2 S7 H4096 I11009`: `packed_forge mlp_full` `1.005 ms` vs Liger `1.536 ms`; `packed_forge gateup_forward` `0.188 ms` vs Liger `0.457 ms`.
- `B1 S32 H4096 I11008`: `packed_forge mlp_full` `0.706 ms` vs Liger `0.792 ms`.

## Competitive Position

Versus Liger: Forge has packed gate+up projection support, exact + tanh support, packed activation layout, safe input-preserving backward, and no ordinary hidden-size row block cap.

Versus Unsloth: public data does not expose comparable GEGLU A100 microbenchmarks. The local isolated runner failed because Unsloth's top-level import hit a `torch._inductor.config` compatibility issue in the RunPod env.

## Next-Stage Plan

1. Validate `geglu_mlp` packed/separate correctness on A100.
2. Run full `--suite a100` gate+up benchmark.
3. Build tiled GEGLU MLP around packed gate+up with recompute in backward.
4. Port the existing LoRA MLP work to a GEGLU activation variant without reusing Unsloth's patching path.

Suggested commands:

```bash
cd kernel-POCs
python -m pytest tests/test_geglu.py -xvs
python kernels/geglu/benchmarks/benchmark_geglu_gate_up_fusion.py --suite smoke --dtype bf16 --modes gateup_forward gateup_full mlp_forward mlp_full --check
python kernels/geglu/benchmarks/benchmark_geglu_gate_up_fusion.py --suite a100 --dtype bf16 --modes gateup_forward gateup_full mlp_forward mlp_full --csv geglu_gateup_a100_bf16.csv --check
```

## Risks

- Flattened Forge may trail Unsloth's flattened kernels on raw activation speed.
- Exact GELU uses heavier special functions and should remain compatibility mode.
- In-place backward saves memory but has higher-order-gradient and tensor-aliasing limitations.
- Full dense MLP fusion remains risky unless scoped to low-token, low-rank LoRA, or memory-bound regimes.
- Local validation has not run because the current shell lacks `torch`; A100 validation is the hard gate.

## Go/No-Go

Go on packed gate+up GEGLU MLP. Keep activation-only GEGLU as a control primitive.

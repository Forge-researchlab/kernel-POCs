# GEGLU Research

Stage: Generate checkpoint  
Date: 2026-05-24  
Local repo state: `kernel-POCs` branch `main` at `c6f69c3`; `Liger-Kernel` refreshed to `bfaaef8`; `unsloth` refreshed to `56e9046b`.

## Recommendation

Stage 2 implements Forge GEGLU as a flattened standalone activation primitive:

1. Standalone `geglu(gate, up, approximate="tanh" | "none", preserve_inputs=False)` with a CPU PyTorch fallback.
2. Packed `geglu_packed(gate_up, approximate=..., preserve_inputs=False)` for `[..., 2 * hidden]`.
3. A 1D flattened Triton launch over total elements rather than Liger's one-row-per-program launch.
4. Both forward and backward, with exact and tanh GELU variants.
5. Benchmarks against PyTorch, Liger `LigerGELUMulFunction`, and Unsloth's exact/tanh GEGLU kernels where importable.

The intended delta is not merely "GEGLU exists in Forge." It is:

- Beat or at least expose a measurable speed gap versus Liger on odd and large intermediate sizes by avoiding per-row power-of-two padding.
- Broaden support beyond Liger by adding exact GELU, packed layout, no hidden-size block cap for ordinary flattened indexing, and a safe `preserve_inputs` mode.
- Keep the activation primitive modular so it can later feed a fused LoRA MLP or tiled MLP experiment.

Dense full-MLP fusion is tracked as a later Tier 3 experiment. It can reduce activation materialization, but in normal training the gate/up pre-activations are still needed for backward unless we intentionally recompute them. A first-pass fused dense GEMM is likely to lose to cuBLAS on large Gemma shapes unless it targets a specific low-batch, small-rank LoRA, MoE, quantized, or long-sequence memory regime.

## Operation

GEGLU is the gated feed-forward activation:

```python
out = gelu(gate) * up
```

For Gemma-family HuggingFace models, the relevant activation is `gelu_pytorch_tanh`, i.e. PyTorch GELU with `approximate="tanh"`. Exact `approximate="none"` is still useful for broader model compatibility and tests, but it should not be the default for Gemma parity.

### PyTorch Reference

```python
import torch


def torch_geglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    approximate: str = "tanh",
) -> torch.Tensor:
    if approximate not in {"tanh", "none"}:
        raise ValueError("approximate must be 'tanh' or 'none'")
    activated = torch.nn.functional.gelu(
        gate.to(torch.float32),
        approximate=approximate,
    ).to(dtype=gate.dtype)
    return activated * up


def torch_geglu_packed_reference(
    gate_up: torch.Tensor,
    approximate: str = "tanh",
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return torch_geglu_reference(gate, up, approximate=approximate)
```

Reference behavior:

- Backward baseline uses PyTorch autograd.
- The reference intentionally computes GELU in fp32, then casts to input dtype before multiplying by `up`, matching Liger/Unsloth's activation cast boundary and HuggingFace's dtype-level behavior.
- Tolerances should start at `1e-5` for fp32 and `2e-2` to `3e-2` for fp16/bf16, then tighten with real CUDA data.

## Math

Let `z = gate`. Let upstream gradient be `dout`.

Exact GELU:

```text
gelu_exact(z) = z * Phi(z)
              = 0.5 * z * (1 + erf(z / sqrt(2)))

d gelu_exact / dz =
    0.5 * (1 + erf(z / sqrt(2)))
  + z * exp(-0.5 * z^2) / sqrt(2 * pi)
```

Tanh approximation:

```text
k = sqrt(2 / pi)
b = 0.044715
u = k * (z + b * z^3)
t = tanh(u)

gelu_tanh(z) = 0.5 * z * (1 + t)

d gelu_tanh / dz =
    0.5 * (1 + t)
  + 0.5 * z * (1 - t^2) * k * (1 + 3 * b * z^2)
```

GEGLU gradients:

```text
activated = gelu(z)

dup   = dout * activated
dgate = dout * up * gelu_prime(z)
```

If a later API adds scalar `gate_multiplier` and `down_multiplier`, the generalized form is:

```text
z = gate * gate_multiplier
out = gelu(z) * up * down_multiplier

dup   = dout * down_multiplier * gelu(z)
dgate = dout * down_multiplier * up * gelu_prime(z) * gate_multiplier
```

I do not recommend adding the multipliers in the initial GEGLU API unless we have a concrete model needing them. SwiGLU needed them for Falcon H1 style compatibility; GEGLU's immediate Gemma use does not.

## Supported Surface

Initial Forge target:

- `gate` and `up` have identical shape `(..., hidden)`.
- `gate_up` packed shape is `(..., 2 * hidden)`, gate first and up second.
- CUDA dtypes: fp32, fp16, bf16.
- CPU fallback: PyTorch reference.
- Layout: wrapper may call `.contiguous()` for CUDA path; non-contiguous stride-native Triton can be deferred.
- Shapes: arbitrary positive total element count; hidden can be odd.
- Exact and tanh GELU variants.
- `preserve_inputs=False` may reuse saved buffers in backward; `preserve_inputs=True` allocates separate gradient tensors and leaves inputs intact.

Unsupported in first pass:

- Quantized integer inputs.
- Sparse inputs.
- Tensor-valued per-channel scaling.
- Fused dense gate/up/down projection.
- Higher-order gradients through the Triton backward path.
- DTensor/distributed tensor semantics.

## Forge Existing State

Closest Forge implementation:

- `kernel-POCs/kernels/swiglu/swiglu.py`
- `kernel-POCs/tests/test_swiglu.py`
- `kernel-POCs/kernels/swiglu/benchmarks/benchmark_swiglu.py`
- `kernel-POCs/docs/swiglu.md`

Useful local patterns from SwiGLU:

- Autograd wrapper with `preserve_inputs`.
- Separate and packed input support.
- CPU fallback.
- Benchmarks comparing PyTorch, Forge fast, Forge safe, packed variants, and Liger.

Needed divergence from current SwiGLU:

- GEGLU should use a flattened 1D launch as the primary kernel instead of a row-wise launch. GEGLU is purely elementwise; there is no row reduction or row-local recurrence. Row-wise launch adds substantial masked-lane waste for Gemma and odd hidden sizes.
- GEGLU should expose `approximate` as a correctness surface.

Environment note:

- The current shell environment does not have `torch` installed. Stage 3 correctness and benchmark validation will run on the user's RunPod A100 environment.

## Competitor Study: Liger

Local refreshed checkout:

- Repo: `Liger-Kernel/`
- Branch: `main`
- Commit after refresh: `bfaaef8`
- Working tree: clean

Relevant paths:

- `Liger-Kernel/src/liger_kernel/ops/geglu.py`
- `Liger-Kernel/src/liger_kernel/transformers/geglu.py`
- `Liger-Kernel/src/liger_kernel/transformers/tiled_mlp.py`
- `Liger-Kernel/src/liger_kernel/ops/tiled_mlp.py`
- `Liger-Kernel/test/transformers/test_geglu.py`
- `Liger-Kernel/test/transformers/test_tiled_mlp.py`
- `Liger-Kernel/benchmark/scripts/benchmark_geglu.py`

API and fusion boundary:

- Low-level op: `LigerGELUMulFunction.apply(a, b)`.
- Functional wrapper: `liger_geglu(a, b)`.
- Model wrapper: `LigerGEGLUMLP`, replacing Gemma MLP forward with `down_proj(LigerGELUMulFunction.apply(gate_proj(x), up_proj(x)))`.
- Fusion is activation-only: `gelu_tanh(gate) * up`.
- The dense projections remain ordinary linear layers.
- The tiled MLP wrapper chunks sequence rows and recomputes the MLP during backward to reduce long-sequence activation memory; it does not introduce a new fused matmul kernel.

Supported behavior:

- Tanh GELU approximation only.
- Same-shape `a` and `b`, reshaped to `(-1, hidden)`.
- Contiguity enforced by wrapper.
- fp32 and bf16 are tested; fp16 is structurally likely but not the primary tested dtype in the GEGLU tests I inspected.
- Gemma 1/2/3/4-style patching is present in current local code.

Optimization strategy:

- One Triton program per flattened row.
- `BLOCK_SIZE = next_power_of_2(hidden)` with a max fused size of 65536 through `calculate_settings`.
- Computes GELU in fp32, casts activated gate to `b` dtype, then multiplies by `b`.
- Backward recomputes the activation and derivative instead of saving activated output.
- Backward writes gradients in place into saved `a` and `b` buffers.

Drawbacks and gaps:

- Exact GELU is explicitly not implemented in the wrapper comments.
- Hidden sizes above the max fused block size fail.
- Odd hidden sizes waste lanes. For Gemma 7B `intermediate_size=24576`, a row-wise power-of-two block is 32768, so one third of lanes are masked. For Qwen/Llama-like `11008`, block is 16384, so about one third of lanes are also masked. For irregular test size `4231`, block is 8192, so nearly half the lanes are masked.
- It mutates saved inputs in backward, which is good for memory but unsafe for higher-order gradients and surprising if a caller expects saved tensors not to be overwritten.
- No packed `[..., 2 * hidden]` activation path.
- No standalone long-index flattened path.
- No LoRA-specific GEGLU integration; LoRA remains outside Liger's GEGLU.

## Competitor Study: Unsloth

Local refreshed checkout:

- Repo: `unsloth/`
- Branch: `main`
- Commit after refresh: `56e9046b`
- Working tree: clean

Relevant paths:

- `unsloth/unsloth/kernels/geglu.py`
- `unsloth/unsloth/kernels/swiglu.py`
- `unsloth/unsloth/kernels/fast_lora.py`
- `unsloth/unsloth/models/gemma.py`
- `unsloth/unsloth/models/gemma2.py`

API and fusion boundary:

- `geglu_exact_forward_kernel(gate, up)`
- `geglu_exact_backward_kernel(DW, e, g)`
- `geglu_approx_forward_kernel(gate, up)`
- `geglu_approx_backward_kernel(DW, e, g)`
- These are used inside Unsloth's custom `LoRA_MLP` autograd path for LoRA/QLoRA MLP training.
- Unsloth also has `fast_geglu_inference` for Gemma inference, but that path uses PyTorch `F.gelu(..., approximate="tanh")` and in-place multiply rather than a Triton GEGLU standalone op.

Optimization strategy:

- Flattened 1D launch over `n_elements` with `BLOCK_SIZE = 1024`.
- Uses int64 offsets when total elements approach int32 limits.
- Has both exact and tanh approximate kernels.
- Forward computes GELU in fp32, casts to input dtype, then multiplies.
- Backward overwrites three buffers in place for the LoRA MLP path: upstream `DW` becomes activation output `h`, `e` becomes the up-branch gradient-like term, and `g` becomes gate gradient.

Drawbacks and gaps:

- The public shape wrapper assumes 3D `(batch, seq_len, hidden)` for forward.
- The backward kernels are optimized for Unsloth's LoRA custom graph, not a general PyTorch `autograd.Function` returning `(dgate, dup)` for standalone use.
- It is coupled to Unsloth's device utilities, quantization, and LoRA plumbing.
- The exact/tanh support is good, but not packaged as a modular Forge-style kernel with separate tests and benchmarks.

## First-Principles Analysis

### Work That Is Unavoidable

Forward per element:

- Read `gate`.
- Read `up`.
- Compute GELU.
- Multiply by `up`.
- Write output.

Backward per element:

- Read `dout`.
- Read saved `gate`.
- Read saved `up`.
- Recompute GELU and derivative.
- Write `dgate` and `dup`.

Exact GELU requires `erf` in forward and `erf + exp` in backward. Tanh approximate requires a cubic polynomial, `tanh`, and derivative algebra. For Gemma parity, tanh is the primary path.

### Memory Traffic

Standalone fused forward lower bound is roughly:

```text
read gate + read up + write out = 3N dtype elements
```

Naive PyTorch eager usually materializes activated gate:

```text
read gate + write activated + read activated + read up + write out
```

So the activation kernel mostly removes an intermediate activation write/read pair and one Python/CUDA launch boundary. It cannot remove gate/up materialization because those are outputs of separate projection matmuls.

Backward lower bound is:

```text
read dout + read gate + read up + write dgate + write dup = 5N dtype elements
```

In-place backward can reduce allocation pressure by writing gradients into saved gate/up buffers. It does not change the logical traffic much, but it reduces peak live tensors.

### Why Flattened 1D Is the Right First Delta

GEGLU has no reduction across the hidden dimension. Row boundaries are irrelevant for correctness. A row-wise kernel is convenient but not mathematically required.

A flattened kernel:

- Uses almost every lane except the final total-element tail block.
- Avoids hidden-size power-of-two waste.
- Avoids the 65536 row block limit.
- Improves tiny-row or irregular-shape occupancy.
- Matches Unsloth's core layout idea while exposing a cleaner Forge standalone API.

This is the clearest Stage 2 delta over Liger with limited implementation risk.

### Fusion Boundaries

Activation-only GEGLU:

- Low implementation risk.
- Clear benchmark comparison.
- Modest speedup over PyTorch eager and possible speedup over Liger for odd/large hidden sizes.
- Does not attack the dominant dense GEMM compute.

Gate+up projection fusion:

- Potentially reuses the input tile for two output projections.
- Can compute activation in the epilogue while accumulators are fp32.
- Still needs to save gate/up pre-activations for training backward unless the design recomputes them.
- Custom Triton dense GEMM must compete with cuBLAS; this is not a guaranteed win on normal large GEMMs.

Full gate+up+activation+down fusion:

- Avoids materializing post-activation `h` in forward if streaming over intermediate chunks.
- The expression nests two reductions: gate/up projections reduce over hidden input, then down projection reduces over intermediate.
- Efficient implementation resembles a custom multi-stage blocked MLP, not a simple epilogue.
- Training still needs either saved pre-activations or recomputation in backward.
- This is a Tier 3 research kernel, not the right first correctness artifact.

LoRA MLP fusion:

- More promising for Forge's roadmap than dense full-MLP fusion because LoRA ranks are small and separate LoRA matmuls are launch/memory-bound.
- Unsloth already demonstrates a layer-level LoRA MLP autograd boundary for GEGLU exact/tanh.
- Forge can later adapt this idea with modular kernel boundaries and better tests rather than copying Unsloth's framework coupling.

Tiled MLP:

- Liger's tiled MLP chunks sequence rows and recomputes during backward, targeting memory at long sequence length.
- This is useful for very long context, but it is a layer wrapper strategy rather than a new GEGLU primitive.
- Forge can use this as a later wrapper around the activation primitive.

## Stage 2 Implementation

Generated files:

- `kernel-POCs/kernels/geglu/geglu.py`
- `kernel-POCs/kernels/geglu/__init__.py`
- `kernel-POCs/tests/test_geglu.py`
- `kernel-POCs/kernels/geglu/benchmarks/benchmark_geglu.py`

Selected design:

- Separate-input API: `geglu(gate, up, approximate="tanh", preserve_inputs=False)`.
- Packed API: `geglu_packed(gate_up, approximate="tanh", preserve_inputs=False)`, with gate first and up second.
- CPU fallback uses the PyTorch reference.
- CUDA path supports fp32, fp16, and bf16.
- `approximate="tanh"` is the Gemma default; `approximate="none"` uses exact GELU.
- Forward and backward use a 1D flat launch with `FLAT_BLOCK_SIZE = 1024`.
- Packed forward/backward flatten output elements and compute row/column offsets into `[..., 2 * hidden]`, so packed layouts also avoid row-wise power-of-two padding.
- Backward recomputes GELU and its derivative rather than saving activated outputs.
- `preserve_inputs=False` writes gradients into saved contiguous buffers to reduce allocation pressure. `preserve_inputs=True` allocates separate gradient buffers for safer debugging and input-preservation tests.
- Long indexing is enabled when total element count approaches int32 offset limits.

Implementation notes:

- The tanh path computes `0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))`.
- The exact path computes `0.5 * x * (1 + erf(x / sqrt(2)))`.
- GELU is computed in fp32 and cast back to the input dtype before multiplication, matching the reference cast boundary used in Liger and Unsloth.
- The kernels guard the forward activation for `+/-inf` to match PyTorch special-value behavior more closely.

Correctness plan:

- Top-level tests cover separate and packed APIs, forward/backward parity, `approximate="tanh"` and `"none"`, fp32/fp16/bf16, odd hidden sizes, preserve-input mode, CPU fallback, invalid inputs, special forward values, and a `hidden=65537` flat-path case.
- Gradcheck is deferred because the Triton CUDA path targets fp32/fp16/bf16, while PyTorch gradcheck expects double precision. Backward parity against PyTorch autograd is the hard gate for the POC.

Benchmark plan:

- Local script benchmarks forward, backward, and full fwd+bwd modes where applicable.
- Providers: PyTorch, Forge fast, Forge safe, PyTorch packed, Forge packed fast, Forge packed safe, Liger when compatible, and optional Unsloth forward if importable.
- Suites include smoke shapes and A100-focused shapes: typical Gemma/Qwen sizes, odd hidden sizes, `intermediate_size=24576`, and `hidden=65537` where Liger's row-wise kernel is skipped.

A100 validation commands:

```bash
cd kernel-POCs
python -m pytest tests/test_geglu.py -xvs
python kernels/geglu/benchmarks/benchmark_geglu.py --suite smoke --dtype bf16 --modes forward backward full
python kernels/geglu/benchmarks/benchmark_geglu.py --suite a100 --dtype bf16 --modes forward full --csv geglu_a100_bf16.csv
```

## External Research Notes

Sources checked:

- PyTorch GELU docs define exact `approximate="none"` and tanh `approximate="tanh"` forms: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.gelu.html
- HuggingFace Gemma docs state Gemma uses GeGLU and defaults `hidden_act` to `gelu_pytorch_tanh`: https://huggingface.co/docs/transformers/model_doc/gemma
- HuggingFace `GemmaMLP` source applies `act_fn(gate_proj(x)) * up_proj(x)`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
- HuggingFace activations source maps `gelu_pytorch_tanh` to `nn.functional.gelu(..., approximate="tanh")`: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- HuggingFace PR #29402 records the Gemma compatibility issue: approximate GELU, not exact GELU, is the correct default for Gemma checkpoints: https://github.com/huggingface/transformers/pull/29402
- HuggingFace issue #21344 explains why PyTorch's fused tanh GELU path replaced older Python multi-op approximations: https://github.com/huggingface/transformers/issues/21344
- Shazeer's GLU variants paper motivates GEGLU/SwiGLU style feed-forward gating: https://arxiv.org/abs/2002.05202
- Liger's paper describes operation fusion as a way to reduce HBM/SRAM traffic and lists GeGLU as a tanh-approximation activation fusion: https://arxiv.org/html/2410.10989
- Triton matmul tutorial shows fp32 accumulation and activation epilogue fusion before downcast: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton fused softmax tutorial quantifies the memory-traffic reason for fusing bandwidth-bound operations: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

Research conclusion:

- Tanh should be the default for Gemma parity.
- Exact should be supported as a compatibility/broadening feature.
- The near-term measurable delta should come from launch/layout/API improvements over Liger, not an unproven dense GEMM replacement.
- Full fusion remains interesting, but the correct target is likely LoRA or tiled long-sequence MLP first, not ordinary dense Gemma MLP.

## Competitive Position

Forge should proceed only if Stage 2 is treated as:

1. A stronger standalone activation primitive than Liger: tanh plus exact, flattened indexing, packed layout, safe mode, and no ordinary hidden-size block cap.
2. A benchmark-controlled base for later fused GEGLU LoRA MLP work.

Expected competitor position:

- Versus Liger: Forge should beat or match on irregular/odd hidden sizes and broaden API support. Liger remains a strong model-patching baseline for Gemma.
- Versus Unsloth: Forge likely will not beat Unsloth's coupled LoRA MLP path on its home workload in the first standalone kernel. Forge's value is modular standalone coverage and a cleaner path to controlled variants.
- Versus PyTorch eager: Forge should reduce activation intermediate traffic and launches.

Go/no-go criteria for Stage 3:

- Correctness passes on RunPod A100 for forward/backward, packed/separate, fp32/fp16/bf16, odd hidden sizes, and special values.
- The benchmark separately reports forward, backward, and full fwd+bwd for PyTorch, Forge flattened, Liger, and Unsloth where available.
- If flattened Forge does not beat Liger in the predicted odd-size regimes, record the result and decide whether to continue into LoRA fusion.

## Tier Classification

Standalone GEGLU is Tier 1: pure elementwise, no cross-column reductions.

Fused GEGLU MLP and LoRA GEGLU MLP are Tier 3: multi-stage matmul/activation composition, possibly custom autograd and recomputation.

## Rejected Alternatives

- Exact-only implementation: wrong default for Gemma and leaves Liger parity untested.
- Tanh-only Liger clone: too little delta and no improvement over the competitive bar.
- Full dense MLP fusion as first artifact: high complexity and likely cuBLAS disadvantage without a narrowed shape target.
- Row-wise Forge GEGLU copied from current SwiGLU: easy but misses the strongest first-principles improvement.
- Only implementing a PyTorch wrapper around `F.gelu(...)*up`: not a kernel contribution.

## Open Questions

- Should Stage 2 include exact support in the same kernel via constexpr branch, or separate exact/tanh kernels for compile-time clarity?
- Should packed support be implemented immediately for parity with Forge SwiGLU, or delayed if the first benchmark suite is already large?
- Should we add a `geglu_mlp` module wrapper for Gemma-style `gate_proj/up_proj/down_proj`, or keep Stage 2 strictly activation-level?
- Which GPU should be the first benchmark target: A100 or H100?

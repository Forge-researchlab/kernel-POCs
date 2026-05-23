# SwiGLU Research

Stage: Analyze checkpoint draft  
Date: 2026-05-23  
Local repo state: `kernel-POCs` on branch `main`; `docs/swiglu.md` and `tests/test_swiglu.py` were empty before this analysis.

## Operation Math

Forward equation for the standalone activation:

```python
out = silu(gate) * up
out = gate * sigmoid(gate) * up
```

Optional multiplier-compatible form, matching recent Liger support and Falcon H1 style MLPs:

```python
out = silu(gate * gate_multiplier) * up * down_multiplier
```

Backward gradients for `out = silu(g * gm) * u * dm`, with upstream gradient `dout`:

```python
z = g * gm
sig = sigmoid(z)
silu_z = z * sig
dout_scaled = dout * dm

dup = dout_scaled * silu_z
dgate = dout_scaled * up * (sig + silu_z * (1 - sig)) * gm
```

For the default case `gm = dm = 1`, this reduces to:

```python
dup = dout * silu(gate)
dgate = dout * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
```

Numerical stability:

- `sigmoid` and `silu` should be computed in fp32 for fp16/bf16 inputs, then cast back to the output dtype before the multiply, matching HuggingFace/Liger/Unsloth behavior.
- The operation is elementwise and has no reductions, so there is no accumulation-order nondeterminism.
- Large positive or negative gates saturate sigmoid. This is expected; the fp32 sigmoid path reduces avoidable precision loss.

Mathematical assumptions:

- `gate` and `up` have identical shape.
- The last dimension is the feature/intermediate dimension.
- Output shape matches input shape.
- Biases and the surrounding linear projections are outside the standalone activation kernel.

Open questions:

- Whether Forge P1 should support `gate_multiplier` and `down_multiplier` in the first implementation or keep the API minimal.
- Whether the first POC should support only separate `gate` and `up` tensors, or also a packed tensor shaped `[..., 2 * hidden]`.

## Supported Surface

Initial proposed support:

- Shapes: `gate` and `up` as matching tensors with shape `(..., hidden)`, typically `(batch, seq_len, intermediate_size)` or flattened `(tokens, intermediate_size)`.
- Layouts: contiguous inputs for the first kernel; wrapper can call `.contiguous()` when needed.
- Dtypes: `float32`, `float16`, `bfloat16`; fp64 only for reference/gradcheck if Triton path supports it poorly.
- Devices: CUDA GPUs via Triton.
- Hidden size: any positive `hidden` up to the selected per-row block limit; use masking for odd sizes.

Explicitly unsupported in the first pass:

- CPU Triton execution.
- Quantized integer inputs.
- Sparse inputs.
- In-place mutation of user inputs.
- Full MLP matmul fusion.
- Distributed DTensor support unless explicitly requested.

## PyTorch Reference

Standalone correctness baseline:

```python
import torch


def swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    gate_fp32 = gate.to(torch.float32) * float(gate_multiplier)
    activated = torch.nn.functional.silu(gate_fp32).to(up.dtype)
    return activated * up * float(down_multiplier)
```

Reference behavior:

- PyTorch autograd provides the backward baseline.
- For fp32, target strict parity within `rtol=1e-5`, `atol=1e-5` for ordinary random inputs.
- For fp16/bf16, target `rtol=1e-2`, `atol=1e-2` initially, then tighten based on real device data.
- Gradcheck should use smaller shapes and fp64 or fp32 reference where practical.

## Forge Existing State

Current local Forge files:

- `kernel-POCs/docs/swiglu.md`: existed but was empty before this report.
- `kernel-POCs/tests/test_swiglu.py`: existed but was empty.
- All current top-level kernel test files and `benchmarks/bench_all.py` are empty, so SWIGLU will likely establish the first concrete test/benchmark pattern in this repo.
- `kernel-POCs/kernels/swiglu/`: not present yet.
- `kernel-POCs/README.md`: lists SwiGLU as P1 and expects `kernels/swiglu/` plus `tests/test_swiglu.py`.
- `kernel-POCs` currently has no `pyproject.toml`, `setup.py`, or package `__init__.py` files, despite the README's `pip install -e .` instruction.

Closest implementation reference inside Forge:

- No standalone activation kernel exists yet.
- `kernel-POCs/kernels/lora_mlp/docs/artifacts/liger_kernel/swiglu_ops.py` and `kernel-POCs/kernels/lora_mlp/docs/artifacts/unsloth/swiglu_triton.py` are useful local reference artifacts.

Repo-structure implication for Stage 2:

- The SWIGLU implementation can either use local imports from tests/benchmarks, or Stage 2 can add minimal package scaffolding so `from kernels.swiglu.swiglu import swiglu` works cleanly.
- Adding scaffolding is a slightly broader repo change, but it may be the right time because this is the first non-empty kernel implementation.

Environment note:

- Current shell Python does not have `torch` installed.
- `nvidia-smi` cannot communicate with an NVIDIA driver in this environment.
- Stage 1 does not require a GPU, but Stage 3 correctness/benchmark validation will need a CUDA environment with PyTorch and Triton installed.

## Liger Comparison

Source paths:

- Local: `Liger-Kernel/src/liger_kernel/ops/swiglu.py`
- Local wrapper: `Liger-Kernel/src/liger_kernel/transformers/swiglu.py`
- Local tests: `Liger-Kernel/test/transformers/test_swiglu.py`
- Local benchmark: `Liger-Kernel/benchmark/scripts/benchmark_swiglu.py`
- Local checkout: clean, commit `547cf4c`
- Current web source checked: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py

API:

- Low-level autograd function: `LigerSiLUMulFunction.apply(a, b, gate_multiplier=1.0, down_multiplier=1.0)`.
- MLP wrappers replace the activation point in LLaMA, Phi3, Qwen3 MoE, Hunyuan, Falcon H1, and related model MLPs.

Fusion boundary:

- Fuses only `silu(gate) * up`.
- Does not fuse `gate_proj`, `up_proj`, or `down_proj` matmuls in the normal SwiGLU path.

Optimization strategy:

- One Triton program per row after reshaping to `(-1, hidden)`.
- Block size is `next_power_of_2(hidden)` with a max fused size of 65536.
- Computes sigmoid/SwiGLU in fp32.
- Recomputes sigmoid/SwiGLU in backward instead of saving activated output.
- Saves `gate` and `up`; returns gradients by writing into the saved `gate` and `up` buffers inside the backward helper.
- Forces contiguous inputs through an `ensure_contiguous` wrapper.

Optimized axes:

- Hidden/intermediate dimension is the main vectorized axis.
- Token rows are independent programs.
- Benchmarks sweep sequence length and model hidden/intermediate sizes.

Input settings allowed:

- Tensor shapes with equal `a.shape == b.shape`, any rank as long as the last dimension is hidden.
- fp32 and bf16 tested locally in Liger; fp16 appears structurally supported.
- Non-contiguous inputs are made contiguous by wrapper.
- Current upstream includes multiplier support and DTensor handling.

Unsupported or unclear:

- Hidden sizes above Liger's `MAX_FUSED_SIZE = 65536` fail.
- DTensor semantics assume local shards can be operated independently; Forge likely does not need this for P1.
- Liger tests show very loose bf16 tolerances for some full MLP cases, so Forge should measure and set tighter activation-level tolerances if possible.

Pros:

- Simple and robust activation-level fusion.
- Good reference for autograd wrapper and backward math.
- Broad model patching coverage.

Cons / gaps:

- Per-row launch layout may underutilize very small hidden dimensions or tiny token counts.
- No packed `[..., 2 * hidden]` path in the standalone op.
- Full matmul epilogue fusion is left for future work.
- Multipliers are now included upstream, but add API surface that Forge may not need immediately.

## Unsloth Comparison

Source paths:

- Local: `unsloth/unsloth/kernels/swiglu.py`
- Local usage: `unsloth/unsloth/kernels/fast_lora.py`
- Local inference path: `unsloth/unsloth/models/llama.py`
- Local checkout: clean, commit `382683eb`
- Current web source checked: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py

API:

- `swiglu_fg_kernel(e, g)` for forward `h = silu(e) * g`.
- `swiglu_DWf_DW_dfg_kernel(DW, e, g)` for LoRA MLP backward/intermediate reuse.

Fusion boundary:

- Pointwise activation fusion for the training kernel.
- In the LoRA MLP path, the activation kernel is part of a larger hand-written autograd flow, not a standalone user-facing autograd function.
- In inference, Unsloth uses fast linear calls and in-place PyTorch SiLU/multiply rather than this exact training autograd wrapper.

Optimization strategy:

- Flat 1D grid over all elements with fixed `BLOCK_SIZE = 1024`.
- Supports long indexing beyond int32-safe element counts.
- Computes gate sigmoid/SwiGLU in fp32.
- Backward mutates buffers in-place: `DW <- h`, `e <- df`, `g <- de`, avoiding extra intermediate allocations in the LoRA pipeline.

Optimized axes:

- Total element count, not row structure.
- Memory reuse in the surrounding LoRA pipeline.
- Large tensors through int64 indexing fallback.

Input settings allowed:

- Forward wrapper expects 3D `(batch, seq_len, hidden)`.
- Backward wrapper expects flattened 2D `(batch_seq_len, hidden)`.
- Contiguous dense tensors are implied.

Unsupported or unclear:

- Not a standalone `torch.autograd.Function`.
- In-place mutation is tightly coupled to Unsloth's LoRA MLP backward.
- No explicit multiplier support in the standalone SwiGLU kernel found locally.

Pros:

- Good reference for flat elementwise kernel and long indexing.
- Good memory-reuse idea for future fused LoRA/MLP work.

Cons / gaps:

- API is less reusable for Forge's modular standalone kernel.
- Fixed block size may be less tuned for row-local hidden sizes.
- In-place mutation increases coupling and correctness risk for a standalone public kernel.

## First-Principles Analysis

Tier classification:

- Tier 1 elementwise: each output element depends only on matching `gate`, `up`, and optional scalar multipliers. There are no reductions or cross-element dependencies.

Unavoidable math per element:

- One sigmoid of the gate value.
- A few multiplies/adds for SiLU and backward derivative.

Unavoidable memory traffic for standalone forward:

- Read `gate`.
- Read `up`.
- Write `out`.

Unavoidable memory traffic for standalone backward:

- Read `dout`.
- Read saved `gate`.
- Read saved `up`.
- Write `dgate`.
- Write `dup`.

Avoidable intermediate tensors:

- Avoid materializing `sigmoid(gate)`.
- Avoid materializing `silu(gate)`.
- Avoid saving forward output solely for backward.

Recompute vs save:

- Recompute `sigmoid(gate)` and `silu(gate)` in backward. This costs one sigmoid but avoids saving another activation-sized tensor.
- Save `gate` and `up`, because both are required for exact gradients.

Fusion opportunities:

- Immediate first POC: fuse only `silu(gate) * up`.
- Near-term: support packed gate/up input from a combined projection output.
- Later: fuse with gate/up projection epilogue, or dual-GEMM activation, once standalone correctness and benchmark baselines are stable.
- LoRA-specific fusion belongs to the LoRA MLP kernel, not this standalone P1 unless the user chooses otherwise.

Where fusion should stop for P1:

- Stop before matmuls. Matmul epilogue fusion changes scheduling, weight layout, module patching, and backward weight-gradient contracts. It is a different Tier 3 kernel, while standalone SwiGLU is a narrow Tier 1 activation with fast validation.

Compatibility targets worth supporting:

- Qwen/LLaMA-style dense MLP with separate `gate_proj`, `up_proj`, `down_proj`.
- Phi-style packed `gate_up_proj` can be supported by chunking externally first, then a packed Triton path later.
- Qwen3-8B reference shape from Forge context: hidden size 4096, intermediate size 11008, typical training shape batch 4, seq 2048, bf16.

Shape regimes competitors may miss:

- Very small token counts where one row per program can have launch/occupancy overhead.
- Odd hidden sizes, covered by masks but worth testing.
- Very large total element counts where flat long indexing is safer than int32 offsets.
- Packed `[..., 2 * hidden]` input, which can reduce Python-side chunk overhead and improve locality for packed-projection models.

Numerical stability risks:

- fp16/bf16 sigmoid precision if not upcast to fp32.
- Matching PyTorch reference exactly requires the same cast point: compute SiLU in fp32, cast activated value to input/output dtype, then multiply by `up`.

Backward correctness risks:

- Missing the extra `gate_multiplier` factor in `dgate`.
- Applying `down_multiplier` in forward but not scaling upstream gradient in backward.
- Accidentally overwriting saved inputs before autograd finishes.

## Memory And Roofline Model

Qwen3-8B reference training shape from Forge context:

- Batch: 4
- Sequence length: 2048
- Token rows: 8192
- Intermediate/SwiGLU hidden: 11008
- Elements per activation tensor: 90,177,536

Minimum activation-kernel memory traffic, excluding surrounding matmuls:

| Shape / dtype | One tensor | Forward lower traffic | Backward lower traffic |
| --- | ---: | ---: | ---: |
| Qwen3 typical bf16/fp16 | 172 MiB | 516 MiB | 860 MiB |
| Qwen3 typical fp32 | 344 MiB | 1032 MiB | 1720 MiB |
| Large hidden bf16, rows=8192, hidden=65536 | 1024 MiB | 3072 MiB | 5120 MiB |

Lower-bound latency from memory traffic only:

| Shape / dtype | Forward at 2 TB/s | Forward at 3 TB/s | Backward at 2 TB/s | Backward at 3 TB/s |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 typical bf16/fp16 | 0.27 ms | 0.18 ms | 0.45 ms | 0.30 ms |
| Qwen3 typical fp32 | 0.54 ms | 0.36 ms | 0.90 ms | 0.60 ms |

Interpretation:

- This kernel is primarily memory-bandwidth and launch-overhead bound, with sigmoid adding non-trivial arithmetic latency.
- A PyTorch expression can materialize `silu(gate)` and possibly cast intermediates, so practical PyTorch traffic is higher than the standalone lower bound.
- Speedup should be most visible in full forward/backward memory and launch count, not raw FLOP throughput.
- Very small tensors cannot approach bandwidth limits; launch overhead dominates.
- Full MLP runtime may still be dominated by GEMMs, so activation speedup must be reported separately and inside an MLP benchmark.

## Implementation Alternatives

Alternative A: row-wise separate-tensor kernel.

- Input: `gate` and `up`, both shaped `(..., hidden)`.
- Grid: one Triton program per flattened row.
- Block: `next_power_of_2(hidden)`, masked.
- Pros: close to Liger; simple strides; excellent for standard LLM intermediate sizes; natural autograd wrapper.
- Cons: hidden-size block cap; less ideal for tiny hidden sizes or tiny row counts.
- Recommendation: first implementation.

Alternative B: flat separate-tensor kernel.

- Input: `gate` and `up`, contiguous flattened element stream.
- Grid: one Triton program per fixed block of elements, e.g. 1024.
- Pros: close to Unsloth; simple long-indexing path; good for arbitrary total element counts.
- Cons: less shape-specialized; may not exploit row-local hidden-size structure; backward still needs two output gradients.
- Recommendation: benchmark variant only after row-wise baseline.

Alternative C: packed `gate_up` kernel.

- Input: packed tensor where one row stores two halves, usually `[..., 2 * hidden]`.
- Output: `[..., hidden]`.
- Pros: helps Phi-style or combined-projection layouts; can avoid Python `chunk` plus possible views/copies.
- Cons: more API and stride complexity; does not help standard separate `gate_proj`/`up_proj` layouts.
- Recommendation: defer until baseline passes correctness and shows room for improvement.

Alternative D: matmul epilogue / dual-GEMM fusion.

- Input: original hidden states and projection weights.
- Computes gate/up projections and activation in a fused matmul path.
- Pros: theoretically best memory behavior because it avoids writing both projected activations before activation.
- Cons: changes the problem from Tier 1 activation to Tier 3 fused MLP; much harder backward, weight-gradient, LoRA, quantization, and patching surface.
- Recommendation: explicitly out of scope for first SWIGLU POC.

## API Decision Matrix

| Decision | Minimal option | Broader option | Recommendation |
| --- | --- | --- | --- |
| Public op | `swiglu(gate, up)` | `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0)` | Broader, because multipliers are scalar and cheap. |
| Input layout | Separate tensors only | Separate plus packed | Separate first; packed later. |
| Contiguity | Require contiguous and raise | Internally call `.contiguous()` | Internally call `.contiguous()` for parity with Liger, but benchmark copy overhead separately. |
| Backward storage | Save `gate`, `up`, maybe `out` | Save `gate`, `up`; recompute SiLU | Recompute. |
| Gradient writes | Allocate fresh `dgate`, `dup` | Mutate saved buffers | Allocate fresh for public safety. |
| Package layout | Test-local imports | Add minimal package scaffolding | Prefer scaffolding if user is okay with repo-level setup. |

## Test Matrix Detail

Standalone operation tests:

- Forward parity for fp32, fp16, bf16 across standard and odd shapes.
- Backward parity for both input gradients.
- Multiplier parity with `(0.7, 1.3)`, `(1.5, 0.5)`, and `(1.0, 1.0)`.
- Non-contiguous input behavior, if wrapper promises it.
- Shape mismatch error behavior.
- Hidden size around powers of two: 1, 2, 7, 8, 9, 1023, 1024, 1025, 11008, 11009.

MLP-level smoke tests:

- Dense LLaMA/Qwen-style: `down_proj(swiglu(gate_proj(x), up_proj(x)))`.
- Packed Phi-style reference can be tested with `gate, up = gate_up.chunk(2, dim=-1)` without implementing a packed kernel yet.

Gradcheck:

- Use tiny fp64 inputs if Triton accepts fp64 path cleanly.
- If fp64 Triton support is poor, use fp32 analytical gradient comparison against PyTorch autograd and record the limitation.

Benchmark detail:

- Activation-only forward, backward, and forward+backward.
- MLP-level wall time with projected `gate/up` already available and with full three-linear MLP.
- Peak memory with `torch.cuda.max_memory_allocated`.
- Effective bandwidth estimate from the minimum-traffic model above.

## External Research

Sources checked:

- Shazeer, "GLU Variants Improve Transformer": https://arxiv.org/abs/2002.05202
- HuggingFace LLaMA MLP source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- Triton `tl.sigmoid` documentation: https://triton-lang.org/main/python-api/generated/triton.language.sigmoid.html
- Liger SWIGLU source: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
- Unsloth SWIGLU source: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/swiglu.py
- Liger issue #936, multiplier support for Falcon H1: https://github.com/linkedin/Liger-Kernel/issues/936
- Liger issue #937, matmul/epilogue fusion alternatives: https://github.com/linkedin/Liger-Kernel/issues/937
- Triton issue #8232, TMA vs naive activation SWIGLU performance discussion: https://github.com/triton-lang/triton/issues/8232

Useful findings:

- SwiGLU is a GLU-family FFN replacement; the kernel target is the componentwise activation/product, not the full FFN by default.
- HuggingFace LLaMA still expresses the MLP as `down_proj(act(gate_proj(x)) * up_proj(x))`, confirming the standalone activation boundary is a natural patch point.
- Liger community discussion has already identified multiplier support and matmul epilogue fusion as separate extensions.
- The Triton TMA issue suggests naive row/block elementwise activation kernels can already approach high memory bandwidth on H100 for some shapes, so TMA/persistent variants should not be assumed better for the first POC.

Rejected ideas for first implementation:

- Full gate/up/down matmul fusion: too broad for P1 and hard to validate without a stable activation baseline.
- In-place mutation like Unsloth backward: valuable for a coupled LoRA path, risky for a standalone autograd API.
- TMA-based activation kernel: not justified for a first Tier 1 POC; measure naive first.

Missing research:

- Local competitor checkouts were not `git pull` refreshed during this stage. Current GitHub web sources and issues were checked, but the local repositories were left untouched.

## Improvement Hypotheses

Idea 1: Standalone row-wise activation kernel with fp32 SiLU and recompute backward.

- What it optimizes: eliminates intermediate `silu(gate)` allocation and PyTorch elementwise kernel launches.
- Why it should matter: forward and backward are memory-bound; reducing reads/writes and launches should improve speed and memory.
- Benefiting shapes/dtypes: bf16/fp16 LLM training shapes, especially `(tokens, intermediate)` with intermediate around 11008.
- Risks: row-wise block limit and poor occupancy for tiny tensors.
- Measurement plan: compare PyTorch baseline against Triton forward, backward, and full fwd+bwd for Qwen3-like and odd-size shapes.
- Fallback: use flat 1D grid if row-wise program layout performs poorly.

Idea 2: Include optional scalar multipliers in the first API.

- What it optimizes: avoids a separate multiply and supports Falcon H1-style MLPs.
- Why it should matter: almost free in the existing math path and aligns with current Liger behavior.
- Benefiting shapes/dtypes: all shapes when multipliers are used; no expected penalty for default `1.0`.
- Risks: slightly wider API and more tests.
- Measurement plan: correctness tests for non-1 multipliers; benchmark default and multiplier paths.
- Fallback: keep public wrapper default-only and leave multiplier args private/internal.

Idea 3: Add a packed input path after the separate-tensor kernel.

- What it optimizes: supports outputs of combined `gate_up_proj` shaped `[..., 2 * hidden]` without Python chunk materialization.
- Why it should matter: some model families use packed gate/up projections.
- Benefiting shapes/dtypes: Phi-style or packed MLP layouts.
- Risks: stride handling and API complexity.
- Measurement plan: compare chunk-plus-separate-kernel vs packed-kernel on packed contiguous tensors.
- Fallback: implement only separate tensor path first.

Idea 4: Flat 1D long-indexing kernel variant.

- What it optimizes: robust huge tensors and possibly better scheduling for small hidden dimensions.
- Why it should matter: Unsloth uses flat total-element indexing with int64 fallback.
- Benefiting shapes/dtypes: very large `numel`, small or irregular hidden sizes.
- Risks: less row-local shape specialization; may underperform for common LLM hidden sizes.
- Measurement plan: benchmark row-wise vs flat across hidden sizes and token counts.
- Fallback: keep row-wise only unless data shows a win.

## Selected Design Candidate

Recommended Stage 2 implementation:

- Create `kernel-POCs/kernels/swiglu/swiglu.py`.
- Provide `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0)` backed by `torch.autograd.Function`.
- Implement separate forward and backward Triton kernels.
- Use row-wise programs over flattened `(-1, hidden)` tensors.
- Compute gate sigmoid/SiLU in fp32 and cast activated value to input dtype before multiplying.
- Save only `gate` and `up`; recompute activation in backward.
- Require contiguous inputs internally by calling `.contiguous()` in the wrapper.
- Do not mutate user-visible inputs.
- Add `kernel-POCs/tests/test_swiglu.py` correctness tests.
- Add `kernel-POCs/kernels/swiglu/benchmarks/benchmark_swiglu.py` local benchmark.

Why this boundary:

- It matches the natural HF/Liger patch point and the existing Forge README P1 scope.
- It produces a small, auditable Triton kernel before broader MLP/LoRA fusion.
- It gives a reliable baseline for later packed or matmul-epilogue variants.

## Correctness Plan

Forward parity:

- Shapes: `(1, 1, 8)`, `(2, 8, 8)`, `(4, 16, 11008)`, `(2, 7, 41)`, flattened `(8192, 11008)` if memory allows.
- Dtypes: fp32, fp16, bf16 when CUDA supports them.
- Multipliers: default and non-default scalar combinations.

Backward parity:

- Compare `gate.grad` and `up.grad` against PyTorch reference.
- Test non-contiguous inputs if wrapper promises `.contiguous()` behavior.

Gradcheck:

- Use tiny fp64/fp32 shapes if Triton path supports the dtype sufficiently; otherwise run analytical gradient comparison against PyTorch autograd.

Stress cases:

- Odd hidden sizes.
- Hidden size one below/above power of two.
- Large hidden sizes near block limit.
- Large total element count if memory is available.

## Benchmark Plan

Baselines:

- PyTorch reference: `torch.nn.functional.silu(gate.float()).to(dtype) * up`.
- Liger local kernel, if dependencies are installed and importable.
- Forge Triton kernel.

Shape strata:

- Qwen3-like: batch 4, seq 2048, intermediate 11008, bf16.
- Small: batch 1, seq 1-128, hidden 64-1024.
- Odd: hidden 41, 4097, 11009.
- Large: hidden 32768 and 65536 if memory allows.

Dtypes:

- fp32, fp16, bf16.

Metrics:

- Forward latency.
- Backward latency.
- Full forward+backward latency.
- Peak memory.
- Effective bandwidth estimate.

Device:

- NVIDIA A100/H100 preferred per Forge context. Current local shell cannot validate GPU behavior.

## Known Failure Boundaries

- Current environment lacks installed `torch` and visible NVIDIA driver, so live tests/benchmarks cannot run here yet.
- Hidden dimension above selected max fused block size will need a multi-block design.
- Non-contiguous inputs add copy overhead if handled by `.contiguous()`.
- Very small tensors may not show speedup due to launch overhead.
- Full MLP fusion and packed gate/up handling are intentionally outside the first standalone POC unless approved.

## Human Checkpoint Questions

1. Should Stage 2 include `gate_multiplier` and `down_multiplier` in the first public API? My recommendation: yes, because the implementation cost is low and it tracks current Liger behavior.
2. Should I create/switch to branch `day1/p1-swiglu-kernel` before code generation? Current branch is `main`.
3. Should the first implementation be separate-tensor only (`gate`, `up`), with packed `[..., 2 * hidden]` deferred? My recommendation: yes, keep packed support as a follow-up benchmark variant.
4. Do you want me to request approval to refresh `Liger-Kernel/` and `unsloth/` with `git pull --ff-only` before Stage 2, or continue from the current local checkouts plus web-checked source?

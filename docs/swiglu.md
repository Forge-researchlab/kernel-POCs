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

- `sigmoid` and `silu` should be computed in fp32 for fp16/bf16 inputs. This matches Liger and Unsloth, but intentionally diverges from HuggingFace eager LLaMA MLP, which applies `ACT2FN[hidden_act](gate_proj(x)) * up_proj(x)` directly in the activation dtype.
- Because the fp32-SiLU path is more accurate than the HF eager expression, HF parity tolerances must allow dtype-order differences; Liger/Unsloth parity should use the same cast boundary as their kernels.
- The operation is elementwise and has no reductions, so there is no accumulation-order nondeterminism.
- Large positive or negative gates saturate sigmoid. This is expected; the fp32 sigmoid path reduces avoidable precision loss.

Mathematical assumptions:

- `gate` and `up` have identical shape.
- The last dimension is the feature/intermediate dimension.
- Output shape matches input shape.
- Biases and the surrounding linear projections are outside the standalone activation kernel.
- `gate_multiplier` and `down_multiplier`, if supported, are scalar Python floats only. Per-row, per-channel, or per-element scale tensors are out of scope for P1.

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
- Fast backward mode does not guarantee input preservation; use `preserve_inputs=True` for that behavior.
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
    # Pin the default parity order: first multiply activated gate by up,
    # then apply the scalar down multiplier in the output dtype.
    return (activated * up) * float(down_multiplier)
```

Reference behavior:

- PyTorch autograd provides the backward baseline.
- The default reference pins the parity order as `(silu_fp32_cast_to_dtype * up) * down_multiplier`.
- A stricter Forge accuracy variant may instead compute `(silu_fp32 * up.float()) * down_multiplier` before the final cast, but that should be benchmarked and tested separately because it diverges from HF/Liger/Unsloth dtype order.
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
- `num_warps` follows Liger's size policy unless benchmarking proves another policy: 4 warps below block 2048, 8 warps for block >= 2048, 16 warps for block >= 8192, and 32 warps for block >= 32768 on CUDA.
- Computes sigmoid/SwiGLU in fp32.
- Recomputes sigmoid/SwiGLU in backward instead of saving activated output.
- Saves `gate` and `up`; returns gradients by writing into the saved `gate` and `up` buffers inside the backward helper.
- Forces contiguous inputs through an `ensure_contiguous` wrapper.
- Applies `down_multiplier` outside the Triton kernels when it is not 1.0, causing one extra output-sized elementwise pass in forward and one in backward.

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
- Non-default `down_multiplier` is not fused into Liger's Triton kernels.
- Backward mutates saved input buffers. This reduces gradient allocation pressure but can corrupt any other reference to contiguous `gate`/`up` tensors and makes higher-order autograd unsafe/unsupported.

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
- Its gate derivative uses `se * (1 + e * (1 - se))`, which is algebraically identical to `sig + silu_e * (1 - sig)`.

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

- Recompute `sigmoid(gate)` and `silu(gate)` in backward. This costs one sigmoid pass but avoids saving a sigmoid/activation-sized auxiliary tensor.
- Save `gate` and `up`, because both are required for exact gradients.
- Saving `out` is not useful for the default backward: `out = silu(gate) * up * down_multiplier` does not recover `sigmoid(gate)` without unstable division and does not avoid the backward sigmoid.

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
- In-place gradient writes can corrupt user-visible contiguous `gate`/`up` tensors because a contiguity wrapper may return the original tensor storage.
- In-place mutation of saved tensors breaks or invalidates higher-order autograd. Avoiding mutation preserves saved tensors, but true gradgrad support still requires explicit differentiable backward or second-order formulas.

Special-value semantics:

- NaN and Inf propagation should be tested directly against the chosen PyTorch reference.
- `sigmoid(+inf) = 1` and `sigmoid(-inf) = 0`, but the naive `x * sigmoid(x)` SiLU formula can still produce implementation-specific edge behavior for `-inf * 0`; tests should lock Forge behavior to the selected reference.
- Multipliers much smaller than 1 can lose mantissa bits or underflow in fp16/bf16 if applied after casting to output dtype. A fp32-product variant could improve this, but it is a deliberate accuracy divergence from the default competitor-parity path.

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
- At the Qwen3 bf16 shape, one activation-sized tensor is 172 MiB. A safety mode that allocates fresh `dgate` and `dup` instead of reusing saved buffers can add up to 344 MiB peak memory versus Liger's in-place backward. This should be opt-in for edge cases, not forced on the common training path.
- For non-default `down_multiplier`, Liger's out-of-kernel scalar multiply adds a full read+write pass in forward and backward. At Qwen3 bf16 shape, that is roughly 344 MiB extra memory traffic per pass, or 688 MiB across forward and backward, before allocator effects.

## Implementation Alternatives

Alternative A: row-wise separate-tensor kernel.

- Input: `gate` and `up`, both shaped `(..., hidden)`.
- Grid: one Triton program per flattened row.
- Block: `next_power_of_2(hidden)`, masked.
- Warps: start with Liger's block-size policy, then autotune/benchmark only if measured shapes show occupancy or latency issues.
- Pros: close to Liger; simple strides; excellent for standard LLM intermediate sizes; natural autograd wrapper.
- Cons: hidden-size block cap; less ideal for tiny hidden sizes or tiny row counts.
- Recommendation: first implementation.

Alternative B: flat separate-tensor kernel.

- Input: `gate` and `up`, contiguous flattened element stream.
- Grid: one Triton program per fixed block of elements, e.g. 1024.
- Pros: close to Unsloth; simple long-indexing path; good for arbitrary total element counts; can remove the row-wise 65536 hidden cap for contiguous tensors.
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
| Backward storage | Save `gate`, `up`; recompute SiLU | Save sigmoid/activation auxiliary | Recompute. Saving `out` is not useful. |
| Gradient writes | Mutate/reuse saved buffers | Allocate fresh `dgate`, `dup` | Default to fast buffer reuse; add a safety flag for callers that need preserved inputs. |
| Scalar multiplier placement | Apply `down_multiplier` outside the kernel | Fuse `down_multiplier` into forward/backward kernels | Fuse it. This is a concrete Liger gap for non-default multipliers. |
| Product precision | Competitor-parity dtype order | Optional fp32 product through final cast | Start with parity; measure fp32-product as an accuracy variant if requested. |
| Hidden > 65536 | Inherit row-wise cap | Flat or multi-block row variant | Stage 2A inherits cap; Stage 2B flat variant can remove it for contiguous tensors. |
| Package layout | Test-local imports | Add minimal package scaffolding | Prefer scaffolding if user is okay with repo-level setup. |

## Competitive Bar Before Implementation

The first implementation must not be justified as "we can write the same thing locally." Liger and Unsloth are the floor.

What Liger already does well:

- Standalone `torch.autograd.Function` for `silu(gate) * up`.
- Row-wise Triton programs over the hidden dimension.
- fp32 sigmoid/SiLU and recompute in backward.
- Optional `gate_multiplier` and `down_multiplier`.
- Contiguity handling and broad HuggingFace monkey-patch coverage.
- DTensor support in current upstream.

What Unsloth already does well:

- Flat elementwise kernels with `LONG_INDEXING` specialization for very large tensors.
- In-place backward buffer reuse inside the LoRA MLP path.
- Tight coupling with fused LoRA MLP custom autograd, where the activation kernel is only one piece of a larger memory-saving design.

What Axolotl-style Unsloth derivatives already do:

- Use activation Triton kernels together with higher-level LoRA MLP custom autograd.
- Recompute activation output and gradients during backward so the surrounding autograd function can avoid extra saved tensors.

Therefore a Forge SWIGLU POC is only worthwhile if it provides at least one of these advantages:

1. **Hybrid coverage advantage:** expose a standalone, safe autograd API that combines Liger's multiplier-compatible public function with Unsloth's long-indexing/flat-kernel robustness where that actually benchmarks better.
2. **Shape-specialized performance advantage:** benchmark both row-wise and flat kernels, then choose by shape regime rather than hard-coding one competitor's strategy.
3. **Packed-layout advantage:** support a packed `gate_up` path for combined projection layouts. Liger's standalone op does not provide this path, while NVIDIA NeMo's combined gate/up projection docs identify combined projection as a launch/memory-efficiency tool.
4. **MLP/LoRA integration advantage:** use the standalone kernel as the baseline, but treat a fused LoRA MLP or packed MLP path as the real route to beating Unsloth. Unsloth's advantage is the whole custom-autograd MLP, not just the activation microkernel.
5. **Evidence advantage:** publish a benchmark matrix and roofline accounting that makes clear when Forge wins, ties, or loses. If the result only ties Liger and is less integrated than Unsloth, the checkpoint should say so and not oversell it.

Forge-vs-Liger measurable advantage candidates:

| Candidate | Why it can beat/broaden Liger | Cost / risk | Measurement |
| --- | --- | --- | --- |
| Fuse `down_multiplier` into forward and backward Triton kernels | Avoids Liger's extra full-tensor multiply pass when `down_multiplier != 1.0`; about 344 MiB saved traffic per pass at Qwen3 bf16 shape | Only affects non-default multipliers; must preserve chosen dtype order | Benchmark multiplier and no-multiplier paths separately |
| Configurable backward storage | Fast mode can match competitor memory behavior; safety mode avoids user-visible activation corruption and keeps saved tensors valid for surrounding autograd checks | Safety mode can add up to +344 MiB peak memory at Qwen3 bf16 shape; not by itself gradgrad support | Benchmark fast vs safety mode; test mutation semantics explicitly |
| Optional fp32 product-through-multiplier variant | Can reduce fp16/bf16 underflow or mantissa loss for small `down_multiplier` | Diverges from HF/Liger/Unsloth parity and may affect convergence | Separate accuracy/convergence and bit-diff tests |
| Packed `gate_up` path | Targets combined-projection layouts not covered by Liger's standalone op | More API/stride complexity | Compare packed kernel vs chunk + separate kernel |
| Flat long-indexing variant | Can remove row-wise hidden cap and match Unsloth's large-tensor robustness | May lose row-wise performance on common hidden sizes | Shape selector benchmarks across tiny, typical, huge, odd shapes |

Minimum accept criteria for Stage 2 design:

- Implementing only a Liger-like row-wise op is acceptable as a **baseline implementation**, not as the final claim.
- The code plan must include either a second variant (`flat` or `packed`) or a documented benchmark gate that decides whether the row-wise baseline is worth keeping.
- The benchmark must compare against at least PyTorch and the closest local competitor implementation available in the environment. If Liger/Unsloth cannot be imported, the report must state that the comparison is blocked.
- The final report should classify each result as one of: better than Liger/Unsloth, parity with lower complexity, parity but not differentiated, or worse.

Revised recommendation:

- Stage 2A: implement a clean row-wise baseline only to establish correctness, tests, package structure, and benchmark harness.
- Stage 2B: before calling SWIGLU "done", implement at least one differentiating path:
  - preferred: packed `gate_up` forward/backward path for combined projection outputs;
  - fallback: flat long-indexing variant selected for small/huge shapes if it beats row-wise;
  - later/stronger: LoRA MLP integration where activation backward recomputes `h`, `dup`, and `dgate` for the surrounding projection gradients.
- Stage 2 output should not claim superiority unless benchmarks show it.

Research implications:

- Liger's paper reports that SwiGLU/GeGLU mostly match baseline speed while reducing peak memory around 1.6x at long sequence length. That suggests standalone activation has limited headroom if Liger is already near bandwidth limits.
- Triton issue #8232 reports a naive activation-style SWIGLU kernel reaching near H100 bandwidth and outperforming a TMA attempt for many sizes. That argues against spending first effort on TMA/persistent complexity.
- Recent external claims of much larger SwiGLU gains appear to involve broader fusion and memory-traffic elimination, not simply rewriting the same pointwise activation.

## Test Matrix Detail

Standalone operation tests:

- Forward parity for fp32, fp16, bf16 across standard and odd shapes.
- Backward parity for both input gradients.
- Multiplier parity with `(0.7, 1.3)`, `(1.5, 0.5)`, and `(1.0, 1.0)`.
- Non-contiguous input behavior, if wrapper promises it.
- Shape mismatch error behavior.
- Hidden size around powers of two: 1, 2, 7, 8, 9, 1023, 1024, 1025, 11008, 11009. The tiny sizes are masking/indexing edge tests, not performance-representative shapes.
- Special values: include `nan`, `+inf`, `-inf`, large finite positives/negatives, zeros, and small non-default multipliers.
- Input mutation: verify `gate` and `up` are unchanged after backward for Forge's safe path.
- Optional higher-order behavior: either explicitly mark gradgrad unsupported or add second-order tests if a differentiable backward is implemented.

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
- Always-safe backward as the only mode: avoids mutation surprises, but pays a large peak-memory cost for rare aliasing/higher-order cases.
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

Implementation status as of 2026-05-24:

- Implemented independent SWIGLU kernel in `kernel-POCs/kernels/swiglu/swiglu.py`.
- Implemented row-wise separate-tensor API: `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- Implemented packed API: `swiglu_packed(gate_up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- Added top-level correctness tests in `kernel-POCs/tests/test_swiglu.py`.
- Added local benchmark in `kernel-POCs/kernels/swiglu/benchmarks/benchmark_swiglu.py`.
- Did not add package scaffolding or touch the shared benchmark folder.
- Local validation was limited to syntax and whitespace checks because this environment has no installed `torch` and no visible NVIDIA driver.
- RunPod A100 validation passed correctness and benchmarked against Liger 0.8.0; see the A100 validation section below.

Recommended Stage 2A baseline implementation:

- Create `kernel-POCs/kernels/swiglu/swiglu.py`.
- Provide `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0)` backed by `torch.autograd.Function`.
- Implement separate forward and backward Triton kernels.
- Use row-wise programs over flattened `(-1, hidden)` tensors.
- Use Liger's initial `BLOCK_SIZE`/`num_warps` policy and record benchmark evidence before changing it.
- Compute gate sigmoid/SiLU in fp32 and cast activated value to input dtype before multiplying for the default competitor-parity path.
- Fuse scalar `down_multiplier` into the forward and backward kernels instead of launching separate full-tensor multiplies.
- Save only `gate` and `up`; recompute activation in backward.
- Require contiguous inputs internally by calling `.contiguous()` in the wrapper.
- Add a backward storage flag, preferably `preserve_inputs: bool = False`.
- Default fast mode may reuse saved `gate`/`up` buffers for `dgate`/`dup`, matching competitor memory behavior for ordinary training.
- `preserve_inputs=True` allocates fresh `dgate` and `dup` and avoids mutating saved/user-visible tensors; report its peak-memory tradeoff.
- Treat hidden sizes above 65536 as unsupported in Stage 2A. The Stage 2B flat variant is the proposed route to remove this cap for contiguous tensors.
- Add `kernel-POCs/tests/test_swiglu.py` correctness tests.
- Add `kernel-POCs/kernels/swiglu/benchmarks/benchmark_swiglu.py` local benchmark.

Why this baseline boundary:

- It matches the natural HF/Liger patch point and the existing Forge README P1 scope.
- It produces a small, auditable correctness target before broader MLP/LoRA fusion.
- It gives a reliable benchmark control for packed, flat, and future fused variants.
- It is not by itself a claim of superiority over Liger.

Recommended Stage 2B differentiating implementation before calling SWIGLU complete:

- Add a packed `swiglu_packed(gate_up, split_dim=-1, ...)` path if we want to target combined-projection models.
- Or add a flat long-indexing variant and shape selector if benchmarks show row-wise is not best for small/huge shapes.
- Or add an fp32-product variant if accuracy/convergence needs justify diverging from competitor parity.
- Or move directly into a LoRA MLP integration if the team wants to beat Unsloth on its strongest axis rather than activation-only microbenchmarks.

Superiority claim rule:

- If only Stage 2A exists, report it as "Forge baseline with Liger-like design."
- If Stage 2B wins on at least one measured shape/dtype/layout regime, report exactly that regime and keep the loss/tie regimes visible.
- If no Stage 2B variant beats Liger/Unsloth, either stop or justify Forge-specific value as package/API/test/benchmark integration rather than performance.

## Correctness Plan

Forward parity:

- Shapes: `(1, 1, 8)`, `(2, 8, 8)`, `(4, 16, 11008)`, `(2, 7, 41)`, flattened `(8192, 11008)` if memory allows.
- Dtypes: fp32, fp16, bf16 when CUDA supports them.
- Multipliers: default and non-default scalar combinations.
- Reference order: default tests use `(silu_fp32_cast_to_dtype * up) * down_multiplier`.
- HF comparison: run separately from Liger/Unsloth comparison because HF eager does not upcast SiLU to fp32.
- Fast backward and `preserve_inputs=True` mode should both match gradients.

Backward parity:

- Compare `gate.grad` and `up.grad` against PyTorch reference.
- Test non-contiguous inputs if wrapper promises `.contiguous()` behavior.
- Test mutation semantics explicitly: fast mode may mutate saved buffers; preserve mode must not.

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
- Unsloth local kernel, if dependencies are installed and importable.
- Forge row-wise baseline.
- Forge differentiating variant: packed or flat, depending on Stage 2B choice.

Shape strata:

- Qwen3-like: batch 4, seq 2048, intermediate 11008, bf16.
- Small: batch 1, seq 1-128, hidden 64-1024.
- Odd: hidden 41, 4097, 11009.
- Large: hidden 32768 and 65536 if memory allows.
- Multiplier-specific: repeat typical and odd shapes with `down_multiplier != 1.0` to measure the Liger extra-pass gap.

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

## A100 Validation Results

Date: 2026-05-24
Device: NVIDIA A100-SXM4-80GB
Benchmark command: `python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite a100 --dtype bf16 --rep 50 --warmup 20 --save results/swiglu_a100_bf16_with_liger.csv`
Additional dtype command: `python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite a100 --dtype fp16 fp32 --multipliers scaled --rep 30 --warmup 10 --save results/swiglu_a100_fp16_fp32_scaled.csv`
Liger: editable `liger_kernel==0.8.0`; benchmark needed a Torch/Liger DTensor compatibility shim because Torch 2.4.1 exposes `DTensor` under `torch.distributed._tensor`.

Correctness:

```text
python -m pytest tests/test_swiglu.py -xvs
75 passed in 4.04s
```

Representative benchmark rows:

| Shape | Multipliers | Provider | Forward ms | Full fwd+bwd ms | Peak MiB |
| --- | --- | --- | ---: | ---: | ---: |
| `2x2048x11008` | `1.0, 1.0` | Liger | 0.213 | 0.498 | 688 |
| `2x2048x11008` | `1.0, 1.0` | Forge fast | 0.223 | 0.592 | 688 |
| `2x2048x11008` | `1.0, 1.0` | Forge safe | 0.216 | 0.492 | 860 |
| `2x2048x11008` | `0.7, 1.3` | Liger | 0.327 | 0.749 | 774 |
| `2x2048x11008` | `0.7, 1.3` | Forge fast | 0.227 | 0.505 | 688 |
| `2x2048x11008` | `0.7, 1.3` | Forge safe | 0.218 | 0.492 | 860 |
| `4x2048x11008` | `1.0, 1.0` | Liger | 0.367 | 0.929 | 1376 |
| `4x2048x11008` | `1.0, 1.0` | Forge fast | 0.380 | 0.917 | 1376 |
| `4x2048x11008` | `1.0, 1.0` | Forge safe | 0.370 | 0.908 | 1720 |
| `4x2048x11008` | `0.7, 1.3` | Liger | 0.588 | 1.352 | 1548 |
| `4x2048x11008` | `0.7, 1.3` | Forge fast | 0.383 | 0.935 | 1376 |
| `4x2048x11008` | `0.7, 1.3` | Forge safe | 0.373 | 0.921 | 1720 |
| `1x32x65536` | `1.0, 1.0` | Liger | 0.087 | 0.227 | 32 |
| `1x32x65536` | `1.0, 1.0` | Forge fast | 0.093 | 0.261 | 32 |
| `1x32x65536` | `0.7, 1.3` | Liger | 0.116 | 0.309 | 36 |
| `1x32x65536` | `0.7, 1.3` | Forge fast | 0.095 | 0.232 | 32 |

Additional scaled-multiplier dtype rows:

| Shape | Dtype | Provider | Forward ms | Full fwd+bwd ms | Peak MiB |
| --- | --- | --- | ---: | ---: | ---: |
| `2x2048x11008` | fp16 | Liger | 0.316 | 0.723 | 774 |
| `2x2048x11008` | fp16 | Forge fast | 0.220 | 0.528 | 688 |
| `4x2048x11008` | fp16 | Liger | 0.580 | 1.339 | 1548 |
| `4x2048x11008` | fp16 | Forge fast | 0.373 | 0.932 | 1376 |
| `2x2048x11008` | fp32 | Liger | 0.578 | 1.313 | 1548 |
| `2x2048x11008` | fp32 | Forge fast | 0.371 | 0.952 | 1376 |
| `4x2048x11008` | fp32 | Liger | 1.078 | 2.553 | 3096 |
| `4x2048x11008` | fp32 | Forge fast | 0.673 | 1.725 | 2752 |

Interpretation:

- Default multipliers: Forge is effectively at Liger parity on the large Qwen-like shapes. Some full-mode rows favor Forge safe/packed and some favor Liger; treat this as parity/noise, not a decisive default-path win.
- Non-default multipliers: Forge wins consistently on the large shapes because `down_multiplier` is fused into the Triton forward/backward kernels. At `4x2048x11008`, Forge fast full is about 1.45x faster than Liger and uses 1376 MiB vs Liger's 1548 MiB.
- The non-default multiplier result holds for bf16, fp16, and fp32 on the large Qwen-like A100 shapes listed here. Smaller and launch-bound shapes remain mixed/noisy, and should not be used as the main performance claim.
- Packed path: correct and useful for combined-projection layouts, but not a consistent speed win over separate tensors. Keep it for integration value rather than claiming universal performance superiority.
- Safe path: preserves saved/user-visible tensors at extra peak memory. It can match fast timing on some large rows, but its purpose is semantics, not memory efficiency.
- PyTorch baseline is much slower and higher memory than Forge/Liger on large shapes because the reference materializes/fuses less.

Competitive conclusion:

- This implementation clears the Stage 3 bar for a standalone SWIGLU P1 kernel.
- It should be described as Liger parity for default multipliers plus a measured Forge win for non-default multipliers.
- Further work should move toward MLP/LoRA integration or a flat long-indexing variant, not another undifferentiated row-wise rewrite.

## Known Failure Boundaries

- Current local development environment lacks installed `torch` and visible NVIDIA driver, so live tests/benchmarks were run on RunPod A100 instead.
- Stage 2A row-wise kernel inherits a 65536 hidden cap. Supporting larger hidden sizes requires either a flat kernel or a multi-block-per-row design.
- Non-contiguous inputs add copy overhead if handled by `.contiguous()`.
- Very small tensors may not show speedup due to launch overhead.
- Activation-only kernels do not substantially beat Liger on default multipliers because both are close to the same bandwidth/launch envelope.
- Forge does beat Liger on non-default `down_multiplier` because it avoids Liger's extra full-tensor scalar multiply passes.
- Full MLP/LoRA fusion is a different, larger kernel surface, but it may be required to beat Unsloth end-to-end.

## Human Checkpoint Decision

Recommendation: mark standalone SWIGLU P1 complete and move to the next integration target.

Resolved decisions:

1. Keep `gate_multiplier` and `down_multiplier` in the public API; A100 data shows this is the measured Forge advantage over Liger for non-default multipliers.
2. Keep `preserve_inputs=False` as the default training path; expose `preserve_inputs=True` for callers that need mutation safety.
3. Keep packed `gate_up` support as layout coverage, not as a universal speed claim.
4. Do not add package scaffolding in this pass, per user scope.

Open next-step choices:

1. Start the next MLP/LoRA integration, where Unsloth's real advantage lives.
2. Add a flat long-indexing selector only if hidden sizes above 65536 or irregular shapes become a real requirement.

# SwiGLU Decision Brief

Stage: Stage 2 implementation handoff  
Decision needed: run correctness and benchmark checks on the A100, then decide whether to tune row-wise/packed paths.

## Recommended Next Action

Pull this code on the A100 RunPod and run the SWIGLU correctness and benchmark commands below.

Implemented path:

1. Row-wise SWIGLU baseline with forward/backward Triton kernels.
2. Fused scalar `gate_multiplier` and `down_multiplier` inside Triton kernels.
3. Fast backward by default, plus `preserve_inputs=True` safety mode.
4. Packed `gate_up` path as the first differentiating variant.
5. Local correctness tests and benchmark script.

## Key Technical Conclusion

SwiGLU itself is a Tier 1 elementwise kernel:

```python
out = silu(gate * gate_multiplier) * up * down_multiplier
```

Backward:

```python
z = gate * gate_multiplier
sig = sigmoid(z)
silu_z = z * sig
dout_scaled = dout * down_multiplier

dup = dout_scaled * silu_z
dgate = dout_scaled * up * (sig + silu_z * (1 - sig)) * gate_multiplier
```

The math is straightforward. The real decision is competitive positioning: Liger and Unsloth already cover the obvious activation fusion, so Forge must either improve a measurable gap or use the baseline only as a control for a stronger variant.

## Competitor Baseline And Forge Advantage

Liger already has:

- standalone `torch.autograd.Function`
- row-wise Triton over hidden dimension
- fp32 sigmoid/SiLU
- recompute backward
- scalar `gate_multiplier` and `down_multiplier`
- broad HF patching

Unsloth already has:

- flat `BLOCK_SIZE=1024` elementwise kernels
- int64 long-indexing fallback
- in-place backward buffer reuse inside LoRA MLP

Forge advantages worth testing:

| Forge idea | Why it matters |
| --- | --- |
| Fuse `down_multiplier` into kernels | Liger applies it outside Triton when non-default, causing extra full-tensor passes. |
| Configurable backward storage | Fast default can match competitor memory behavior; safety flag preserves inputs for edge cases. |
| Packed `gate_up` path | Targets combined-projection layouts not covered by Liger's standalone op. |
| Flat variant/selector | Can match Unsloth robustness for huge/tiny shapes if row-wise is weak. |

Important tradeoff: preserving inputs can add up to two output-sized gradient buffers. At Qwen3 bf16 shape, this is up to ~344 MiB peak memory versus in-place mutation. That cost should be opt-in, not mandatory for the common training path.

## Evidence

Qwen3-8B reference shape:

- batch = 4
- sequence = 2048
- rows = 8192
- intermediate = 11008
- elements = 90,177,536
- one bf16 activation tensor = 172 MiB

Minimum traffic:

- forward: read gate + read up + write output = 516 MiB
- backward: read dout + gate + up, write dgate + dup = 860 MiB

Implication: activation-only SWIGLU is memory-bandwidth/launch bound. A simple row-wise clone of Liger is unlikely to be a major performance win. The strongest near-term measurable Liger gap is fusing non-default `down_multiplier`.

Numerics:

- Forge should compute sigmoid/SiLU in fp32 for fp16/bf16. This matches Liger/Unsloth, but intentionally diverges from HF eager, which does not auto-upcast LLaMA MLP SiLU.
- Default parity order should be fixed as `(silu_fp32_cast_to_dtype * up) * down_multiplier`.

## Stage 2 Plan

Implemented:

- `kernel-POCs/kernels/swiglu/swiglu.py`
- `kernel-POCs/tests/test_swiglu.py`
- `kernel-POCs/kernels/swiglu/benchmarks/benchmark_swiglu.py`

Implementation details:

- `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- `swiglu_packed(gate_up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- Use row-wise Triton programs over flattened `(-1, hidden)`.
- Start with Liger's `BLOCK_SIZE`/`num_warps` policy.
- Compute sigmoid/SiLU in fp32, cast activated value to output dtype for default parity.
- Fuse `down_multiplier` inside forward and backward kernels.
- Save only `gate` and `up`; recompute SiLU in backward.
- Default fast path may reuse saved buffers for `dgate`/`dup`.
- Safe path allocates fresh `dgate`/`dup` and does not mutate saved/user tensors.
- Stage 2A inherits the 65536 hidden-size cap.

Already included differentiator:

- Packed `gate_up` forward/backward path.

Still optional after A100 results:

- Flat long-indexing variant and shape selector if row-wise is weak.
- LoRA MLP integration if the goal is to beat Unsloth end-to-end rather than activation-only.

## Tests And Benchmarks Needed

Correctness:

- forward/backward parity for fp32, fp16, bf16
- odd and power-of-two boundary hidden sizes
- multiplier cases: default and non-default
- non-contiguous input behavior if wrapper promises `.contiguous()`
- NaN, +Inf, -Inf special values
- verify fast mode and preserve-input mode behavior separately
- explicitly mark gradgrad unsupported unless implemented

Benchmarks:

- PyTorch baseline
- Liger local kernel if importable
- Unsloth local kernel if importable
- Forge row-wise baseline
- Forge Stage 2B variant

Must include:

- Qwen3-like bf16 shape
- small launch-overhead shapes
- odd hidden sizes
- large hidden sizes near 65536
- non-default `down_multiplier` shapes to test the Liger extra-pass gap

## Rejected For Next Step

- Full matmul epilogue fusion: too large for first SWIGLU step; it changes the problem into fused MLP.
- TMA/persistent activation kernel: not justified before measuring simple row-wise/flat variants.
- Always-safe backward as the only mode: pays a large peak-memory cost for rare aliasing/higher-order cases.
- Claiming row-wise baseline beats Liger without benchmarks: not acceptable.

## Open Questions

1. Do correctness tests pass on A100?
2. Does fused `down_multiplier` beat Liger for non-default multipliers?
3. Does packed `gate_up` beat chunk + separate SWIGLU?
4. Is row-wise weak enough on any shape to justify a flat long-indexing variant?

## Go / No-Go

Go if:

- Correctness passes for forward/backward across fp32/fp16/bf16.
- `forge_fast` is competitive with PyTorch and Liger on default multipliers.
- `forge_fast` beats or ties Liger on non-default `down_multiplier`.
- Packed path is correct and shows either speed or integration value.

No-go if:

- Correctness fails.
- Forge is slower than Liger across the board and packed path provides no value.
- Fast backward mutation causes practical training issues that safety mode cannot isolate.

## RunPod Commands

From `kernel-POCs/`:

```bash
python -m pytest tests/test_swiglu.py -xvs
python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite smoke --dtype bf16 --rep 20 --warmup 10 --save results/swiglu_smoke.csv
python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite a100 --dtype bf16 --rep 50 --warmup 20 --save results/swiglu_a100_bf16.csv
python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite a100 --dtype fp16 fp32 --multipliers scaled --rep 30 --warmup 10 --save results/swiglu_a100_fp16_fp32_scaled.csv
```

Send back:

- full pytest output
- the three CSV files
- GPU name from `nvidia-smi`
- whether Liger provider was skipped or imported

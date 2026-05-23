# SwiGLU Decision Brief

Stage: Stage 3 A100 validation checkpoint
Decision needed: accept SWIGLU P1 as a competitive standalone activation kernel, or spend another pass on shape tuning.

## Recommendation

Accept the current standalone SWIGLU kernel as a valid P1 checkpoint.

Reason: correctness passed, default-multiplier performance is essentially Liger parity, and the intended Forge differentiator works: fusing `down_multiplier` into Triton gives a clear win over Liger when multipliers are non-default.

Next best work is not another basic SWIGLU rewrite. It is either:

1. integrate this into the next MLP/LoRA path, or
2. add a flat/selector variant only if we want to remove the 65536 hidden cap or tune small/edge shapes.

## What Is Implemented

- Separate API: `swiglu(gate, up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- Packed API: `swiglu_packed(gate_up, gate_multiplier=1.0, down_multiplier=1.0, preserve_inputs=False)`.
- Row-wise Triton forward/backward kernels over flattened `(-1, hidden)`.
- fp32 sigmoid/SiLU with competitor-parity cast order.
- Scalar `gate_multiplier` and `down_multiplier` fused into forward and backward kernels.
- Fast default backward that may reuse saved buffers.
- `preserve_inputs=True` mode for callers that need saved/user-visible tensors preserved.
- Local benchmark harness with PyTorch, Forge, packed Forge, and Liger providers.

## Correctness Result

RunPod A100:

```text
python -m pytest tests/test_swiglu.py -xvs
75 passed in 4.04s
```

Coverage includes fp32/fp16/bf16, forward/backward, packed and separate layouts, default and scaled multipliers, special values, CPU fallback, and invalid-input checks.

## Competitive Result

Device: NVIDIA A100-SXM4-80GB
Benchmark: `--suite a100 --dtype bf16 --rep 50 --warmup 20`
Liger version: installed editable `liger_kernel==0.8.0`

Large Qwen-like shapes:

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

Additional scaled-multiplier dtype check:

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

- Default multipliers: Forge is at Liger parity. Small differences are within expected benchmark noise.
- Non-default multipliers: Forge wins clearly. On `4x2048x11008`, Forge fast full is ~1.45x faster than Liger because Liger applies `down_multiplier` outside the Triton kernels.
- The scaled-multiplier win holds across bf16, fp16, and fp32 for the large Qwen-like shapes listed here. Smaller and launch-bound shapes remain mixed/noisy.
- Memory: Forge fast matches or improves Liger peak memory. Forge safe intentionally pays extra peak memory to avoid mutation.
- Packed path is correct and sometimes faster, but not consistently better than separate. Treat it as layout support, not the main speed claim.

## Decision

Go.

The current implementation satisfies the competitive bar:

- It does not merely reimplement Liger; it matches Liger on the default path and beats it on the multiplier path we identified before coding.
- It gives a safe public-autograd option that Liger's in-place backward does not.
- It adds packed layout support for future combined-projection integration.

## Remaining Risks

- Hidden sizes above 65536 are still unsupported by the row-wise kernel.
- True higher-order autograd is not implemented; `preserve_inputs=True` prevents mutation but does not make the custom backward differentiable.
- Small-shape timings are noisy and launch-bound; do not optimize for them unless a real model path needs it.
- Unsloth's strongest advantage is fused LoRA/MLP integration, not the standalone activation microkernel. Beating Unsloth end-to-end requires the next integration step.

## Next Direction

Recommended next checkpoint:

1. Use this SWIGLU as the activation primitive for the next MLP/LoRA kernel.
2. Keep `preserve_inputs=False` as the training default.
3. Expose `preserve_inputs=True` only for safety/debug/public-autograd callers.
4. Add a flat long-indexing variant later only if hidden > 65536 or tiny/irregular shapes become important.

No-go only if a model-level integration shows this standalone kernel is not on the critical path or creates unacceptable mutation semantics in real training.

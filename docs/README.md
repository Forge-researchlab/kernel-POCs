# Kernel Research Docs

> Configure this folder by adding one focused research report per kernel or kernel family. Keep active task state in `../../TASKS.md`; use this folder for durable reasoning, competitor comparison, benchmark methodology, and failure boundaries that should be pushed with `kernel-POCs`.

Use this folder for kernel research before and during implementation in `../kernels/`.

Recommended report path:

```text
docs/<kernel-name>-research.md
```

Recommended report structure:

```text
# <Kernel Name> Research

## Operation Math
- Forward equation:
- Backward gradients:
- Numerical stability concerns:

## Supported Surface
- Shapes:
- Strides / layouts:
- Dtypes:
- Devices:
- Explicitly unsupported cases:

## PyTorch Reference
- Reference function:
- Assumptions:

## Liger Comparison
- Source paths:
- API:
- Fusion boundary:
- Optimizations:
- Edge cases:
- Pros:
- Cons / gaps:

## Unsloth Comparison
- Source paths:
- API:
- Fusion boundary:
- Optimizations:
- Edge cases:
- Pros:
- Cons / gaps:

## External Research
- Papers / issues / community kernels:
- Useful ideas:
- Rejected ideas:

## Improvement Hypotheses
- Idea:
- What it optimizes:
- Why it should matter:
- Benefiting shapes / dtypes:
- Risks and tradeoffs:
- Measurement plan:

## Selected Design
- Decision:
- Why this boundary:
- Why not fuse further:
- Implementation notes:

## Correctness Plan
- Forward parity:
- Backward parity:
- Gradcheck:
- Stress cases:
- Approximate paths, if any:

## Benchmark Plan
- Baselines:
- Shape strata:
- Dtypes:
- Device:
- Metrics:

## Known Failure Boundaries
- Dtype instability:
- Extreme shapes:
- Non-contiguous tensors:
- Large vocab / long sequence:
- Odd hidden sizes:
- Hardware-specific behavior:
```

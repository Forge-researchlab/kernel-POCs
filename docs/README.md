# Kernel Research Docs

> Configure this folder by adding one focused research report per kernel or kernel family. These docs are running notes for both agents and the user: update them at each stage instead of reconstructing the reasoning at the end.

Use this folder for kernel research before and during implementation in `../kernels/`.

Do not maintain a separate kernel profile file. The per-kernel report here is the canonical place for math, competitor study, first-principles analysis, research, implementation decisions, correctness results, benchmark results, and known boundaries.

Recommended report path:

```text
docs/<kernel-name>.md
```

Recommended report structure:

```text
# <Kernel Name> Research

## Operation Math
- Forward equation:
- Backward gradients:
- Numerical stability concerns:
- Mathematical assumptions:
- What is still unclear:

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
- Optimized axes:
- Input settings allowed:
- Input settings unsupported / unclear:
- Edge cases:
- Current model/training usage context:
- Surrounding kernels/components:
- Fusion opportunities:
- Why fusion likely stops where it does:
- Pros:
- Cons / gaps:
- What appears unoptimized:

## Unsloth Comparison
- Source paths:
- API:
- Fusion boundary:
- Optimizations:
- Optimized axes:
- Input settings allowed:
- Input settings unsupported / unclear:
- Edge cases:
- Current model/training usage context:
- Surrounding kernels/components:
- Fusion opportunities:
- Why fusion likely stops where it does:
- Pros:
- Cons / gaps:
- What appears unoptimized:

## First-Principles Analysis
- Unavoidable math:
- Unavoidable memory traffic:
- Avoidable intermediate tensors:
- Recompute vs save decisions:
- Fusion opportunities with surrounding kernels:
- Fusion limitations / where to stop:
- Edge cases worth targeting:
- Compatibility targets worth supporting:
- Shape regimes competitors may miss:
- Numerical stability risks:
- Backward correctness risks:

## External Research
- Papers / technical notes:
- Triton examples / official guidance:
- Community kernels / benchmark repos:
- Liger GitHub issues / PRs:
- Unsloth GitHub issues / PRs:
- PyTorch / HuggingFace / community issues:
- Useful ideas:
- Rejected ideas with reasons:
- Missing research due to blocked network / approval:

## Improvement Hypotheses
- Idea:
- What it optimizes:
- Why it should matter:
- Benefiting shapes / dtypes:
- Risks and tradeoffs:
- Correctness implications:
- Measurement plan:
- Fallback plan:

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

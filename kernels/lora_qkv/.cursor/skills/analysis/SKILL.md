# Performance Analysis & Bottleneck Identification Skill

> Generic skill for analyzing kernel performance, identifying bottlenecks,
> and recommending next optimizations. Usable across any kernel project.

## When to Use

- After any benchmark run completes
- When user asks "why is this slow?" or "what should I optimize next?"
- When a new version shows unexpected regression
- When comparing against baseline implementations
- Periodically to assess overall project progress

## Workflow

### Step 1: Gather Context

Read all relevant data before analyzing:

1. **Latest kernel code**: `experiments/v{N}/{KERNEL_NAME}_kernel_v{N}_{M}.py`
   - Understand the algorithmic approach
   - Count operations, memory accesses, kernel launches

2. **Baseline code**: `docs/artifacts/` and `reference/{KERNEL_NAME}_pytorch.py`
   - What operations does the baseline perform?
   - How many kernel launches? How many HBM round-trips?

3. **Benchmark results**: `benchmarks/results/*.csv` (most recent)
   - Latest timing numbers for our kernel
   - Baseline timing numbers
   - Across which shapes/ranks was it tested?

4. **Benchmark log**: `docs/benchmarks.md`
   - Historical performance trajectory
   - What's improved, what hasn't

5. **Previous analysis reports**: `docs/analysis/*.md`
   - Don't repeat work already done
   - Build on previous findings

### Step 2: Operation-Level Breakdown

Create a detailed comparison table:

```markdown
| Operation | Baseline (launches) | Ours (launches) | Baseline HBM R/W | Ours HBM R/W |
|-----------|--------------------|-----------------|--------------------|---------------|
| ...       | ...                | ...             | ...                | ...           |
```

For each operation, note:
- Number of kernel launches
- Bytes read from HBM
- Bytes written to HBM
- Whether it's compute-bound or memory-bound
- FLOPs (theoretical)

### Step 3: Bottleneck Analysis Checklist

Work through these potential bottlenecks in order:

#### Compute Bottlenecks
- [ ] Is the base matmul achieving good TFLOPS? (compare to peak for the GPU)
- [ ] Is Triton matmul competitive with cuBLAS? (target: >85% of cuBLAS)
- [ ] Are there redundant computations that can be eliminated?
- [ ] Is fp32 accumulation causing unnecessary register pressure?

#### Memory Bottlenecks
- [ ] How many times is input X read from HBM? (target: 1x per fused op)
- [ ] Are LoRA intermediates (X@A) materialized to HBM or kept in SRAM?
- [ ] Are large intermediates written then immediately re-read?
- [ ] What is the arithmetic intensity (FLOPs / bytes transferred)?

#### Launch Overhead
- [ ] Total kernel launches vs baseline?
- [ ] At small batch/seq, does launch overhead dominate?
- [ ] Could multiple kernels be fused into one?

#### Occupancy & Parallelism
- [ ] What's the thread block occupancy? (check register/SRAM usage)
- [ ] Is the grid large enough to saturate the GPU?
- [ ] Are there warp divergence issues?

#### LoRA-Specific
- [ ] Does LoRA overhead scale linearly with rank?
- [ ] At what rank does register spilling occur?
- [ ] Is the LoRA path compute-bound or memory-bound?

### Step 4: Identify Optimization Patterns

Based on bottlenecks found, suggest specific patterns:

| Pattern | When to Apply | Expected Benefit |
|---------|---------------|------------------|
| Fuse LoRA into base matmul | X@A materialized to HBM | Eliminate 1 R/W per projection |
| Fuse multiple projections | Same input read multiple times | Reduce X reads by N-1 |
| Keep cuBLAS for large matmuls | Triton < 85% cuBLAS throughput | Recover matmul speed |
| Triton epilogue fusion | Small ops after cuBLAS matmul | Eliminate intermediate writes |
| Register-level LoRA | rank <= 16 | Zero SRAM overhead for LoRA |
| SRAM ping-pong | rank 32-64 | Keep LoRA in shared memory |
| Software pipelining | Memory-bound kernels | Hide latency with prefetch |
| Tile size tuning | Suboptimal occupancy | Better SM utilization |

### Step 5: Run Comparison Microbenchmarks (if needed)

If the analysis suggests a specific hypothesis, design a targeted microbenchmark:

```python
# Template: isolate a single operation
ms_op = triton.testing.do_bench(lambda: operation(...), warmup=10, rep=50)
print(f"Operation X: {ms_op:.3f} ms")
print(f"Achieved bandwidth: {bytes_transferred / ms_op / 1e6:.1f} GB/s")
print(f"Achieved TFLOPS: {flops / ms_op / 1e9:.2f}")
```

### Step 6: Write Analysis Report

Save to `docs/analysis/YYYY-MM-DD_v{N}_description.md` with this template:

```markdown
# Analysis: {description}

**Date**: YYYY-MM-DD
**Kernel version**: v{N}_{M}
**Compared against**: {baseline}

## Summary

{1-2 sentence executive summary of findings}

## Performance Data

| Config | Baseline (ms) | Ours (ms) | Speedup | Bottleneck |
|--------|--------------|-----------|---------|------------|
| ...    | ...          | ...       | ...     | ...        |

## Operation Breakdown

{detailed operation-level comparison table}

## Bottleneck Identification

{which checklist items were identified as bottlenecks}

## Recommended Next Steps

1. {highest-impact optimization to try}
2. {second priority}
3. {optional stretch goal}

## Cross-Version Comparison

| Version | Date | Fwd (ms) | vs Baseline | Launches | Key Approach |
|---------|------|-----------|-------------|----------|--------------|
| ...     | ...  | ...       | ...         | ...      | ...          |
```

### Step 7: Cross-Version Comparison Table

ALWAYS include a table comparing ALL versions against ALL baselines:

```markdown
| Kernel | Time (ms) | vs Baseline | Launches (fwd) | Key Change |
|--------|-----------|-------------|----------------|------------|
| Baseline (unsloth/pytorch) | X.XX | 1.00x | N | — |
| v1 | X.XX | X.XXx | N | {approach} |
| v1_2 | X.XX | X.XXx | N | {refinement} |
| v2 | X.XX | X.XXx | N | {approach} |
| ... | ... | ... | ... | ... |
```

## Principles

1. **Measure before guessing** — never assume where the bottleneck is
2. **Isolate variables** — change one thing at a time between versions
3. **Operation-level comparison** — don't just compare total time; break it down
4. **Always show cross-version table** — makes progress (or lack thereof) visible
5. **Arithmetic intensity is key** — compute FLOPs/byte for every major operation
6. **cuBLAS is hard to beat** — know when to use it vs replace it
7. **Rank matters** — performance characteristics change with LoRA rank

## Project-Specific Paths (lora_qkv)

```
Base:        /workspace/kernel-POCs/kernels/lora_qkv/
Experiments: experiments/v{N}/lora_qkv_kernel_v{N}.py
Results:     benchmarks/results/*.csv
Analysis:    docs/analysis/
Baselines:   docs/artifacts/, reference/lora_qkv_pytorch.py
Benchmark:   docs/benchmarks.md
```

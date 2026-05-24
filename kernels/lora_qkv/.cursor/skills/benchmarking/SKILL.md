# Benchmark Execution & Comparison Skill

> Generic skill for running benchmarks, collecting results, and comparing
> kernel performance against baselines. Usable across any kernel project.

## When to Use

- After implementing a new kernel version or upgrade
- When user asks to benchmark or compare performance
- Before writing an analysis report
- When validating that a change didn't regress performance
- When collecting data for a specific shape/configuration

## Workflow

### Step 1: Identify What to Benchmark

1. **Determine scope**:
   - Forward pass only? Backward pass only? Both?
   - Single configuration or full sweep?
   - Which kernel versions to compare?

2. **Select configurations**:
   - Use standard shapes from the benchmark methodology (see below)
   - Include the primary benchmark config for quick comparison
   - Include rank sweep if LoRA-related changes were made

### Step 2: Run Benchmarks

#### Forward Pass
```bash
cd kernels/{KERNEL_NAME}
python benchmarks/bench_{KERNEL_NAME}.py --mode forward --save benchmarks/results/
```

#### Backward Pass
```bash
python benchmarks/bench_{KERNEL_NAME}.py --mode backward --save benchmarks/results/
```

#### Single Configuration (quick check)
```bash
python benchmarks/bench_{KERNEL_NAME}.py \
  --hidden {H} --num-heads {NH} --head-dim {HD} --rank {R} \
  --seq {S} --batch {B} --dtype bf16
```

#### Full Sweep
```bash
python benchmarks/bench_{KERNEL_NAME}.py --sweep --save benchmarks/results/
```

### Step 3: Time Analysis

For each benchmark result, verify:

1. **Sanity check**: Are times reasonable? (not 0.0ms, not seconds for small inputs)
2. **Variance**: Run multiple times if results seem noisy (increase `--rep`)
3. **Warmup**: Ensure first-run compilation costs aren't included
4. **Comparison**: Calculate speedup = baseline_ms / kernel_ms

### Step 4: Memory Analysis

Measure peak GPU memory for each kernel:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Run kernel
output = kernel_fn(...)
torch.cuda.synchronize()

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
```

Report:
- Peak memory (MB) for forward
- Peak memory (MB) for forward + backward
- Comparison against baseline memory usage
- Memory savings from fusion (intermediates eliminated)

### Step 5: Standard Shapes to Test

#### Model-Scale Configurations

| Model | Hidden | Num Heads | Head Dim | Total QKV Dim |
|-------|--------|-----------|----------|---------------|
| LLaMA-3 8B | 4096 | 32 | 128 | 4096 × 3 |
| LLaMA-3 70B | 8192 | 64 | 128 | 8192 × 3 |
| Mistral 7B | 4096 | 32 | 128 | 4096 × 3 |

#### Sweep Dimensions

| Parameter | Values | Notes |
|-----------|--------|-------|
| Batch size | 1, 2, 4, 8 | Micro-batch during training |
| Sequence length | 512, 1024, 2048, 4096 | Tokens per sample |
| LoRA rank | 8, 16, 32, 64 | Low-rank bottleneck |
| Dtype | bf16, fp32 | bf16 is production target |

#### Primary Benchmark Config (LLaMA-3 8B)
```
batch=4, seq=2048, hidden=4096, num_heads=32, head_dim=128, rank=16, dtype=bf16
→ M = batch*seq = 8192
→ N = hidden = 4096 (per Q/K/V projection)
→ K = hidden = 4096
```

### Step 6: Save Results

Results MUST be saved to CSV with timestamps:

```
benchmarks/results/{version}_{YYYYMMDD}_{HHMMSS}.csv
```

CSV columns (minimum):
```
version,mode,batch,seq_len,hidden,num_heads,head_dim,rank,dtype,baseline_ms,kernel_ms,speedup,memory_mb
```

### Step 7: Compare Against Baselines

Always compare against these baselines:

1. **PyTorch reference** (`reference/{KERNEL_NAME}_pytorch.py`):
   - Naive separate operations
   - Represents maximum kernel launch overhead

2. **Unsloth-style** (if applicable):
   - Autograd-level fusion with cuBLAS matmuls
   - Represents current SOTA open-source approach

3. **Previous versions** (all):
   - Show the trajectory of improvement
   - Identify regressions

### Step 8: Update docs/benchmarks.md

After collecting results:

1. Add new rows to the results table
2. Update the "Historical Summary" table with the new version
3. Note any surprising findings or regressions
4. Add the CSV filename for traceability

### Step 9: Trigger Analysis

After benchmarking is complete, invoke the **Analysis Skill** to:
- Identify bottlenecks in the new results
- Compare against previous versions
- Recommend next optimizations
- Write a dated analysis report to `docs/analysis/`

## Output Format

When reporting benchmark results to the user, always include:

```markdown
## Benchmark Results: {version}

**Config**: batch={B}, seq={S}, hidden={H}, heads={NH}, head_dim={HD}, rank={R}, dtype={D}

| Kernel | Fwd (ms) | Bwd (ms) | Memory (MB) | vs Baseline |
|--------|----------|----------|-------------|-------------|
| PyTorch ref | X.XX | X.XX | XXX | 1.00x |
| Ours (v{N}) | X.XX | X.XX | XXX | X.XXx |

CSV saved to: `benchmarks/results/{filename}.csv`
```

## Principles

1. **Reproducibility**: save ALL results to CSV with full configuration metadata
2. **Fair comparison**: same hardware, same input data, same warmup/rep counts
3. **Separate fwd/bwd**: forward and backward have different characteristics
4. **Include memory**: latency alone is insufficient; memory is a first-class metric
5. **Standard shapes**: always include the primary benchmark config for comparability
6. **Historical context**: show how performance has evolved across versions
7. **After benchmarking, analyze**: raw numbers without interpretation are incomplete

## Project-Specific Paths (lora_qkv)

```
Base:        /workspace/kernel-POCs/kernels/lora_qkv/
Harness:     benchmarks/bench_lora_qkv.py
Results:     benchmarks/results/
Methodology: docs/benchmarks.md
Analysis:    docs/analysis/
```

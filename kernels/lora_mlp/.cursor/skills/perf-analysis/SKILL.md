---
name: perf-analysis
description: Analyze and compare Triton kernel performance against Unsloth baseline. Use when asked to analyze bottlenecks, compare operations, profile kernels, find why Triton is slower than cuBLAS, suggest code changes, or pick better patterns from Unsloth's code.
---

# Performance Analysis & Comparison Skill

Diagnoses why our Triton kernel is slower than Unsloth's cuBLAS path, identifies specific bottleneck operations, and proposes targeted code changes.

## When Activated

1. **Read** current state (kernel code, benchmark numbers, Unsloth baseline).
2. **Analyze** the specific bottleneck the user is asking about.
3. **Compare** operation-by-operation against Unsloth's code.
4. **Propose** concrete code changes with expected impact.

---

## Step 1: Gather Context

Read these files before any analysis:

| What | Where |
|------|-------|
| Our latest kernel | `experiments/v*/` — highest version, prefer `_upgrade_` files |
| Unsloth baseline code | `reference/unsloth_baseline.py` |
| Latest benchmark CSV | `benchmarks/results/` — most recent CSV |
| Benchmark log | `docs/benchmarks.md` |
| Benchmark harness | `benchmarks/bench_lora_mlp.py` |

---

## Step 2: Operation-Level Breakdown

Break both implementations into individual GPU operations and compare.

### Unsloth's matmul_lora (per projection)

```
Op 1: torch.matmul(X, W.t())       — cuBLAS GEMM, highly optimized
Op 2: torch.matmul(X, A.t())       — cuBLAS GEMM (skinny: K×r)
Op 3: out.addmm_(XA, B.t(), α=s)   — cuBLAS GEMM (skinny: r×N) + add
```

Key: cuBLAS selects optimal algorithm per shape (GEMM, GEMV, etc.).

### Unsloth's full MLP

```
Ops 1-3:  matmul_lora for gate (3 cuBLAS)
Ops 4-6:  matmul_lora for up   (3 cuBLAS)
Op  7:    Triton _fg_kernel     (pointwise SwiGLU)
Ops 8-10: matmul_lora for down  (3 cuBLAS)
```

### Our Triton v1 (per projection)

```
Op 1: fused K-loop (base matmul + X@A accumulation)
Op 2: (XA) @ B tile multiply + add to acc
Op 3: store output
```

All in one kernel launch, but the tiled matmul must compete with cuBLAS.

---

## Step 3: Bottleneck Analysis Checklist

Run through these checks in order. Stop at the first bottleneck found.

### A. Is the base matmul itself slow?

Benchmark the kernel **without LoRA** (A=None, B=None) and compare to a bare `torch.matmul`:

```python
import triton
# Base matmul only
ms_cublas = triton.testing.do_bench(lambda: torch.matmul(X, W.t()), warmup=10, rep=50)
ms_triton = triton.testing.do_bench(lambda: fused_lora_matmul(X, W), warmup=10, rep=50)
print(f"cuBLAS: {ms_cublas:.3f}ms  Triton: {ms_triton:.3f}ms  ratio: {ms_cublas/ms_triton:.2f}x")
```

If Triton base matmul is >1.2x slower than cuBLAS, the bottleneck is matmul tuning, not LoRA.

**Common causes:**
- Missing autotune configs (try more BLOCK_M/N/K combos, especially 256×64, 64×256)
- No L2 swizzle (GROUP_SIZE_M for better L2 cache reuse)
- Missing `num_stages` tuning (software pipelining depth)
- Row-major vs column-major access patterns for W
- Not enough `num_warps` for large tiles

### B. How much overhead does the LoRA K-loop add?

Compare with-LoRA vs without-LoRA on the same shape:

```python
ms_no_lora = triton.testing.do_bench(lambda: fused_lora_matmul(X, W), warmup=10, rep=50)
ms_lora = triton.testing.do_bench(lambda: fused_lora_matmul(X, W, A, B, 1.0), warmup=10, rep=50)
overhead_pct = (ms_lora - ms_no_lora) / ms_no_lora * 100
print(f"LoRA overhead: {overhead_pct:.1f}%")
```

If overhead > 30%, the LoRA computation inside the K-loop is too expensive.

**Common causes:**
- A tile load inside the K-loop thrashes L1/L2 (A is small but loaded every iteration)
- Register pressure from xa accumulator (BLOCK_M × BLOCK_R registers)
- tl.dot for the xa accumulation is inefficient at small BLOCK_R

### C. Is the final (XA)@B dot a bottleneck?

This is a [BLOCK_M, BLOCK_R] × [BLOCK_R, BLOCK_N] matmul. For BLOCK_R=16, this is tiny and should be free. If BLOCK_R=64, it starts to matter.

### D. Memory bandwidth analysis

Estimate bytes read/written vs theoretical peak:

```python
# Per output tile [BLOCK_M, BLOCK_N]:
bytes_X = BLOCK_M * K * elem_size           # X rows (read once in fused loop)
bytes_W = BLOCK_N * K * elem_size           # W columns
bytes_A = BLOCK_R * K * elem_size           # A rows (LoRA, in fused loop)
bytes_B = BLOCK_R * BLOCK_N * elem_size     # B tile (loaded once after loop)
bytes_Y = BLOCK_M * BLOCK_N * elem_size     # output (written once)
total = bytes_X + bytes_W + bytes_A + bytes_B + bytes_Y
```

Compare `total / kernel_time` against GPU HBM bandwidth (e.g., A100 = 2 TB/s).

---

## Step 4: Specific Patterns to Try

### Pattern 1: L2 Swizzle (GROUP_SIZE_M)

cuBLAS reorders thread blocks for L2 cache locality. Add to the kernel:

```python
# In grid calculation, reorder blocks for L2 locality
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### Pattern 2: Preload A into shared memory

Since A is [r, K] and r is small, preload the entire A matrix into shared memory before the K-loop. This avoids re-loading A tiles every K iteration.

### Pattern 3: Separate LoRA as a post-pass

Instead of fusing into the K-loop, compute X@W^T first (matching cuBLAS speed), then add the LoRA term as a cheap post-pass. This trades one extra X read for a faster base matmul.

### Pattern 4: Borrow Unsloth's addmm_ pattern

If per-projection fusion can't beat cuBLAS, consider Unsloth's approach but fuse at the MLP level instead: call cuBLAS for individual matmuls but fuse the gate+up+SwiGLU pipeline to save HBM round-trips.

### Pattern 5: More autotune configs

```python
triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
```

---

## Step 5: Run Comparison Microbenchmark

When asked to diagnose, run this script to isolate the bottleneck:

```python
import torch, triton
from experiments.v1.lora_mlp_kernel_v1 import fused_lora_matmul
from reference.unsloth_baseline import matmul_lora as unsloth_matmul_lora

M, N, K, r = 8192, 14336, 4096, 16
dtype = torch.bfloat16
X = torch.randn(M, K, device="cuda", dtype=dtype)
W = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
A = torch.randn(r, K, device="cuda", dtype=dtype) * 0.02
B = torch.randn(N, r, device="cuda", dtype=dtype) * 0.02

# 1. Bare cuBLAS matmul
ms_bare = triton.testing.do_bench(lambda: torch.matmul(X, W.t()), warmup=10, rep=50)

# 2. Unsloth matmul_lora (cuBLAS + addmm)
ms_unsloth = triton.testing.do_bench(lambda: unsloth_matmul_lora(X, W, None, A, B, 1.0), warmup=10, rep=50)

# 3. Our kernel without LoRA
ms_triton_base = triton.testing.do_bench(lambda: fused_lora_matmul(X, W), warmup=10, rep=50)

# 4. Our kernel with LoRA
ms_triton_lora = triton.testing.do_bench(lambda: fused_lora_matmul(X, W, A, B, 1.0), warmup=10, rep=50)

print(f"cuBLAS bare matmul:     {ms_bare:.3f}ms")
print(f"Unsloth matmul_lora:    {ms_unsloth:.3f}ms  (overhead: {(ms_unsloth/ms_bare - 1)*100:.1f}%)")
print(f"Triton base (no LoRA):  {ms_triton_base:.3f}ms  (vs cuBLAS: {ms_bare/ms_triton_base:.2f}x)")
print(f"Triton fused LoRA:      {ms_triton_lora:.3f}ms  (vs Unsloth: {ms_unsloth/ms_triton_lora:.2f}x)")
print(f"LoRA fusion overhead:   {(ms_triton_lora/ms_triton_base - 1)*100:.1f}%")
```

Report findings as a table and recommend which pattern to try next.

---

## Step 6: Report Template

After analysis, produce a report in this format:

```markdown
## Performance Analysis — [date]

### Bottleneck: [one-line summary]

### Numbers
| Operation | Time (ms) | vs cuBLAS |
|-----------|-----------|-----------|
| cuBLAS bare matmul | X.XX | 1.00x |
| Unsloth matmul_lora | X.XX | X.XXx |
| Triton base (no LoRA) | X.XX | X.XXx |
| Triton fused LoRA | X.XX | X.XXx |

### Root Cause
[Explain why the bottleneck exists — be specific about which operation, which memory access pattern, which register pressure issue.]

### Recommended Fix
[Specific code change, with before/after pseudocode if applicable.]

### Expected Impact
[Quantified estimate: "This should bring Triton base from 0.7x to ~0.9x cuBLAS."]
```

Save the report to `docs/analysis/` with a dated filename.

---

## Step 7: Cross-Version Comparison

Every analysis must include a summary table comparing **all** kernel versions against all baselines. This tracks progress over time and prevents regressions.

### Required Baselines

| Baseline | What it measures | Code |
|----------|-----------------|------|
| cuBLAS bare `torch.matmul` | Theoretical speed floor for the base matmul | `torch.matmul(X, W.t())` |
| Unsloth `matmul_lora` | Per-projection: cuBLAS + addmm_ (3 launches) | `reference/unsloth_baseline.py` |
| Unsloth `LoRA_MLP` | Full MLP: 10 launches + Triton SwiGLU | `reference/unsloth_baseline.py` |

### Required Kernel Versions

Dynamically discover versions from `experiments/v*/`. For each version found, benchmark both per-projection and full-MLP.

### Comparison Table Template

**Per-projection** (M=8192, N=14336, K=4096, r=16, bf16):

| Kernel | Time (ms) | vs cuBLAS | vs Unsloth | Launches |
|--------|-----------|-----------|------------|----------|
| cuBLAS bare | X.XX | 1.00x | — | 1 |
| Unsloth matmul_lora | X.XX | X.XXx | 1.00x | 3 |
| **v1** fused LoRA | X.XX | X.XXx | X.XXx | 1 |
| **v1_upgrade_1** (if exists) | X.XX | X.XXx | X.XXx | 1 |
| **v2** (if exists) | X.XX | X.XXx | X.XXx | 1 |

**Full MLP** (batch=4, seq=2048, hidden=4096, intermediate=14336, r=16, bf16):

| Kernel | Time (ms) | vs Unsloth | Launches |
|--------|-----------|------------|----------|
| Unsloth LoRA_MLP | X.XX | 1.00x | 10 |
| **v1** (3× fused + SwiGLU) | X.XX | X.XXx | 4 |
| **v2** (if exists) | X.XX | X.XXx | — |

### Script to Generate the Table

```python
import sys, os, glob
sys.path.insert(0, '.')
import torch, triton
from reference.unsloth_baseline import (
    matmul_lora as unsloth_matmul_lora,
    apply_lora_mlp_swiglu as unsloth_lora_mlp,
    make_lora_mlp_params as unsloth_make_params,
    swiglu_fg_kernel,
)

M, N, K, r = 8192, 14336, 4096, 16
dtype = torch.bfloat16
X = torch.randn(M, K, device='cuda', dtype=dtype)
W = torch.randn(N, K, device='cuda', dtype=dtype) * 0.02
A = torch.randn(r, K, device='cuda', dtype=dtype) * 0.02
B = torch.randn(N, r, device='cuda', dtype=dtype) * 0.02

# Baselines
ms_cublas = triton.testing.do_bench(lambda: torch.matmul(X, W.t()), warmup=10, rep=50)
ms_unsloth = triton.testing.do_bench(lambda: unsloth_matmul_lora(X, W, None, A, B, 1.0), warmup=10, rep=50)

print(f"{'Kernel':<30} {'ms':>8} {'vs cuBLAS':>10} {'vs Unsloth':>11}")
print("-" * 62)
print(f"{'cuBLAS bare':<30} {ms_cublas:8.3f} {'1.00x':>10} {'—':>11}")
print(f"{'Unsloth matmul_lora':<30} {ms_unsloth:8.3f} {ms_cublas/ms_unsloth:9.2f}x {1.0:10.2f}x")

# Discover and benchmark each version
for vdir in sorted(glob.glob('experiments/v*')):
    vname = os.path.basename(vdir)
    # Find the latest .py file in the version directory
    py_files = sorted(glob.glob(os.path.join(vdir, '*.py')))
    for py_file in py_files:
        mod_name = os.path.splitext(os.path.basename(py_file))[0]
        # Import fused_lora_matmul from the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'fused_lora_matmul'):
            fn = mod.fused_lora_matmul
            ms = triton.testing.do_bench(lambda: fn(X, W, A, B, 1.0), warmup=10, rep=50)
            label = mod_name.replace('lora_mlp_kernel_', '')
            print(f"{label:<30} {ms:8.3f} {ms_cublas/ms:9.2f}x {ms_unsloth/ms:10.2f}x")
```

Run this script after every kernel change and paste the output into `docs/benchmarks.md`.

### When to Update

- After creating a new version or upgrade
- After changing autotune configs
- After applying any optimization from the patterns list
- When the user asks "where are we at" or "what are our current speeds"

---

## Principles

- **Measure before guessing** — always run the microbenchmark before proposing changes.
- **Isolate one variable** — compare base matmul separately from LoRA overhead.
- **Compare at the operation level** — don't just compare end-to-end; break into individual ops.
- **Always show the cross-version table** — every analysis must end with the comparison table so progress is visible.
- **cuBLAS is the floor** — if we can't match cuBLAS on the base matmul, no amount of LoRA fusion will help.
- **Borrow what works** — if Unsloth uses a specific op (e.g., addmm_ with alpha) that's faster, adopt it.

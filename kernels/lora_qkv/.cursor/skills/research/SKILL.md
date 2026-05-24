# Iterative Kernel Research Skill

> Generic skill for iterative kernel research, development, and versioning.
> Usable across any kernel project in the kernel-POCs monorepo.

## When to Use

- Starting research on a new kernel optimization
- Creating a new version (v1, v2, v3) of an existing kernel
- Upgrading a version (v1_2, v1_3) with a refinement
- Debugging a correctness or performance issue
- Resuming work after a break (need to rediscover project state)

## Workflow

### Phase 1: Scan Current State

Before writing any code or docs, read the project to understand where things stand:

1. **Read project docs**:
   - `README.md` — done gates, structure overview
   - `CHANGELOG.md` — what versions exist, latest results
   - `docs/research.md` — improvement axes, open questions
   - `docs/benchmarks.md` — current performance numbers
   - `docs/artifacts/ANALYSIS.md` — baseline analysis

2. **Read latest kernel code**:
   - Find the highest version in `experiments/v{N}/`
   - Read the latest file (e.g., `{KERNEL_NAME}_kernel_v{N}_{M}.py`)
   - Read its module docstring for approach and limitations

3. **Read reference implementations**:
   - `reference/{KERNEL_NAME}_pytorch.py` — correctness ground truth
   - Any baseline code in `docs/artifacts/`

4. **Read test results**:
   - `tests/test_{KERNEL_NAME}.py` — what's tested, what's passing
   - `benchmarks/results/` — latest CSV outputs

5. **Summarize state**: Before proceeding, write a brief summary of:
   - Current best version and its performance
   - Known limitations / bottlenecks
   - What the user is asking for

### Phase 2: Act on Request

Based on the user's request, execute one of these patterns:

#### A) New Research / Exploration

1. Identify the question to answer (e.g., "how does library X handle this?")
2. Search relevant code/docs/papers
3. Document findings in `docs/research.md` (add to appropriate section)
4. If code artifacts found, save to `docs/artifacts/`
5. Update `CHANGELOG.md` with research entry

#### B) New Version (v{N+1})

1. Identify the new algorithmic approach (must be fundamentally different from v{N})
2. Create `experiments/v{N+1}/{KERNEL_NAME}_kernel_v{N+1}.py`
3. Start from the PyTorch reference logic, then optimize
4. Write module docstring explaining approach, design, and expected benefits
5. Run correctness tests against reference
6. Benchmark against previous version AND baselines
7. Update all docs (CHANGELOG, benchmarks.md, research.md)

#### C) Minor Upgrade (v{N}_{M+1})

1. Identify the specific improvement (tuning, bug fix, optimization)
2. Copy latest version: `v{N}_{M}.py` → `v{N}_{M+1}.py`
3. Apply the change
4. Test correctness (must still pass all existing tests)
5. Benchmark (must show improvement or document why not)
6. Update CHANGELOG with the refinement

#### D) Debug

1. Reproduce the issue (failing test, wrong output, crash)
2. Isolate: use small inputs, fp64, `tl.device_print`
3. Compare tile-by-tile against reference
4. Fix in a new minor version (never edit existing files)
5. Verify fix with tests

### Phase 3: Update Living Docs

After any change, update these docs:

1. **CHANGELOG.md**: Add entry with date, approach, results, limitations
2. **docs/benchmarks.md**: Update results table, add new data points
3. **docs/research.md**: Update open questions, mark resolved ones
4. **README.md**: Update done gates if any criteria newly met

## Done Gates Reference Table

| Gate | Criteria | Verification |
|------|----------|--------------|
| Forward correctness | Output matches PyTorch ref within tolerance | `pytest tests/ -k forward` |
| Backward correctness | Gradients match PyTorch autograd | `pytest tests/ -k backward` |
| Gradcheck | `torch.autograd.gradcheck` passes in fp64 | `pytest tests/ -k gradcheck` |
| Multi-shape | Tested at small + production shapes | Parametrized test sweep |
| Multi-rank | Tested at r=8,16,32,64 | Parametrized test sweep |
| Benchmark baseline | Compared against baseline at production scale | CSV in `benchmarks/results/` |
| Speedup achieved | Measurable improvement over baseline | Results in `docs/benchmarks.md` |
| Memory within budget | Peak memory <= baseline | Benchmark memory column |

## Principles

1. **Never modify old versions** — every file is an immutable snapshot
2. **Benchmark before and after** — quantify impact of every change
3. **fp32 accumulation** — always accumulate in fp32, cast output to input dtype
4. **Measure then optimize** — profile before guessing at bottlenecks
5. **Document failures** — record what didn't work and why in CHANGELOG
6. **Small steps** — one change per minor version, easy to bisect
7. **Reference is ground truth** — if tests fail, the kernel is wrong (not the reference)

## Project-Specific Paths (lora_qkv)

```
Base:        /workspace/kernel-POCs/kernels/lora_qkv/
Reference:   reference/lora_qkv_pytorch.py
Tests:       tests/test_lora_qkv.py
Benchmarks:  benchmarks/bench_lora_qkv.py
Results:     benchmarks/results/
Experiments: experiments/v{N}/lora_qkv_kernel_v{N}.py
Docs:        docs/research.md, docs/benchmarks.md
Changelog:   CHANGELOG.md
```

---
name: lora-mlp-kernel-research
description: Iterative research driver for the LoRA MLP fused kernel. Use when asked to research, plan next steps, update docs, create/improve kernel versions, benchmark, or review current state.
---

# LoRA MLP Kernel — Research Skill

Generic, iterative skill. Does not hardcode specifics — reads the project to discover current state, then acts.

## When Activated

1. **Scan** the project to understand where things stand.
2. **Act** on whatever the user asked (new version, benchmark, research, etc.).
3. **Update** the living docs so the next invocation starts informed.

---

## Step 1: Scan Current State

Before doing anything, read these files (skip any that don't exist yet):

| What | Where | Why |
|------|-------|-----|
| Research context | `docs/research.md` | Understand the math, related work, open questions |
| Benchmark log | `docs/benchmarks.md` | Know current perf numbers and methodology |
| Change history | `CHANGELOG.md` | Know what versions exist and what each tried |
| Project overview | `README.md` | Done-gates, key params, structure |
| Latest kernel code | `experiments/v*/` — read the highest-numbered version, prefer `_upgrade_` files over base | Understand current implementation |
| Tests | `tests/test_lora_mlp.py` | Know what's tested, what's passing |
| Benchmarks script | `benchmarks/bench_lora_mlp.py` | Know what's measured and how |
| Benchmark results | `benchmarks/results/` — read any CSV/JSON files | Know actual numbers |
| Reference impls | `reference/` — read any `.py` files | Know the baselines being compared against |

After scanning, form a mental model of: **what exists, what works, what's broken, what's never been tried.**

---

## Step 2: Act on the Request

### If asked to **research / explore**

1. Read `docs/research.md` — identify which research axes have been explored vs. untouched.
2. Search for recent papers, libraries, or Triton patterns relevant to the open questions.
3. Propose concrete next experiments ranked by expected impact vs. effort.
4. Update `docs/research.md` with new findings under a dated subsection.

### If asked to **create a new kernel version**

1. Determine the next version number from `experiments/v*/`.
2. Create `experiments/v{N}/lora_mlp_kernel_v{N}.py`.
3. Start the file with a module docstring explaining the approach and how it differs from the previous version.
4. Implement forward (+ backward if applicable), validate against PyTorch reference.
5. Run tests: `pytest tests/test_lora_mlp.py -v`
6. Run benchmarks: `python benchmarks/bench_lora_mlp.py --save benchmarks/results/`
7. **Update docs** (Step 3 below).

### If asked to **upgrade an existing version**

1. Copy `lora_mlp_kernel_v{N}.py` → `lora_mlp_kernel_v{N}_upgrade_{M}.py`.
2. Document what changed in the docstring.
3. Run full test + benchmark suite.
4. **Update docs** (Step 3 below).

### If asked to **benchmark**

1. Read `benchmarks/bench_lora_mlp.py` to understand the harness.
2. Run benchmarks at standard shapes (check `docs/benchmarks.md` for the sweep config).
3. Compare against all baselines listed in `docs/benchmarks.md`.
4. Save results to `benchmarks/results/`.
5. **Update docs** (Step 3 below).

### If asked to **debug / fix correctness**

1. Run `pytest tests/test_lora_mlp.py -v` to see what fails.
2. If `gradcheck` fails: check backward math, fp32 accumulation, mask bounds.
3. If forward diverges: compare tile-by-tile against the PyTorch reference.
4. Fix in-place or as an upgrade. Run tests again.

---

## Step 3: Update Living Docs

After every action, update the relevant docs so the project stays self-describing:

### `CHANGELOG.md`

Add or update an entry with:
- **Approach**: what algorithmic idea this version/upgrade uses
- **Changes**: what specifically changed from the previous state
- **Results**: benchmark numbers (forward speedup, backward speedup, memory)
- **Limitations**: what doesn't work well, what should be tried next

### `docs/benchmarks.md`

- Fill in the results table for the latest version.
- Add a new section if a new version was created.
- Update the "Historical Summary" table at the bottom.

### `docs/research.md`

- Move any answered "Open Questions" into a "Resolved" section with the answer.
- Add new open questions discovered during the work.
- Add any new research axes or related work found.
- If a fusion strategy was tried, record whether it worked and why/why not.

### `README.md`

- Check/uncheck done-gates as they pass.
- Update "Quick Start" if commands or file paths changed.

---

## Done-Gates Reference

A kernel version is shippable when ALL of these pass:

| Gate | Check |
|------|-------|
| G1 | `torch.autograd.gradcheck` at fp64 with default tolerances |
| G2 | Forward bf16 correctness at 4+ shapes (`rtol=1e-2`) |
| G3 | Backward bf16 correctness (`rtol=1e-2`, no NaN) |
| G4 | Speed ≥ 1.0× PyTorch (fwd+bwd, every tested shape) |
| G5 | Peak memory ≤ PyTorch (every tested shape) |
| G6 | CSV row in `benchmarks/results/` per (kernel, shape, dtype) |
| G7 | `chapter_lora_mlp.md` with formulas + benchmark table |

---

## Principles

- **Never modify a previous version file** — copy forward, iterate on the copy.
- **Benchmark before and after every change** — no unquantified claims.
- **fp32 accumulation** in all reductions, even when inputs are bf16.
- **Test at non-power-of-2 shapes** to catch masking bugs.
- **Flush L2 cache** between benchmark runs.
- **Read the project before writing** — this skill is iterative, not generative from scratch.

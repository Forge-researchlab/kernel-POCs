# Internet Deep Research Skill

> Systematic internet research for finding papers, SOTA techniques, GPU optimization patterns,
> and Triton/CUDA tricks relevant to the current kernel project. Complements the project-level
> research skill by bringing in **external knowledge**.

## When to Use

- Starting a new kernel project and need to survey the landscape
- Looking for optimization techniques to break through a performance plateau
- Investigating how other projects solve similar problems
- Checking for new papers/commits in Triton, PyTorch, CUTLASS repos
- User asks "what's out there?" or "find me papers on X"
- Before designing a new kernel version (gather prior art first)

## Prerequisites

This skill requires **WebSearch** and **WebFetch** tools. It produces artifacts in the
project's `docs/research/` directory.

## Research Phases

### Phase 1: Baseline Landscape

**Goal**: Understand what exists — who does what, and how well.

Run these searches and document findings:

```
Search queries:
  - "fused LoRA kernel Triton"
  - "{KERNEL_NAME} kernel fusion GPU"
  - "unsloth matmul_lora implementation"
  - "PEFT LoRA efficient forward pass"
  - "grouped GEMM LoRA GPU"
  - "cuBLAS vs Triton matmul benchmark {YEAR}"
  - "{KERNEL_NAME} attention projection fusion"
  - "LoRA-aware GEMM kernel"
```

For each result:
1. Open the URL with WebFetch to get full content
2. Extract: approach, performance claims, limitations
3. Save to `docs/research/papers/YYYY-MM-DD_short_title.md`
4. Add entry to `docs/research/papers/INDEX.md`

### Phase 2: Technique Mining

**Goal**: Find specific optimization patterns from papers, blogs, and codebases.

```
Search queries:
  - "Triton matmul optimization techniques {YEAR}"
  - "GPU kernel fusion LoRA training"
  - "attention projection fusion Triton"
  - "grouped GEMM Triton implementation"
  - "cuBLAS epilogue fusion custom kernel"
  - "register tiling Triton kernel"
  - "SRAM management Triton tl.dot"
  - "Triton autotune best practices"
  - "software pipelining Triton num_stages"
  - "L2 cache swizzle GPU kernel"
  - "warp specialization Triton"
  - "persistent kernel Triton matmul"
```

For each technique found:
1. Summarize the technique in 3-5 sentences
2. Assess applicability: high/medium/low with justification
3. Note if code is available (link to it)
4. Add to `docs/research/techniques/INDEX.md`
5. If the technique is highly applicable, create a dedicated file in `docs/research/techniques/`

### Phase 3: Architecture-Specific Research

**Goal**: Understand GPU-specific optimization opportunities.

```
Search queries:
  - "A100 vs H100 Triton kernel performance"
  - "H100 tensor core Triton programming"
  - "NVIDIA A100 SRAM shared memory size"
  - "H100 TMA Triton support"
  - "GPU memory bandwidth A100 H100 specs"
  - "Triton kernel occupancy optimization"
  - "CUDA compute capability 8.0 9.0 differences Triton"
  - "bf16 tensor core throughput A100 H100"
```

For each finding:
1. Document hardware-specific details
2. Note implications for kernel design decisions
3. Add to `docs/research/techniques/INDEX.md` under "Architecture-Specific" section

### Phase 4: Bleeding Edge

**Goal**: Find the latest work that might not be in papers yet.

```
Search queries:
  - "site:github.com triton matmul LoRA" (recent commits)
  - "site:arxiv.org LoRA efficient inference {YEAR}"
  - "site:arxiv.org fused kernel attention projection {YEAR}"
  - "Triton 3.0 new features"
  - "PyTorch torch.compile custom triton kernel"
  - "FlashAttention-3 Triton"
  - "CUTLASS 3.x grouped GEMM"
  - "site:discuss.pytorch.org Triton kernel optimization"
```

For each finding:
1. Note the date and freshness
2. Assess maturity (experimental/production-ready)
3. Add to appropriate INDEX.md

## Output Structure

### Paper/Finding File Template

Save each significant finding to `docs/research/papers/YYYY-MM-DD_short_title.md`:

```markdown
# {Title}

**Authors**: {authors or organization}
**Date**: {publication/discovery date}
**URL**: {link}
**Source type**: paper / blog / code / discussion
**Relevance**: high / medium / low

## Summary

{3-5 sentences describing what this is about}

## Key Techniques

- {technique 1}: {brief description}
- {technique 2}: {brief description}

## Applicability to {KERNEL_NAME}

{2-3 sentences on how this applies to our specific kernel project}

**Applicable aspects**:
- {specific technique or insight we can use}
- {another applicable aspect}

**Not applicable / limitations**:
- {why some parts don't apply}

## Code Snippets

{If code is available, include the most relevant snippets with annotations}

## Follow-up Actions

- [ ] {concrete next step based on this finding}
```

### Papers INDEX.md Template

Maintain `docs/research/papers/INDEX.md` as a living table:

```markdown
# Research Papers & Findings Index

> Master index of all papers, blog posts, and code findings.
> Sorted by relevance to the current project. Updated after each research session.

| Date Found | Title | Source | Relevance | Key Takeaway | File |
|------------|-------|--------|-----------|--------------|------|
| YYYY-MM-DD | ... | paper/blog/code | high/med/low | 1-sentence summary | [link](./YYYY-MM-DD_short_title.md) |
```

### Techniques INDEX.md Template

Maintain `docs/research/techniques/INDEX.md`:

```markdown
# Optimization Techniques Catalog

> Catalog of optimization techniques discovered through research.
> Status tracks whether we've tried each technique in our kernel.

| Technique | Source | Status | Expected Benefit | Notes |
|-----------|--------|--------|------------------|-------|
| ... | paper/blog | untried/tried/adopted/rejected | ... | ... |

## Tried & Adopted

{Techniques that worked and are in our current best kernel}

## Tried & Rejected

{Techniques we tried but didn't help, with explanation of why}

## Untried — High Priority

{Techniques we haven't tried yet but expect high impact}

## Untried — Low Priority

{Techniques that might help but are lower priority}
```

### Baselines INDEX.md Template

Maintain `docs/research/baselines/INDEX.md`:

```markdown
# Known Baselines

> Catalog of known baseline implementations and their characteristics.

| Baseline | Source | Approach | Launches (fwd) | X HBM Reads | Strengths | Weaknesses |
|----------|--------|----------|-----------------|-------------|-----------|------------|
| ... | ... | ... | ... | ... | ... | ... |
```

## Post-Research Steps

After completing a research session:

1. **Update `docs/research.md`** — Add a dated section summarizing new findings:
   ```markdown
   ### YYYY-MM-DD — Internet Research: {topic}

   **Queries run**: {list of search queries}
   **Papers/findings**: {count} new entries added to INDEX

   Key takeaways:
   1. {most important finding}
   2. {second most important}
   3. {third}

   Impact on our approach:
   - {how this changes or validates our kernel design}
   ```

2. **Update technique catalog** — Add any new techniques to `docs/research/techniques/INDEX.md`

3. **Update baselines** — If new baselines discovered, add to `docs/research/baselines/INDEX.md`

4. **Prioritize** — Re-rank techniques by expected impact given new information

## Principles

1. **Breadth first, depth second** — survey the landscape before deep-diving into any one paper
2. **Relevance filter** — not everything found is applicable; rate honestly
3. **Code over claims** — actual implementations are more valuable than paper descriptions
4. **Date matters** — prefer recent work (Triton evolves fast, 2-year-old patterns may be outdated)
5. **Reproduce before trusting** — performance claims from papers need independent verification
6. **Document negatives** — "this doesn't apply because X" is valuable information
7. **Connect to our kernel** — every finding should end with "what does this mean for us?"

## Project-Specific Paths (lora_qkv)

```
Base:          /workspace/kernel-POCs/kernels/lora_qkv/
Papers:        docs/research/papers/
Techniques:    docs/research/techniques/
Baselines:     docs/research/baselines/
Research log:  docs/research.md
Artifacts:     docs/artifacts/
```

### lora_qkv-Specific Search Queries

In addition to the generic queries above, run these project-specific searches:

```
- "fused QKV projection LoRA Triton"
- "attention weight projection kernel fusion"
- "multi-head attention LoRA efficient implementation"
- "grouped query attention LoRA kernel"
- "QKV packed weight LoRA compatibility"
- "LoRA rank register pressure GPU kernel"
- "cuBLAS addmm_ fusion LoRA"
- "FlashAttention QKV projection integration"
```

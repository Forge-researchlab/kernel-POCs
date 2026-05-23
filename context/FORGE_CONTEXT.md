# Forge: GPU Kernel Optimization Library for LLM Training

## What Forge Is

Forge is an internal research project building a library of **custom Triton GPU kernels** that accelerate LLM fine-tuning and training. It replaces standard PyTorch operations inside HuggingFace Transformer models with hand-written, fused Triton kernels that are faster and use significantly less VRAM.

The project is developed by a team at Meesho, initially targeting **Qwen3-8B** as the reference model architecture, with plans to expand to Llama, Gemma, Mistral, and others.

Forge sits in the same space as **Unsloth** (proprietary optimizations, 500+ models) and **Liger-Kernel** (open-source Triton kernels, BSD-2 license), but with a research-first design: every kernel ships with a pluggable registry, automated A/B benchmarking, convergence testing, and published gradient derivations. The long-term goal is an open-source release at CP5 (Day 75).

---

## The Team

**Core members:** Srinivasan, Jithamanyu, Shaurya, Devansh, Hitarth, Gautam (guide/anchor, relatively busy), Xhitij

**In training:** Sasank (under training with Xhitij, will contribute later), Prakash (TBD)

The team is organized into **3 subteams** of 2-3 people. Each subteam takes one L1 (easier) kernel and one L2 (harder) kernel per sprint. Gautam anchors one subteam as a guide given his limited availability.

---

## What We Are Building

### The Forge Library

A Python package (`forge/`) with the following structure:

```
forge/
├── forge/
│   ├── kernels/               # All Triton kernels
│   │   ├── rmsnorm.py         # Triton kernel + autograd.Function wrapper
│   │   ├── layernorm.py
│   │   ├── rope.py
│   │   ├── cross_entropy.py
│   │   ├── fused_linear_ce.py
│   │   ├── embedding.py
│   │   └── registry.py        # forge.kernels.register() / forge.kernels.get()
│   ├── patching/              # Model patching infrastructure
│   │   ├── patch.py           # forge.patch(model, kernels=['rmsnorm', 'rope', ...])
│   │   ├── unpatch.py         # forge.unpatch(model) for correctness verification
│   │   └── qwen3.py           # Qwen3-specific patch mappings
│   ├── training/              # Training pipeline (SFT, GRPO)
│   └── export/                # GGUF export pipeline
├── tests/
│   ├── kernels/               # Per-kernel: gradcheck, correctness vs PyTorch ref, benchmarks
│   ├── integration/           # Patched model output == unpatched model output
│   └── convergence/           # Multi-step training: patched loss curve matches unpatched
├── benchmarks/                # Automated kernel benchmarking harness
├── configs/                   # YAML training configs
└── docs/
    └── kernel_comparisons/    # Per-kernel Liger vs Unsloth analysis
```

### Kernel API Contract

Every kernel must:

- Have a Triton forward kernel + backward kernel (both `@triton.jit`)
- Be wrapped in a `torch.autograd.Function` (forward, backward, `ctx.save_for_backward`)
- Be registered via `@register_kernel("kernel_name")`
- Pass `torch.autograd.gradcheck` against a PyTorch reference implementation
- Pass forward correctness tests (output matches PyTorch ref within 1e-5 relative tolerance)
- Have a benchmark entry in the benchmark harness
- Support bf16 and fp16 inputs with fp32 accumulation internally
- Integrate into HuggingFace models via `forge.patch(model)` with zero accuracy loss

---

## Roadmap

### CP1 — Day 15: Qwen3 Single-GPU MVP

**Kernels (first sprint, 6 kernels, all forward + backward):**

| Kernel | Difficulty | Description |
|--------|-----------|-------------|
| RMSNorm | L1 | Root mean square normalization, fwd+bwd |
| LayerNorm | L1 | Standard layer normalization, fwd+bwd |
| Embedding | L1 | Token embedding kernel, fwd+bwd |
| RoPE (fused Q+K) | L2 | Rotary positional encoding applied to Q and K jointly |
| Cross-Entropy (chunked vocab) | L2 | Chunked cross-entropy to avoid materializing full logits |
| Fused Linear + Cross-Entropy | L2 | Linear projection + CE loss without logit materialization |

**Additional CP1 targets:**
- LoRA MLP fused kernel, LoRA QKV fused kernel
- Manual autograd (bypass `torch.autograd` overhead)
- int64 indexing for 500K+ context lengths
- SFT and GRPO training methods (LoRA, QLoRA, 8-bit, 16-bit)
- Gradient checkpointing with CPU offload
- Sample packing (bin-packing, padding-free)
- Chunked log-softmax for memory-efficient GRPO
- GGUF export, HuggingFace Hub upload
- Kernel registry with automated A/B benchmarking
- Convergence test suite

**Performance targets:** ~2x training speed vs HuggingFace baseline, ~60% VRAM reduction.

### CP2 — Day 30: Multi-Architecture + Multi-GPU

- Add Llama 3/3.1/3.2/3.3, Llama 4, Gemma 2/3, partial Mistral/Mixtral, Qwen3 MoE
- Multi-GPU via FSDP2
- Full fine-tuning support
- Partial FP8 training, partial AMD ROCm
- Embedding kernel
- WandB/TensorBoard integration
- Published benchmarks vs competitors

### CP3 — Day 45: Post-Training + Vision

- DPO, PPO, ORPO, CPO, SimPO, KTO training methods
- DPO/ORPO/CPO/SimPO/KTO fused loss kernels, KL Divergence kernel
- Vision models (Qwen-VL, Llama-Vision)
- Phi-3/4, partial DeepSeek-R1, partial embedding models
- Optimized Fused Linear + CE v2
- Partial quantized matmul kernels (FP8, INT4)
- Multi-node training (partial), AMD ROCm, partial Intel XPU
- Partial web UI, QAT support

### CP4 — Day 60: Research Divergence

- Novel research: custom Flash Attention kernel, ring/blockwise attention variants, novel RL algorithms
- Pre-training from scratch
- Knowledge distillation, full QAT
- TTS/audio models (partial)
- Optimized Fused Linear + CE v3
- Full quantized matmul kernels
- ONNX/TensorRT export (partial)
- 20+ model architectures

### CP5 — Day 75: Open Source Launch

- Public repo, documentation site, community Discord
- PyPI package, contribution guidelines
- 30+ model architectures
- Full TTS/audio, ONNX/TensorRT export
- Auto dataset creation (partial)

---

## Competitive Landscape

### Forge vs Competitors at CP1 (Day 15)

| Capability | Forge CP1 | Unsloth Free | Unsloth Pro | Liger-Kernel | Torchtune | Axolotl | LLaMA-Factory |
|------------|-----------|--------------|-------------|--------------|-----------|---------|---------------|
| Custom Triton kernels | 10 kernels | Yes (all) | Yes (all) | Yes (12+ kernels) | No (torch.compile) | No | No |
| Training speed vs HF | ~2x | ~2x | ~2.5x * N GPUs | ~1.2x | ~1.3x | ~1x | ~1x |
| VRAM reduction | ~60% | ~60% | ~80% | ~60% | Moderate | Moderate | Moderate |
| Model architectures | 1 (Qwen3) | 500+ | 500+ | 10+ | ~15 | ~30+ | ~100+ |
| Multi-GPU | No | No | Yes (8 GPUs) | Yes (FSDP/DS) | Yes | Yes | Yes |
| LoRA-specific kernels | Yes | Yes | Yes | No | No | No | No |
| Manual autograd | Yes | Yes | Yes | Partial | No | No | No |
| Kernel registry + A/B test | Yes | No | No | No | No | No | No |
| Convergence testing | Yes | No | No | Yes | No | No | No |
| Modular kernel API | Yes | No (coupled) | No (coupled) | Yes | N/A | N/A | N/A |
| License | TBD (will be OSS) | Apache 2.0 | Proprietary | BSD-2 | BSD-3 | Apache 2.0 | Apache 2.0 |
| Price | Free | Free | Contact sales | Free | Free | Free | Free |

### Key Differentiators

**Where Forge is stronger than all competitors:**
- **Research infrastructure**: Kernel registry, automated A/B benchmarking, convergence test suite, and published gradient derivations are unique to Forge. No other tool lets you plug in a new kernel variant and automatically compare it against baselines.
- **Modular kernel API**: Kernels can be used standalone or composed. Unsloth's kernels are tightly coupled to its framework.
- **Gradcheck verification**: Every kernel is numerically verified against PyTorch reference. Liger does this internally but doesn't expose the framework.

**Where Forge is behind at CP1:**
- **Model coverage**: 1 architecture (Qwen3) vs 500+ (Unsloth) or 100+ (LLaMA-Factory). This grows across CPs.
- **Multi-GPU**: Not supported until CP2. Unsloth Pro and Torchtune support multi-GPU already.
- **Full fine-tuning and pre-training**: Not in CP1, comes in CP2/CP4.
- **Web UI / no-code**: Unsloth Studio and LlamaBoard exist. Forge web UI is CP4+.
- **torch.compile**: Only partial support. Liger and Torchtune fully support it.

**Where Forge matches Unsloth Free (our primary comparison):**
- Same kernel set (RMSNorm, RoPE, SwiGLU, CE, Fused Linear CE, LoRA kernels)
- Same training speed (~2x) and VRAM reduction (~60%)
- Same 0% accuracy loss guarantee
- Same training methods at CP1 (SFT, GRPO, LoRA/QLoRA)
- Same export capabilities (GGUF, HF Hub, safetensors)

---

## How a Kernel Gets Built: The 8-Phase Lifecycle

Each subteam follows this process for every kernel (L1 or L2):

| Phase | Days | What Happens |
|-------|------|--------------|
| 1. Study the Math | 1-2 | Hand-derive forward equation and backward gradients. Understand computational characteristics (memory-bound vs compute-bound). Calculate memory footprint at Qwen3-8B scale (hidden=4096, intermediate=11008). |
| 2. Comparative Study | 2-4 | Compare PyTorch baseline, Liger, and Unsloth implementations side-by-side. Deliver a 15-20 min presentation to the full team. Fill out a tradeoff table (HBM reads/writes, saved tensors, memory per layer, fp32 accumulation, int64 indexing). Decide Forge's approach. |
| 3. Implement Kernel | 4-8 | Write Triton forward kernel, backward kernel, `torch.autograd.Function` wrapper, and `torch.nn.Module` wrapper. Register in the kernel registry. Test forward output against PyTorch reference. |
| 4. HuggingFace Patching | 8-9 | Write patch function that intercepts the specific HF class/method (e.g., `Qwen3RMSNorm.forward`). Verify patched vs unpatched outputs match. Verify patches are reversible and composable with LoRA/quantization. |
| 5. Testing | 9-11 | Three levels: (1) `torch.autograd.gradcheck` with float64, (2) forward+backward correctness vs PyTorch across dtypes (bf16, fp16) and shapes, (3) convergence testing over 500-1000 training steps. |
| 6. Benchmarking | 11-13 | Measure speed (forward, backward, fwd+bwd) and peak VRAM across batch sizes (1-16), sequence lengths (512-8192), and dtypes. Compare against PyTorch, Liger, Unsloth. Optimize via autotune, reduced HBM access, or eliminated intermediate tensors. |
| 7. Integration | 13-14 | Final PR with kernel code, autograd wrapper, patch registration, all 3 test levels, benchmarks, and documentation. Code review with checklist. |
| 8. Next Kernel | 14+ | Repeat for the L2 kernel. Pipeline familiarity from L1 means L2 takes ~5-6 days vs ~7-8 for L1. |

---

## Hardware and Compute

**Target hardware:** NVIDIA A100 80GB or H100 80GB (1 GPU per subteam)

**Estimated compute budget:**
- 3 subteams, 3 hours/day, 30-day sprint
- A100 at ~$1.49/hr (Lambda Labs / JarvisLabs): **~$402/month**
- H100 at ~$1.75/hr (SF Compute): **~$472/month** (2-3x faster, recommended)

**Target model for testing:** Qwen3-8B (or smaller Qwen3 variant for fast iteration)

**Qwen3-8B reference dimensions:**
- Hidden size: 4096
- Intermediate size: 11008
- Typical training shape: batch=4, seq_len=2048, dtype=bf16

---

## Self-Study Curriculum

The team follows a structured **38-prompt, 9-module curriculum** (~48-59 hours total) stored in `forge_learning/`. Each prompt is a standalone learning session. Modules:

| Module | Topic | Hours | Outcome |
|--------|-------|-------|---------|
| 0 | GPU Foundations (SIMT, memory hierarchy, roofline) | 3-4 | Mental model of why kernel optimization matters |
| 1 | Triton Basics (programs, blocks, grids, vector add through tiled matmul) | 6-8 | Can write simple Triton kernels |
| 2 | Custom Autograd & Testing (autograd internals, gradcheck, benchmarking) | 4-5 | Can wrap kernels in autograd.Function with tests |
| 2a | Autograd Deep Dive (hidden costs, Unsloth's layer-level approach) | 3-4 | Understands when to bypass PyTorch's autograd engine |
| 3 | Transformer Internals (attention, RMSNorm, SwiGLU, RoPE in Qwen3) | 4-5 | Full mental model of every op in Qwen3 forward pass |
| 4 | Kernel Math (hand-derive gradients for RMSNorm, SwiGLU, RoPE, online softmax, fused CE, LoRA) | 8-10 | Can derive backward pass for any kernel |
| 5 | Kernel Implementation (Triton fwd+bwd for all kernels) | 8-10 | Working Triton kernels for RMSNorm, SwiGLU, RoPE, CE, LoRA |
| 6 | Training Pipeline (HF internals, model patching, sample packing, gradient checkpointing) | 4-5 | SFT trainer with packing and gradient checkpointing |
| 7 | RL and GRPO (policy gradients, GRPO algorithm, memory-efficient log-softmax) | 3-4 | GRPO trainer with chunked log-softmax |
| 8 | Quantization & Export (NF4/INT8, GGUF, llama.cpp) | 2-3 | QLoRA loading + GGUF export |

**Role-based paths:**
- Kernel engineers: Modules 0, 1, 2, 3, 4, 5 (required), 6, 7 (optional)
- Training pipeline engineers: Modules 0, 3, 6, 7 (required), 1, 2 (recommended)
- Infrastructure / patching: Modules 0, 3, 6, 8 (required)
- Testing / benchmarking: Modules 0, 1, 2 (required), 4, 5 (optional)

---

## Backlog: Kernels After CP1

Once the first sprint's 6 kernels are done, the following are queued:

1. LoRA MLP fused kernel
2. LoRA QKV fused kernel
3. Manual autograd (bypass torch.autograd entirely)
4. KL Divergence kernel
5. DPO fused loss kernel
6. ORPO fused loss kernel
7. CPO fused loss kernel
8. int64 indexing for 500K+ context support

---

## Key Technical Decisions

- **Triton over CUDA**: Triton provides a higher-level abstraction that compiles to efficient GPU code. All competitors (Unsloth, Liger) use Triton for their custom kernels. Triton kernels are portable across NVIDIA GPU generations and (increasingly) AMD.
- **Per-kernel autograd vs layer-level autograd**: Unsloth uses layer-level autograd (one monolithic backward for an entire transformer layer). Forge starts with per-kernel autograd (each kernel has its own backward) for modularity, with manual/layer-level autograd as a later optimization.
- **fp32 accumulation**: All reductions accumulate in fp32 to guarantee 0% accuracy loss, even when inputs are bf16/fp16.
- **Kernel registry**: Unique to Forge. Enables plugging in new kernel variants and automatically A/B testing them against baselines, which is critical for a research team iterating on kernel designs.
- **HuggingFace patching**: Forge monkey-patches HF Transformer models at runtime via `forge.patch(model)`. This means users don't need to modify model code or use a custom model class. Patches are reversible via `forge.unpatch(model)`.
- **Target: torch.compile compatibility**: Liger is fully torch.compile compatible. Forge targets this but has only partial support at CP1.

---

## Evidence Sources for Competitive Claims

All competitive claims in this document are sourced from:

- **Unsloth**: GitHub README (github.com/unslothai/unsloth), pricing page (unsloth.ai/pricing), documentation (unsloth.ai/docs), and the GRPO long-context technical blog
- **Liger-Kernel**: Tech report (arXiv 2410.10989), GitHub README, and source code analysis
- **Torchtune**: GitHub README (PyTorch official project)
- **Axolotl**: GitHub repo and Spheron 2026 comparison benchmarks
- **LLaMA-Factory**: GitHub repo and documentation
- **Benchmark reference**: Unsloth 3.2hrs vs Axolotl 5.8hrs vs Torchtune 4.7hrs for Llama-3.1-8B QLoRA on A100 (Spheron 2026)

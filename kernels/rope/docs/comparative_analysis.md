# RoPE Fused Q+K — Comparative Analysis & Design

**Owner:** Shaurya (H2 — Forge Hackathon Day 1)
**Status:** Draft (to be reviewed before implementation)
**Date:** 2026-05-23
**Ambition:** Match Unsloth-equivalent perf with a cleaner modular API. Forge's edge is the test/bench/registry infra around the kernel, not micro-architecture novelty.
**Sources analyzed:** 6 (HF Qwen3, Unsloth, Liger, Megatron-LM, TransformerEngine, TorchTitan). Forge internal curriculum pending (see `rope_knowledge_base/06_forge_internal/`).

---

## 0. Correction to the hackathon prior

The hackathon doc (`day1.html`, H2 task card) states:

> *"Unsloth is the reference, not Liger. Liger applies RoPE to Q and K separately (two launches). Unsloth fuses them."*

**This is backwards in the default code paths.** Reading the actual fetched sources:

- **Liger's `_triton_rope`** is fused: grid `(batch * seq_len,)`, one program handles BOTH Q (across all Q heads) AND K (across all KV heads) for one token. Always single-launch.
- **Unsloth's default `fast_rope_embedding`** (no `rope_embedding_indices` argument) calls `Fast_RoPE_Embedding.apply()` twice — once for Q, once for K — i.e. **two separate launches**. The fused `Fast_RoPE_Embedding_QK` kernel only runs when TRL-style rope indices are passed in.

**Implication for our design:** Liger is the closer template for our HF-cos/sin training path. We still borrow Unsloth's clever GQA trick from `Fast_RoPE_Embedding_QK` (`if head_position < n_heads_K`), but the grid + fusion strategy is Liger-shaped.

---

## 1. Math Contract (Forge RoPE)

**Equation (forward, HF split-half convention):**
```
half = head_dim // 2
For each (b, s, h, d) where d ∈ [0, half):
  y[b, h, s, d]        = x[b, h, s, d]        * cos[s, d] - x[b, h, s, d + half] * sin[s, d]
  y[b, h, s, d + half] = x[b, h, s, d + half] * cos[s, d] + x[b, h, s, d]        * sin[s, d]
```

We rely on HF's convention that `cos`/`sin` are full `head_dim` wide with `cos[s, d + half] == cos[s, d]` (because `emb = cat((freqs, freqs), dim=-1)`). Our kernel only loads the first `half` columns.

**Equation (backward):**
For the loss gradient `dy`, the input gradient `dx` is:
```
dx[b, h, s, d]        = dy[b, h, s, d]        * cos[s, d] + dy[b, h, s, d + half] * sin[s, d]
dx[b, h, s, d + half] = dy[b, h, s, d + half] * cos[s, d] - dy[b, h, s, d]        * sin[s, d]
```

This is structurally identical to the forward with `sin → -sin`. We exploit this: backward calls the same Triton kernel with a `BACKWARD_PASS=True` constexpr that flips the sign of `sin` after the load. **Saves an entire kernel file.** Both Liger and Unsloth use this trick; TE does NOT (TE uses a dedicated bwd kernel because it works on raw `freqs` and recomputes `sincosf` per call, but for our cos/sin layout the negate-sin trick is exact).

**Dtype contract:**
- Inputs Q, K: bf16 or fp16 (training dtype)
- cos, sin: fp32 (HF computes these in fp32, casts at the end — we accept fp32 or input dtype)
- Internal accumulation: **fp32** (cast bf16/fp16 loads up to fp32 via `.to(tl.float32)`)
- Output Q', K': same dtype as input (cast down on store)

**Shape contract:**
- `q.shape == (batch, n_q_heads, seq_len, head_dim)` — HF Qwen3 convention (n_heads dim is axis 1)
- `k.shape == (batch, n_kv_heads, seq_len, head_dim)` — n_kv_heads ≤ n_q_heads (GQA)
- `cos.shape == sin.shape == (batch, seq_len, head_dim)` OR `(1, seq_len, head_dim)` (broadcastable batch). HF passes the latter after `unsqueeze(unsqueeze_dim=1)` — our kernel handles both via `cos_batch_size`.
- `head_dim` must be a power of 2 (Qwen3: 128). Partial RoPE (`rotary_dim < head_dim`) is **deferred** — not in Qwen3, defer to a v2 if needed.

---

## 2. Decision Matrix

Each row: how the 6 sources handle a dimension, plus **Forge's choice + one-line rationale**.

### 2.1 Grid shape and fusion strategy

| Source | Approach |
|---|---|
| HF | Pure PyTorch — N/A |
| **Unsloth `Fast_RoPE_Embedding`** (default) | Per-tensor: grid `(b·s, n_groups)` where `n_groups = ceil(n_heads / 4)`. Processes 4 heads per program (ROPE_GROUP_SIZE). Q and K in separate launches. |
| **Unsloth `Fast_RoPE_Embedding_QK`** (TRL indices only) | Fused: grid `(b·s, n_heads_Q)`. Each program does 1 Q head + (if `head_pos < n_heads_K`) 1 K head. |
| **Liger** | Fused: grid `(b·s,)`. Each program does ALL Q heads + ALL K heads for one token. Tile = `(pad_n_qh, hd/2)` and `(pad_n_kh, hd/2)`. |
| Megatron / TE | Per-tensor (called twice). CUDA grid `(s, b)`. Inside block: `(blockDim.x, blockDim.y)` tile across `(d2, h)`. |
| TorchTitan | Pure PyTorch — N/A |

**Forge choice: grid `(batch · seq_len, n_heads_Q)`** with Unsloth-QK's GQA mask trick.

**Why:**
- Liger's `(b·s,)` 1-D grid launches just `4 × 2048 = 8192` programs at Qwen3 demo shapes. With Qwen2.5-0.5B (n_q=14, n_kv=2, head_dim=64) the per-program tile is `(16, 32)` which is fine for registers — but for the larger Qwen3-8B (n_q=32, n_kv=8, head_dim=128) the tile becomes `(32, 64)` for Q alone, which starts pressuring registers. We want headroom.
- Unsloth-QK's `(b·s, n_qh)` grid launches `8192 × 14 = 114,688` programs on demo shape, `8192 × 32 = 262,144` on Qwen3-8B. Per-program tile is just `(hd/2,)` — small, register-friendly, more SMs saturated.
- The GQA mask trick (`if head_pos < n_heads_K`) keeps the K work inside the same launch without inflating grid dimensions.
- **Cost:** cos/sin are loaded redundantly across programs of the same token (once per program instead of once per token). The redundancy is `n_heads_Q × head_dim/2 × 4 bytes` per token = trivial compared to Q traffic.

### 2.2 cos/sin layout and source

| Source | Approach |
|---|---|
| HF | Full `head_dim` cos/sin (left half == right half via `cat((freqs, freqs))`). |
| Unsloth (both kernels) | Full `head_dim`, loads only first half. |
| Liger | Full `head_dim`, loads only first half. |
| TE | Takes raw `freqs` (shape `(s, 1, 1, d/2)`), computes `sincosf` inside the kernel via shared memory. |
| TorchTitan | Both `complex` (precomputed `cis` complex tensor) and `cos_sin` paths supported. |

**Forge choice: accept HF-format full `head_dim` cos/sin, load only the first half.**

**Why:** Zero-friction `forge.patch` integration — we receive exactly what `Qwen3RotaryEmbedding.forward()` returns. The wasted `head_dim/2` of cos/sin memory is `(seq_len × head_dim/2 × 2B) ≈ 256KB` at demo shape — negligible. Not worth the API friction of slicing.

**Not choosing TE's in-kernel sincos:** We need fp32 accuracy without CUDA intrinsics. Triton's `tl.cos`/`tl.sin` exist but pre-computing once per forward is simpler and the cos/sin are reusable across attention layers within the same model (HF caches them on the rotary module).

### 2.3 cos/sin caching strategy

| Source | Approach |
|---|---|
| HF | Recomputed every forward call of `Qwen3RotaryEmbedding` (inside `@torch.no_grad()`). |
| Unsloth | Pre-computed by HF, passed in. Saved on `ctx` (not via `save_for_backward`). |
| Liger | Pre-computed, passed in. `ctx.save_for_backward(cos, sin)`. |
| TE | Pre-computed `freqs` passed in. `ctx.save_for_backward(freqs, ...)`. |
| Megatron | Pre-computed `freqs` passed in. |
| TorchTitan | Module-level `self.cache` precomputed once at `__init__`. |

**Forge choice: caller-precomputed cos/sin, saved via `ctx.save_for_backward(cos, sin)`.**

**Why:** Compatible with HF's `Qwen3RotaryEmbedding` which already caches `inv_freq` and recomputes cos/sin per call (cheap — `seq_len × head_dim/2` mul + sincos). `save_for_backward` is the right idiom for backward dependencies (handles FSDP2 correctly; that's literally the smoke test we need to pass). Unsloth's `ctx.cos = cos` approach is **wrong under FSDP2** and would break the H15 smoke test.

### 2.4 Rotation layout (split-half vs interleaved vs complex)

| Source | Approach |
|---|---|
| HF | Split-half (`rotate_half(x) = cat((-x2, x1))`). |
| Unsloth | Split-half. |
| Liger | Split-half. |
| TE | Both via `interleaved: bool`. |
| Megatron | Both via `rotary_interleaved: bool`. |
| TorchTitan | Complex (Llama3/4) and cos/sin split-half (Qwen3). |

**Forge choice: split-half only (HF convention).**

**Why:** Qwen3 uses split-half. Gemma uses split-half. Adding `interleaved` is dead weight for our CP1 target. If a future model needs interleaved, we add a `tl.constexpr` branch then.

### 2.5 GQA handling (n_q_heads ≠ n_kv_heads)

| Source | Approach |
|---|---|
| HF | Each tensor handled independently (`q` and `k` separately in apply). |
| Unsloth-QK | `(b·s, n_heads_Q)` grid + `if head_position < n_heads_K` mask. |
| Liger | Loads `(pad_n_qh, hd/2)` and `(pad_n_kh, hd/2)` 2D tiles separately, mask via `tl.arange(...) < n_qh`, `... < n_kh`. |
| TE / Megatron | Per-tensor — GQA handled by caller calling kernel twice with different shapes. |
| TorchTitan | Per-tensor apply, called twice. |

**Forge choice: Unsloth-QK's mask trick — `if head_position < n_heads_K: do K work`.**

**Why:** Simplest, no padded 2D tiles, scales to any n_q_heads / n_kv_heads ratio without special cases. We already chose Unsloth-QK's grid in §2.1 — this falls out of that choice.

### 2.6 BLOCK_SIZE strategy

| Source | Approach |
|---|---|
| Unsloth-QK | `calculate_settings(head_dim)` — picks BLOCK_SIZE based on dim. |
| Unsloth default | `calculate_settings(head_dim // 2)`. |
| Liger | `max(pad_n_qh, pad_n_kvh)` — block covers head dim, not column dim. |
| TE | CUDA `blockDim` configured externally. |
| Megatron | N/A (dispatcher). |

**Forge choice: `BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)` with `num_warps` chosen by a small autotune (`[2, 4, 8]`).**

**Why:** Our grid is `(b·s, n_qh)` — per-program work is `head_dim/2` columns. For Qwen3 `head_dim=128`, BLOCK_SIZE=64. For arbitrary head_dim (Gemma is 256, some smaller models are 64), pow-2 padding handles it cleanly. Autotune `num_warps` only — `BLOCK_SIZE` is fixed by head_dim so no tuning there.

### 2.7 Backward strategy

| Source | Approach |
|---|---|
| Unsloth | Same kernel with `BACKWARD_PASS=True` constexpr, sin negated. |
| Liger | Same kernel with `BACKWARD_PASS=True` constexpr, sign flips on the multiply ops. |
| TE | Dedicated `fused_rope_backward_kernel` — needed because TE works with raw `freqs` (not cos/sin), so the partner-position sin lookup differs in backward. |
| Megatron | N/A. |
| TorchTitan | N/A (PyTorch autograd). |

**Forge choice: shared kernel with `BACKWARD_PASS: tl.constexpr`, sin negated after load.**

**Why:** Math is exact for the HF cos/sin convention (we verified in §1). Halves kernel-code surface. Constexpr-driven branch — zero runtime cost (Triton specializes).

### 2.8 Numerical precision / fp32 accumulation

| Source | Approach |
|---|---|
| HF | Computes cos/sin in fp32 inside `@torch.no_grad()`, casts to input dtype before apply. The apply itself runs in input dtype. |
| Unsloth-QK | bf16/fp16 throughout — relies on Triton's internal upcasting for mul/add. **Risk:** subtle precision loss vs HF. |
| Unsloth default | `.to(sin1.dtype)` — casts Q to sin's dtype (typically fp32 since cos/sin are fp32 in HF). |
| Liger | `.to(sin_row.dtype)` — same pattern as Unsloth default. |
| TE | Explicit `float v_src = src[...]`, computes in float, casts back on store. |

**Forge choice: explicit fp32 accumulation — `q0 = tl.load(...).to(tl.float32)`, compute in fp32, cast back to input dtype on store.**

**Why:** Liger's `.to(sin_row.dtype)` only works if cos/sin are fp32 (which HF guarantees, but not by contract). Explicit `.to(tl.float32)` is unambiguous and guarantees rtol=1e-5 against HF reference. This is also what FORGE_CONTEXT.md mandates ("fp32 accumulation for all kernels").

### 2.9 `base` parameterization and scaling extensibility

| Source | Approach |
|---|---|
| HF | `config.rope_parameters["rope_theta"]` (base) + `ROPE_INIT_FUNCTIONS[rope_type]` for scaled variants (YaRN, NTK, Llama3). |
| Unsloth / Liger | Take cos/sin as inputs — base handling is the caller's problem. |
| TE | `rotary_base: float = 10000.0` constructor arg; only basic linear interpolation supported in module. |
| TorchTitan | Config-driven: `theta`, `scaling: "none"|"llama"|"yarn"`, full YaRN params on the Config dataclass. |

**Forge choice: kernel takes cos/sin only — base + scaling lives on the `ForgeRoPE(nn.Module)` constructor.**

```python
class ForgeRoPE(nn.Module):
    def __init__(self, head_dim, base=10000.0, max_seq_len=2048,
                 rotary_percent=1.0, scaling=None):
        # precompute cos/sin once; expose .forward(q, k, position_ids=None)
```

**Why:**
- Keeps the kernel simple — no compile-time `scaling` constexpr.
- API matches HF: Gemma just constructs with `base=<gemma_value>` (P8's "trivial 20-min change" per the hackathon plan).
- YaRN/Llama-scaling — when needed (CP3+) — slots in by changing the precompute, not the kernel. TorchTitan's design proves this layering is clean.

### 2.10 In-place vs out-of-place

| Source | Approach |
|---|---|
| HF | Out-of-place (returns new tensors). |
| Unsloth-QK | In-place if contiguous, clone if not. |
| Unsloth default | In-place. |
| Liger | Out-of-place via `.transpose().contiguous()` (transposes for memory layout, returns new). |
| TE | Out-of-place. |

**Forge choice: out-of-place by default, optional in-place via a flag (defer to v2).**

**Why:** In-place is faster (no allocator hit, half the HBM writes) but breaks autograd ordering if anything else aliases Q/K. Out-of-place is safer for the hackathon — we can prove correctness without fighting alias issues. In-place is a v2 speedup once we have benchmarks showing it's worth the risk. Note: Unsloth's "in-place if contiguous else clone" is a reasonable compromise we may adopt later.

---

## 3. Pseudocode

### 3.1 Forward + backward kernel (shared)

```python
@triton.jit
def _forge_rope_kernel(
    Q_ptr, Q_batch_stride, Q_head_stride, Q_seq_stride,
    K_ptr, K_batch_stride, K_head_stride, K_seq_stride,
    OutQ_ptr, OutQ_batch_stride, OutQ_head_stride, OutQ_seq_stride,
    OutK_ptr, OutK_batch_stride, OutK_head_stride, OutK_seq_stride,
    cos_ptr, cos_batch_stride, cos_seq_stride,
    sin_ptr, sin_batch_stride, sin_seq_stride,
    seq_len, n_heads_K: tl.constexpr,
    head_dim: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,   # head_dim // 2
    COS_HAS_BATCH: tl.constexpr,   # True if cos.shape[0] == batch (vs 1)
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,      # next_pow2(head_dim // 2)
):
    # ---- Grid: (batch * seq_len, n_heads_Q) ----
    row_pos = tl.program_id(0)           # 0 .. batch*seq_len - 1
    head_pos = tl.program_id(1)          # 0 .. n_heads_Q - 1

    batch_id = row_pos // seq_len
    seq_id   = row_pos % seq_len

    # ---- Load cos/sin (head_dim/2 cols, broadcast across columns < HALF_HEAD_DIM) ----
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask    = col_offsets < HALF_HEAD_DIM

    cos_row_ptr = cos_ptr \
        + (batch_id * cos_batch_stride if COS_HAS_BATCH else 0) \
        + seq_id * cos_seq_stride
    sin_row_ptr = sin_ptr \
        + (batch_id * sin_batch_stride if COS_HAS_BATCH else 0) \
        + seq_id * sin_seq_stride

    cos_row = tl.load(cos_row_ptr + col_offsets, mask=col_mask, other=0).to(tl.float32)
    sin_row = tl.load(sin_row_ptr + col_offsets, mask=col_mask, other=0).to(tl.float32)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # ---- Process Q head ----
    q_row_ptr = Q_ptr \
        + batch_id * Q_batch_stride \
        + head_pos * Q_head_stride \
        + seq_id   * Q_seq_stride
    q_lo = tl.load(q_row_ptr + col_offsets,                  mask=col_mask, other=0).to(tl.float32)
    q_hi = tl.load(q_row_ptr + col_offsets + HALF_HEAD_DIM,  mask=col_mask, other=0).to(tl.float32)

    out_q_lo = q_lo * cos_row - q_hi * sin_row
    out_q_hi = q_hi * cos_row + q_lo * sin_row

    out_q_row_ptr = OutQ_ptr \
        + batch_id * OutQ_batch_stride \
        + head_pos * OutQ_head_stride \
        + seq_id   * OutQ_seq_stride
    tl.store(out_q_row_ptr + col_offsets,                  out_q_lo, mask=col_mask)
    tl.store(out_q_row_ptr + col_offsets + HALF_HEAD_DIM,  out_q_hi, mask=col_mask)

    # ---- Process K head if this program's head_pos maps to a valid K head ----
    if head_pos < n_heads_K:
        k_row_ptr = K_ptr \
            + batch_id * K_batch_stride \
            + head_pos * K_head_stride \
            + seq_id   * K_seq_stride
        k_lo = tl.load(k_row_ptr + col_offsets,                  mask=col_mask, other=0).to(tl.float32)
        k_hi = tl.load(k_row_ptr + col_offsets + HALF_HEAD_DIM,  mask=col_mask, other=0).to(tl.float32)

        out_k_lo = k_lo * cos_row - k_hi * sin_row
        out_k_hi = k_hi * cos_row + k_lo * sin_row

        out_k_row_ptr = OutK_ptr \
            + batch_id * OutK_batch_stride \
            + head_pos * OutK_head_stride \
            + seq_id   * OutK_seq_stride
        tl.store(out_k_row_ptr + col_offsets,                  out_k_lo, mask=col_mask)
        tl.store(out_k_row_ptr + col_offsets + HALF_HEAD_DIM,  out_k_hi, mask=col_mask)
```

**Note on GQA mask correctness:** Unsloth-QK uses `if head_position < n_heads_K` (a compile-time-friendly Python `if` since `head_pos` is a `tl.program_id` scalar). Triton handles this as a thread-group branch — no warp divergence within a program. This is the right pattern.

### 3.2 autograd.Function wrapper

```python
class ForgeRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # q: (b, n_q, s, d), k: (b, n_kv, s, d), cos/sin: (b_or_1, s, d)
        b, n_q, s, d = q.shape
        n_kv = k.shape[1]
        half_d = d // 2

        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

        cos_has_batch = (cos.shape[0] == b)
        BLOCK_SIZE = triton.next_power_of_2(half_d)

        grid = (b * s, n_q)
        _forge_rope_kernel[grid](
            q, q.stride(0), q.stride(1), q.stride(2),
            k, k.stride(0), k.stride(1), k.stride(2),
            out_q, out_q.stride(0), out_q.stride(1), out_q.stride(2),
            out_k, out_k.stride(0), out_k.stride(1), out_k.stride(2),
            cos, cos.stride(0) if cos_has_batch else 0, cos.stride(-2),
            sin, sin.stride(0) if cos_has_batch else 0, sin.stride(-2),
            seq_len=s, n_heads_K=n_kv,
            head_dim=d, HALF_HEAD_DIM=half_d,
            COS_HAS_BATCH=cos_has_batch,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # autotune later
        )
        ctx.save_for_backward(cos, sin)
        ctx.cos_has_batch = cos_has_batch
        return out_q, out_k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        b, n_q, s, d = dq.shape
        n_kv = dk.shape[1]
        half_d = d // 2

        dq_in = torch.empty_like(dq)
        dk_in = torch.empty_like(dk)
        BLOCK_SIZE = triton.next_power_of_2(half_d)

        grid = (b * s, n_q)
        _forge_rope_kernel[grid](
            dq, dq.stride(0), dq.stride(1), dq.stride(2),
            dk, dk.stride(0), dk.stride(1), dk.stride(2),
            dq_in, dq_in.stride(0), dq_in.stride(1), dq_in.stride(2),
            dk_in, dk_in.stride(0), dk_in.stride(1), dk_in.stride(2),
            cos, cos.stride(0) if ctx.cos_has_batch else 0, cos.stride(-2),
            sin, sin.stride(0) if ctx.cos_has_batch else 0, sin.stride(-2),
            seq_len=s, n_heads_K=n_kv,
            head_dim=d, HALF_HEAD_DIM=half_d,
            COS_HAS_BATCH=ctx.cos_has_batch,
            BACKWARD_PASS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return dq_in, dk_in, None, None
```

### 3.3 nn.Module wrapper

```python
class ForgeRoPE(nn.Module):
    """Drop-in replacement for HF Qwen3RotaryEmbedding's apply step."""
    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_seq_len: int = 2048, rotary_percent: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * rotary_percent)
        assert self.rotary_dim == head_dim, "Partial RoPE deferred — v2"
        self.base = base
        # cos/sin precompute lives in the HF rotary embedding module that calls us.
        # ForgeRoPE is the *apply* step only — matches HF's split.

    def forward(self, q, k, cos, sin):
        return ForgeRoPEFunction.apply(q, k, cos, sin)
```

Per the H8 patch wiring: `forge.patch(model)` replaces `apply_rotary_pos_emb` in `modeling_qwen3.py` with `ForgeRoPE.forward` (after instantiation). The `Qwen3RotaryEmbedding` module that produces cos/sin stays as-is — we plug in only at the apply step. (For Gemma the patch passes a different `base` to `Qwen3RotaryEmbedding`-equivalent; that's P8's wiring concern, not ours.)

---

## 4. Test Contract

Lives in `kernels/rope/tests/test_rope.py`. Three levels per the Forge kernel contract:

### 4.1 Gradcheck (fp64)
- Shape: `(b=2, n_q=4, s=8, d=64)`, `n_kv=2` (small for fp64)
- `torch.autograd.gradcheck(ForgeRoPEFunction.apply, (q, k, cos, sin), eps=1e-6, atol=1e-5)`
- Cast everything to fp64 first; kernel must support it (`tl.float64` accumulation just works).

### 4.2 Forward + backward correctness vs HF reference
- Reference: `kernels/rope/rope_knowledge_base/03_hf_transformers/modeling_qwen3_rope.py::apply_rotary_pos_emb`
- Dtypes: `[torch.bfloat16, torch.float16, torch.float32]`
- Shape sweep:
  | Shape | Notes |
  |---|---|
  | `(b=1, n_q=14, n_kv=2, s=512, d=64)` | Qwen2.5-0.5B demo shape |
  | `(b=4, n_q=32, n_kv=8, s=2048, d=128)` | Qwen3-8B target |
  | `(b=1, n_q=32, n_kv=8, s=8192, d=128)` | Long-context |
  | `(b=2, n_q=16, n_kv=16, s=1024, d=128)` | No GQA (n_q == n_kv) |
  | `(b=2, n_q=8, n_kv=1, s=1024, d=128)` | MQA edge case (n_kv == 1) |
- Tolerances:
  - bf16: `rtol=1e-3, atol=1e-3`
  - fp16: `rtol=1e-3, atol=1e-3`
  - fp32: `rtol=1e-5, atol=1e-5`
- For backward: feed random `dq, dk`, compare `dq_in, dk_in` against HF + `torch.autograd.grad`.

### 4.3 Patched-vs-unpatched output equivalence (integration prep for H10)
- Build `Qwen3RotaryEmbedding` from HF.
- Compare `apply_rotary_pos_emb(q, k, cos, sin)` (HF) vs `ForgeRoPE()(q, k, cos, sin)`.
- Must match exactly (same tolerances as 4.2).

### 4.4 Edge cases
- Non-contiguous Q/K (transposed input) — must not crash, output correct.
- cos/sin with `shape[0] == 1` (broadcast batch) and `shape[0] == batch`.
- `n_kv == n_q` (no GQA).
- `n_kv == 1` (MQA).

---

## 5. Benchmark Contract

Lives in `kernels/rope/benchmarks/bench_rope.py`. Single benchmark harness.

**Compared against:**
1. **PyTorch reference** (HF `apply_rotary_pos_emb`) — baseline.
2. **Liger `liger_rotary_pos_emb`** — our closest peer (single-launch fused).
3. **Unsloth `fast_rope_embedding`** — the hackathon's stated reference.

**Metrics:**
- Forward latency (µs)
- Backward latency (µs)
- Fwd+bwd combined (training-realistic)
- Peak VRAM during fwd+bwd

**Shape sweep (matches test sweep):** Qwen2.5-0.5B, Qwen3-8B, long-context, no-GQA, MQA.

**Done gate (from H2 task card):**
- Hard floor: ≥1.0× PyTorch (no regression)
- Target: ≥1.3× PyTorch on Qwen3-8B shape
- Stretch: ≥ Unsloth `Fast_RoPE_Embedding_QK` on the GQA case (where we apply the same trick)

**HBM bandwidth sanity check (Qwen3-8B, b=4, s=2048, n_q=32, n_kv=8, d=128, bf16):**
- Q traffic: 4·32·2048·128·2B = 64MB read + 64MB write = 128MB
- K traffic: 4·8·2048·128·2B = 16MB read + 16MB write = 32MB
- cos/sin: 2 × 4·2048·128·2B = 4MB
- Total ≈ 164MB
- H100 HBM ≈ 3 TB/s → floor ≈ 55µs
- Anything > 110µs means we're losing ≥50% of bandwidth — investigate.

---

## 6. Open questions / risks before implementation

1. **GQA mask trick under autotune.** Unsloth-QK's `if head_pos < n_heads_K` works because `head_pos` is a `program_id` scalar. If we autotune over `num_warps` only, this should be safe — but worth verifying the generated PTX doesn't introduce divergence.

2. **In-place vs out-of-place.** Plan keeps out-of-place. If benchmarks come back at <1.3× target, the easiest win is allocating output to the same buffer as input (in-place). Defer this decision to post-benchmark.

3. **Partial RoPE (`rotary_percent < 1.0`).** Deferred. Qwen3 doesn't need it. If H8/Gemma wiring surfaces a need, we add `ROTARY_DIM: tl.constexpr` and a pass-through tail branch — TE's CUDA kernel shows the pattern (lines 59-69 of `fused_rope.cu`).

4. **Non-contiguous input handling.** Plan accepts arbitrary strides via per-tensor stride args (like Unsloth-QK). May want to add a fast-path `.contiguous()` for the common case if benchmarks show non-contiguous strides hurt — Liger does this (`q = q.contiguous()`).

5. **torch.compile compatibility.** Liger is fully compatible; Unsloth disables it (`@torch.compiler.disable`). We're a thin autograd.Function — should be compile-compatible by default. Sanity-check in tests.

6. **FSDP2 readiness (H15 smoke test).** Our kernel doesn't hold model weights — only cos/sin (per-call buffers, not parameters). `ctx.save_for_backward(cos, sin)` is the right idiom and survives FSDP2 sharding. We're fine on the smoke test as long as we don't store cos/sin as module parameters.

---

## 7. Implementation order

1. `kernels/rope/rope_v1.py` — port the pseudocode in §3 to working Triton. Forward only.
2. `tests/test_rope.py::test_forward_correctness` — verify against HF on the demo shape.
3. Add backward branch in the kernel.
4. `tests/test_rope.py::test_gradcheck` — fp64 verification.
5. `tests/test_rope.py::test_backward_correctness` — bf16/fp16 sweep.
6. `benchmarks/bench_rope.py` — measure vs PyTorch + Liger + Unsloth.
7. Wrap in `ForgeRoPE(nn.Module)` and register in the kernel registry (H7's scaffold).
8. Hand off to P8 for `forge.patch(Qwen3)` wiring (H10 Day 2 morning).

Target wall-clock: ~3 hrs for steps 1-7 (leaving 1-2 hrs of the 4-5 hr budget for debug + autotune polish).

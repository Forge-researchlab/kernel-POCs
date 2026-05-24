"""ForgeRoPE V1 — fused Q+K RoPE Triton kernel.

Design summary (see kernels/rope/docs/comparative_analysis.md for the full doc):
- Grid:           (batch * seq_len, n_q_heads)
- Layout:         HF split-half — y_lo = x_lo*cos - x_hi*sin ; y_hi = x_hi*cos + x_lo*sin
- GQA handling:   `if head_pos < n_kv` — Unsloth-QK's mask trick
- Accumulation:   explicit fp32 upcast on load, cast back to input dtype on store
- Backward:       same kernel with BACKWARD_PASS=True constexpr; sin negated after load
- cos/sin source: caller-precomputed (HF Qwen3RotaryEmbedding format, full head_dim)
- Saved for bwd:  ctx.save_for_backward(cos, sin)  -- FSDP2-safe
- Out-of-place:   forward and backward both allocate fresh output tensors

Not yet in V1 (will land in V2/V3):
- GQA-aligned head grouping (G = n_q // n_kv) for better cos/sin reuse
- @triton.autotune over num_warps and G
- Optional in-place mode

Shape contract:
- q:   (batch, n_q_heads,  seq_len, head_dim)
- k:   (batch, n_kv_heads, seq_len, head_dim)         n_kv_heads <= n_q_heads
- cos: (1, seq_len, head_dim) or (batch, seq_len, head_dim)
- sin: same as cos

head_dim must be a power of 2 (Qwen3: 128).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forge_rope_v1_kernel(
    Q_ptr, Q_batch_stride, Q_head_stride, Q_seq_stride,
    K_ptr, K_batch_stride, K_head_stride, K_seq_stride,
    OutQ_ptr, OutQ_batch_stride, OutQ_head_stride, OutQ_seq_stride,
    OutK_ptr, OutK_batch_stride, OutK_head_stride, OutK_seq_stride,
    cos_ptr, cos_batch_stride, cos_seq_stride,
    sin_ptr, sin_batch_stride, sin_seq_stride,
    seq_len,
    n_heads_K: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program handles RoPE for one (batch, seq) token at one Q-head index.
    If head_pos < n_heads_K, the same program also handles the matching K head.
    """
    # ---- Grid coordinates ----
    row_pos = tl.program_id(0).to(tl.int64)         # 0 .. batch*seq_len - 1
    head_pos = tl.program_id(1).to(tl.int64)        # 0 .. n_q_heads - 1

    batch_id = row_pos // seq_len
    seq_id   = row_pos % seq_len

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask    = col_offsets < HALF_HEAD_DIM

    # ---- Load cos/sin row (head_dim/2 cols) ----
    # cos_batch_stride is 0 when cos.shape[0] == 1 (broadcast)
    cos_row_ptr = cos_ptr + batch_id * cos_batch_stride + seq_id * cos_seq_stride
    sin_row_ptr = sin_ptr + batch_id * sin_batch_stride + seq_id * sin_seq_stride

    cos_row = tl.load(cos_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    sin_row = tl.load(sin_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # ---- Process Q head ----
    q_row_ptr = Q_ptr + batch_id * Q_batch_stride + head_pos * Q_head_stride + seq_id * Q_seq_stride

    q_lo = tl.load(q_row_ptr + col_offsets,                  mask=col_mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_row_ptr + col_offsets + HALF_HEAD_DIM,  mask=col_mask, other=0.0).to(tl.float32)

    out_q_lo = q_lo * cos_row - q_hi * sin_row
    out_q_hi = q_hi * cos_row + q_lo * sin_row

    out_q_row_ptr = OutQ_ptr + batch_id * OutQ_batch_stride + head_pos * OutQ_head_stride + seq_id * OutQ_seq_stride
    tl.store(out_q_row_ptr + col_offsets,                  out_q_lo, mask=col_mask)
    tl.store(out_q_row_ptr + col_offsets + HALF_HEAD_DIM,  out_q_hi, mask=col_mask)

    # ---- Process K head if this Q-head index also maps to a valid K head ----
    if head_pos < n_heads_K:
        k_row_ptr = K_ptr + batch_id * K_batch_stride + head_pos * K_head_stride + seq_id * K_seq_stride

        k_lo = tl.load(k_row_ptr + col_offsets,                  mask=col_mask, other=0.0).to(tl.float32)
        k_hi = tl.load(k_row_ptr + col_offsets + HALF_HEAD_DIM,  mask=col_mask, other=0.0).to(tl.float32)

        out_k_lo = k_lo * cos_row - k_hi * sin_row
        out_k_hi = k_hi * cos_row + k_lo * sin_row

        out_k_row_ptr = OutK_ptr + batch_id * OutK_batch_stride + head_pos * OutK_head_stride + seq_id * OutK_seq_stride
        tl.store(out_k_row_ptr + col_offsets,                  out_k_lo, mask=col_mask)
        tl.store(out_k_row_ptr + col_offsets + HALF_HEAD_DIM,  out_k_hi, mask=col_mask)


def _launch_kernel(q_in, k_in, q_out, k_out, cos, sin, backward_pass: bool):
    """Internal: launch the Triton kernel. Assumes shapes are validated."""
    batch, n_q, seq_len, head_dim = q_in.shape
    n_kv = k_in.shape[1]
    half_head_dim = head_dim // 2

    cos_has_batch = (cos.shape[0] == batch)
    cos_batch_stride = cos.stride(0) if cos_has_batch else 0
    sin_batch_stride = sin.stride(0) if cos_has_batch else 0

    BLOCK_SIZE = triton.next_power_of_2(half_head_dim)
    grid = (batch * seq_len, n_q)

    _forge_rope_v1_kernel[grid](
        q_in,  q_in.stride(0),  q_in.stride(1),  q_in.stride(2),
        k_in,  k_in.stride(0),  k_in.stride(1),  k_in.stride(2),
        q_out, q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out, k_out.stride(0), k_out.stride(1), k_out.stride(2),
        cos, cos_batch_stride, cos.stride(1),
        sin, sin_batch_stride, sin.stride(1),
        seq_len=seq_len,
        n_heads_K=n_kv,
        HALF_HEAD_DIM=half_head_dim,
        BACKWARD_PASS=backward_pass,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


class ForgeRoPEv1Function(torch.autograd.Function):
    """autograd.Function wrapper. Out-of-place; saves cos/sin for FSDP2 safety."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.dim() == 4, f"q must be 4D, got shape {tuple(q.shape)}"
        assert k.dim() == 4, f"k must be 4D, got shape {tuple(k.shape)}"
        assert cos.dim() == 3, f"cos must be 3D, got shape {tuple(cos.shape)}"
        assert sin.dim() == 3, f"sin must be 3D, got shape {tuple(sin.shape)}"

        batch, n_q, seq_len, head_dim = q.shape
        n_kv = k.shape[1]

        assert k.shape == (batch, n_kv, seq_len, head_dim), f"k shape {tuple(k.shape)} mismatch"
        assert cos.shape[0] in (1, batch), \
            f"cos.shape[0] must be 1 or batch={batch}, got {cos.shape[0]}"
        assert cos.shape[1] == seq_len and cos.shape[2] == head_dim, \
            f"cos shape {tuple(cos.shape)} mismatch (expected (.,{seq_len},{head_dim}))"
        assert sin.shape == cos.shape, \
            f"sin {tuple(sin.shape)} / cos {tuple(cos.shape)} shape mismatch"
        assert head_dim & (head_dim - 1) == 0, f"head_dim must be a power of 2, got {head_dim}"

        # Ensure inputs are contiguous in the head/seq/dim axes for stride-based access.
        # Triton handles arbitrary strides, but contiguous makes the access pattern coalesced.
        q = q.contiguous() if not q.is_contiguous() else q
        k = k.contiguous() if not k.is_contiguous() else k

        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

        _launch_kernel(q, k, out_q, out_k, cos, sin, backward_pass=False)

        ctx.save_for_backward(cos, sin)
        return out_q, out_k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors

        dq = dq.contiguous() if not dq.is_contiguous() else dq
        dk = dk.contiguous() if not dk.is_contiguous() else dk

        dq_in = torch.empty_like(dq)
        dk_in = torch.empty_like(dk)

        _launch_kernel(dq, dk, dq_in, dk_in, cos, sin, backward_pass=True)

        return dq_in, dk_in, None, None


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to Q and K using ForgeRoPE V1 fused kernel.

    Args:
        q:   (batch, n_q_heads,  seq_len, head_dim) — bf16 / fp16 / fp32
        k:   (batch, n_kv_heads, seq_len, head_dim)
        cos: (1, seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: same shape as cos

    Returns:
        (q_out, k_out)
    """
    return ForgeRoPEv1Function.apply(q, k, cos, sin)


class ForgeRoPEv1(torch.nn.Module):
    """Drop-in apply-step replacement for HF's apply_rotary_pos_emb.

    Does NOT precompute cos/sin — relies on the caller (typically HF's
    Qwen3RotaryEmbedding module) to produce them, matching HF's split.
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin):
        return apply_rope(q, k, cos, sin)


__all__ = ["apply_rope", "ForgeRoPEv1Function", "ForgeRoPEv1"]

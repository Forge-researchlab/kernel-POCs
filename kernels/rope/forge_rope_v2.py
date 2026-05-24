"""ForgeRoPE V2 — GQA-aligned head grouping.

The headline change vs V1:
- Grid is now (batch * seq_len, n_kv_heads) instead of (batch * seq_len, n_q_heads).
- Each program handles G = n_q // n_kv consecutive Q heads + the single matching K head.
- 2D tile across G heads × head_dim/2 columns for Q. K is a single head per program.
- cos/sin loaded once per program, reused across all G Q heads and the 1 K head.
- No GQA mask branch: every program does the same amount of work.

Theory:
- For Qwen3-8B (n_q=32, n_kv=8): grid (b*s, 8) instead of (b*s, 32) — 4x fewer programs.
  Each program does 4 Q heads + 1 K head, cos/sin reused 5x.
- For MHA (n_q == n_kv): G=1, identical to V1's Unsloth-QK shape.
- For MQA (n_kv=1): G=n_q, one program does all heads — Liger-like.

Constraints:
- Requires n_q % n_kv == 0 (always true for real GQA models).
- G is determined at launch (constexpr); not autotuned. V3 could autotune G ∈ {1, 2, 4, n_q//n_kv}.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forge_rope_v2_kernel(
    Q_ptr, Q_batch_stride, Q_head_stride, Q_seq_stride,
    K_ptr, K_batch_stride, K_head_stride, K_seq_stride,
    OutQ_ptr, OutQ_batch_stride, OutQ_head_stride, OutQ_seq_stride,
    OutK_ptr, OutK_batch_stride, OutK_head_stride, OutK_seq_stride,
    cos_ptr, cos_batch_stride, cos_seq_stride,
    sin_ptr, sin_batch_stride, sin_seq_stride,
    seq_len,
    G: tl.constexpr,              # n_q // n_kv (exact division required)
    G_BLOCK: tl.constexpr,        # next_pow2(G), for Triton arange
    HALF_HEAD_DIM: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,     # next_pow2(HALF_HEAD_DIM)
):
    """One program: one (batch, seq) token × one K head's group of (G Q heads + 1 K head).

    program_id(1) directly indexes the K head; the Q heads served by that K head
    are [group_id*G, group_id*G + G - 1].
    """
    # ---- Grid coordinates ----
    row_pos  = tl.program_id(0).to(tl.int64)          # 0 .. batch*seq_len - 1
    group_id = tl.program_id(1).to(tl.int64)          # 0 .. n_kv - 1; == K head index

    batch_id = row_pos // seq_len
    seq_id   = row_pos %  seq_len

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask    = col_offsets < HALF_HEAD_DIM

    # ---- Load cos/sin row (head_dim/2 cols) ONCE per program ----
    cos_row_ptr = cos_ptr + batch_id * cos_batch_stride + seq_id * cos_seq_stride
    sin_row_ptr = sin_ptr + batch_id * sin_batch_stride + seq_id * sin_seq_stride

    cos_row = tl.load(cos_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    sin_row = tl.load(sin_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # ---- Q work: 2D tile over G heads × HALF_HEAD_DIM columns ----
    # Q head indices served by this program: [group_id*G, group_id*G + G - 1]
    g_offsets = tl.arange(0, G_BLOCK)
    g_mask    = g_offsets < G
    q_head_base = group_id * G

    q_base_ptr = Q_ptr + batch_id * Q_batch_stride + seq_id * Q_seq_stride
    # 2D offset: (G_BLOCK, BLOCK_SIZE) shape
    q_2d_offsets = (q_head_base + g_offsets)[:, None] * Q_head_stride + col_offsets[None, :]
    q_2d_mask    = g_mask[:, None] & col_mask[None, :]

    q_lo = tl.load(q_base_ptr + q_2d_offsets,                 mask=q_2d_mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_base_ptr + q_2d_offsets + HALF_HEAD_DIM, mask=q_2d_mask, other=0.0).to(tl.float32)

    # Broadcast cos_row/sin_row (1D, HALF_HEAD_DIM) across G heads via [None, :]
    out_q_lo = q_lo * cos_row[None, :] - q_hi * sin_row[None, :]
    out_q_hi = q_hi * cos_row[None, :] + q_lo * sin_row[None, :]

    out_q_base_ptr = OutQ_ptr + batch_id * OutQ_batch_stride + seq_id * OutQ_seq_stride
    out_q_2d_offsets = (q_head_base + g_offsets)[:, None] * OutQ_head_stride + col_offsets[None, :]
    tl.store(out_q_base_ptr + out_q_2d_offsets,                 out_q_lo, mask=q_2d_mask)
    tl.store(out_q_base_ptr + out_q_2d_offsets + HALF_HEAD_DIM, out_q_hi, mask=q_2d_mask)

    # ---- K work: single head per program (group_id == K head index) ----
    k_base_ptr = K_ptr + batch_id * K_batch_stride + group_id * K_head_stride + seq_id * K_seq_stride

    k_lo = tl.load(k_base_ptr + col_offsets,                 mask=col_mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_base_ptr + col_offsets + HALF_HEAD_DIM, mask=col_mask, other=0.0).to(tl.float32)

    out_k_lo = k_lo * cos_row - k_hi * sin_row
    out_k_hi = k_hi * cos_row + k_lo * sin_row

    out_k_base_ptr = OutK_ptr + batch_id * OutK_batch_stride + group_id * OutK_head_stride + seq_id * OutK_seq_stride
    tl.store(out_k_base_ptr + col_offsets,                 out_k_lo, mask=col_mask)
    tl.store(out_k_base_ptr + col_offsets + HALF_HEAD_DIM, out_k_hi, mask=col_mask)


def _launch_kernel(q_in, k_in, q_out, k_out, cos, sin, backward_pass: bool):
    """Internal launcher. Assumes shapes are validated."""
    batch, n_q, seq_len, head_dim = q_in.shape
    n_kv = k_in.shape[1]

    assert n_q % n_kv == 0, \
        f"V2 requires n_q % n_kv == 0 (got n_q={n_q}, n_kv={n_kv}). Use V1 for non-divisible cases."

    G = n_q // n_kv
    G_BLOCK = triton.next_power_of_2(G)
    half_head_dim = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(half_head_dim)

    cos_has_batch = (cos.shape[0] == batch)
    cos_batch_stride = cos.stride(0) if cos_has_batch else 0
    sin_batch_stride = sin.stride(0) if cos_has_batch else 0

    grid = (batch * seq_len, n_kv)

    _forge_rope_v2_kernel[grid](
        q_in,  q_in.stride(0),  q_in.stride(1),  q_in.stride(2),
        k_in,  k_in.stride(0),  k_in.stride(1),  k_in.stride(2),
        q_out, q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out, k_out.stride(0), k_out.stride(1), k_out.stride(2),
        cos, cos_batch_stride, cos.stride(1),
        sin, sin_batch_stride, sin.stride(1),
        seq_len=seq_len,
        G=G,
        G_BLOCK=G_BLOCK,
        HALF_HEAD_DIM=half_head_dim,
        BACKWARD_PASS=backward_pass,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


class ForgeRoPEv2Function(torch.autograd.Function):
    """autograd.Function wrapper for V2. Out-of-place; FSDP2-safe."""

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
            f"cos shape {tuple(cos.shape)} mismatch"
        assert sin.shape == cos.shape, \
            f"sin/cos shape mismatch: {tuple(sin.shape)} vs {tuple(cos.shape)}"
        assert head_dim & (head_dim - 1) == 0, f"head_dim must be a power of 2, got {head_dim}"

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
    """Apply RoPE to Q and K using ForgeRoPE V2 (GQA-aligned head grouping).

    Args:
        q:   (batch, n_q_heads,  seq_len, head_dim) — bf16 / fp16 / fp32
        k:   (batch, n_kv_heads, seq_len, head_dim)        n_q % n_kv == 0 required
        cos: (1, seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: same shape as cos

    Returns:
        (q_out, k_out)
    """
    return ForgeRoPEv2Function.apply(q, k, cos, sin)


class ForgeRoPEv2(torch.nn.Module):
    """Drop-in apply-step replacement for HF's apply_rotary_pos_emb (V2 with GQA grouping)."""

    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin):
        return apply_rope(q, k, cos, sin)


__all__ = ["apply_rope", "ForgeRoPEv2Function", "ForgeRoPEv2"]

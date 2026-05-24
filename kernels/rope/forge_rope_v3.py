"""ForgeRoPE V3 — V2 + Triton autotune over num_warps and num_stages.

V2 picked num_warps=4 unconditionally. That's not optimal across shapes:
- For small grids (MQA: n_kv=1 → grid axis 1 is tiny) we may want fewer warps so
  more programs run concurrently on SMs.
- For large grids (Qwen3-8B: n_kv=8) we may want more warps to hide memory latency.

V3 adds `@triton.autotune` over:
- num_warps ∈ {2, 4, 8, 16}
- num_stages ∈ {2, 3}                       # async load pipelining

Tuning key: `seq_len` (changes grid size and thus optimal warp count).
Constexpr re-compilation already specializes per (HALF_HEAD_DIM, G, G_BLOCK, BACKWARD_PASS),
so each shape combination gets its own tuned config independently.

Cost: first call per key is slow (compiles + measures 8 configs and picks the best).
After that it's free — Triton caches the chosen config.
"""

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in (2, 4, 8, 16)
    for ns in (2, 3)
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["seq_len"])
@triton.jit
def _forge_rope_v3_kernel(
    Q_ptr, Q_batch_stride, Q_head_stride, Q_seq_stride,
    K_ptr, K_batch_stride, K_head_stride, K_seq_stride,
    OutQ_ptr, OutQ_batch_stride, OutQ_head_stride, OutQ_seq_stride,
    OutK_ptr, OutK_batch_stride, OutK_head_stride, OutK_seq_stride,
    cos_ptr, cos_batch_stride, cos_seq_stride,
    sin_ptr, sin_batch_stride, sin_seq_stride,
    seq_len,
    G: tl.constexpr,
    G_BLOCK: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel body identical to V2 — only the launch decoration differs."""
    row_pos  = tl.program_id(0).to(tl.int64)
    group_id = tl.program_id(1).to(tl.int64)

    batch_id = row_pos // seq_len
    seq_id   = row_pos %  seq_len

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask    = col_offsets < HALF_HEAD_DIM

    cos_row_ptr = cos_ptr + batch_id * cos_batch_stride + seq_id * cos_seq_stride
    sin_row_ptr = sin_ptr + batch_id * sin_batch_stride + seq_id * sin_seq_stride

    cos_row = tl.load(cos_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    sin_row = tl.load(sin_row_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    if BACKWARD_PASS:
        sin_row = -sin_row

    g_offsets = tl.arange(0, G_BLOCK)
    g_mask    = g_offsets < G
    q_head_base = group_id * G

    q_base_ptr = Q_ptr + batch_id * Q_batch_stride + seq_id * Q_seq_stride
    q_2d_offsets = (q_head_base + g_offsets)[:, None] * Q_head_stride + col_offsets[None, :]
    q_2d_mask    = g_mask[:, None] & col_mask[None, :]

    q_lo = tl.load(q_base_ptr + q_2d_offsets,                 mask=q_2d_mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_base_ptr + q_2d_offsets + HALF_HEAD_DIM, mask=q_2d_mask, other=0.0).to(tl.float32)

    out_q_lo = q_lo * cos_row[None, :] - q_hi * sin_row[None, :]
    out_q_hi = q_hi * cos_row[None, :] + q_lo * sin_row[None, :]

    out_q_base_ptr = OutQ_ptr + batch_id * OutQ_batch_stride + seq_id * OutQ_seq_stride
    out_q_2d_offsets = (q_head_base + g_offsets)[:, None] * OutQ_head_stride + col_offsets[None, :]
    tl.store(out_q_base_ptr + out_q_2d_offsets,                 out_q_lo, mask=q_2d_mask)
    tl.store(out_q_base_ptr + out_q_2d_offsets + HALF_HEAD_DIM, out_q_hi, mask=q_2d_mask)

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
        f"V3 requires n_q % n_kv == 0 (got n_q={n_q}, n_kv={n_kv})."

    G = n_q // n_kv
    G_BLOCK = triton.next_power_of_2(G)
    half_head_dim = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(half_head_dim)

    cos_has_batch = (cos.shape[0] == batch)
    cos_batch_stride = cos.stride(0) if cos_has_batch else 0
    sin_batch_stride = sin.stride(0) if cos_has_batch else 0

    grid = (batch * seq_len, n_kv)

    _forge_rope_v3_kernel[grid](
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
        # num_warps and num_stages are now picked by autotune
    )


def get_autotune_cache_for(q, k, cos, sin, backward_pass: bool = False):
    """Diagnostic: return the autotune cache key + chosen config for a given input.

    Useful for reporting which num_warps/num_stages was picked per shape in the
    benchmark JSON output.
    """
    batch, n_q, seq_len, head_dim = q.shape
    n_kv = k.shape[1]
    G = n_q // n_kv
    G_BLOCK = triton.next_power_of_2(G)
    half_head_dim = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(half_head_dim)
    # Triton autotune stores chosen configs in `.best_config` after the first call.
    # The cache is keyed by tuple of (key arg values, constexpr values).
    # We can read `_forge_rope_v3_kernel.best_config` after a call.
    try:
        best = _forge_rope_v3_kernel.best_config
        return {
            "num_warps": best.num_warps,
            "num_stages": best.num_stages,
            "all_kwargs": dict(best.kwargs) if hasattr(best, "kwargs") else None,
        }
    except Exception as e:
        return {"error": str(e)}


class ForgeRoPEv3Function(torch.autograd.Function):
    """autograd.Function wrapper for V3. Same semantics as V2; only the kernel is autotuned."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.dim() == 4
        assert k.dim() == 4
        assert cos.dim() == 3 and sin.dim() == 3

        batch, n_q, seq_len, head_dim = q.shape
        n_kv = k.shape[1]

        assert k.shape == (batch, n_kv, seq_len, head_dim)
        assert cos.shape[0] in (1, batch)
        assert cos.shape[1] == seq_len and cos.shape[2] == head_dim
        assert sin.shape == cos.shape
        assert head_dim & (head_dim - 1) == 0

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
    """Apply RoPE to Q and K using ForgeRoPE V3 (V2 + autotune).

    First call per (head_dim, G, seq_len) triggers tuning over 8 configs;
    subsequent calls use the cached best config.
    """
    return ForgeRoPEv3Function.apply(q, k, cos, sin)


class ForgeRoPEv3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin):
        return apply_rope(q, k, cos, sin)


__all__ = ["apply_rope", "ForgeRoPEv3Function", "ForgeRoPEv3", "get_autotune_cache_for"]

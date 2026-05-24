"""ForgeRoPE V1 test runner.

Runs:
  1. Environment fingerprint
  2. Import smoke test
  3. Forward correctness (Forge V1 vs HF reference + baselines)
  4. Backward correctness (gradients vs HF autograd)
  5. Gradcheck on fp64
  6. Light forward-only timing (Forge V1 vs PyTorch / Liger / Unsloth-default / Unsloth-fused-QK)

Emits two artifacts:
  - kernels/rope/tests/results/v1_results.json   (machine-readable, structured)
  - kernels/rope/tests/results/v1_summary.md     (human-readable summary)

Usage:
  python -m kernels.rope.tests.test_v1
  # or
  python kernels/rope/tests/test_v1.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as a script (python test_v1.py) by adding repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import triton
import triton.testing

from kernels.rope.forge_rope_v1 import apply_rope as forge_apply_rope, ForgeRoPEv1Function
from kernels.rope.baselines import liger, unsloth


# ---------------------------------------------------------------------------
# HF reference (inlined to avoid pulling in the full transformers import)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Reference from huggingface/transformers Qwen3 modeling file."""
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype, device="cuda", seed=0,
               cos_batched=False, base=10000.0):
    """Generate (q, k, cos, sin) matching HF's apply_rotary_pos_emb convention."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, n_q, seq_len, head_dim, device=device, dtype=dtype, generator=g)
    k = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=dtype, generator=g)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)        # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (seq_len, head_dim)

    cos_2d = emb.cos().to(dtype)
    sin_2d = emb.sin().to(dtype)
    if cos_batched:
        cos = cos_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()
        sin = sin_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()
    else:
        cos = cos_2d.unsqueeze(0)         # (1, seq_len, head_dim)
        sin = sin_2d.unsqueeze(0)
    return q, k, cos, sin


def max_abs(a, b):
    """Max absolute diff between two tensors, computed in fp32 for comparability."""
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Tolerances per dtype (chosen to match Forge kernel contract)
# ---------------------------------------------------------------------------

# bf16 has ~3 decimal digits of precision. Forge accumulates in fp32 then rounds to
# the input dtype on store. HF reference computes in input dtype throughout. The
# diff is the bf16 quantization noise of Forge's fp32-accurate result — up to ~2 ULPs
# at unit scale = 0.0156. Tolerance of 5e-2 is 6 ULPs, safe for any sane input.
TOLERANCES = {
    "torch.bfloat16": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float16":  {"atol": 5e-2, "rtol": 5e-2},
    "torch.float32":  {"atol": 1e-5, "rtol": 1e-5},
    "torch.float64":  {"atol": 1e-7, "rtol": 1e-7},
}


def passes_tolerance(diff: float, dtype: torch.dtype) -> bool:
    tol = TOLERANCES.get(str(dtype), {"atol": 1e-3})
    return diff <= tol["atol"]


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

# (label, batch, n_q, n_kv, seq_len, head_dim)
SHAPES = [
    ("demo_tiny",         2,  4,  2,   16,  64),
    ("qwen25_0p5b_short", 1, 14,  2,  512,  64),    # Qwen2.5-0.5B demo
    ("qwen3_8b_short",    4, 32,  8,  512, 128),    # Qwen3-8B shape, shorter for speed
    ("qwen3_8b_train",    2, 32,  8, 2048, 128),    # Qwen3-8B training shape
    ("mha_no_gqa",        2, 16, 16, 1024, 128),    # no-GQA case
    ("mqa_extreme",       2,  8,  1, 1024, 128),    # extreme GQA / MQA
]

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def test_forward_correctness() -> dict[str, Any]:
    """Forward output: Forge V1 vs HF-fp32 reference + cross-check vs baselines.

    Reference is computed in fp32 (upcast q/k/cos/sin), then cast back to input
    dtype for comparison. This isolates Forge's actual error from HF's bf16-only
    rounding noise. Forge uses fp32 accumulation by design (per FORGE_CONTEXT.md
    mandate), so Forge's output should be *more* accurate than HF-in-dtype.
    """
    cases = []
    for label, batch, n_q, n_kv, seq_len, head_dim in SHAPES:
        for dtype in DTYPES:
            for cos_batched in [False, True]:
                q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim,
                                            dtype, cos_batched=cos_batched)

                # HF reference computed in fp32 then cast to input dtype
                q_hf_fp32, k_hf_fp32 = hf_apply_rotary_pos_emb(
                    q.float(), k.float(), cos.float(), sin.float()
                )
                q_hf = q_hf_fp32.to(dtype)
                k_hf = k_hf_fp32.to(dtype)

                # Forge V1
                q_f, k_f = forge_apply_rope(q.clone(), k.clone(), cos, sin)

                # Baselines (Liger always supports batched cos; Unsloth does not)
                q_l, k_l = liger.apply_rope(q.clone(), k.clone(), cos, sin)

                diffs = {
                    "forge_vs_hffp32_q": max_abs(q_f, q_hf),
                    "forge_vs_hffp32_k": max_abs(k_f, k_hf),
                    "forge_vs_liger_q":  max_abs(q_f, q_l),
                    "forge_vs_liger_k":  max_abs(k_f, k_l),
                }

                if not cos_batched:
                    # Unsloth's `cos.squeeze()` only works for cos.shape[0] == 1
                    q_u,  k_u  = unsloth.apply_rope(q.clone(), k.clone(), cos, sin)
                    q_uq, k_uq = unsloth.apply_rope_qk_fused(q.clone(), k.clone(), cos, sin)
                    diffs["forge_vs_unsloth_q"]    = max_abs(q_f, q_u)
                    diffs["forge_vs_unsloth_k"]    = max_abs(k_f, k_u)
                    diffs["forge_vs_unsloth_qk_q"] = max_abs(q_f, q_uq)
                    diffs["forge_vs_unsloth_qk_k"] = max_abs(k_f, k_uq)
                else:
                    diffs["unsloth_skipped"] = "cos.shape[0] != 1 — Unsloth limitation"

                passed = (
                    passes_tolerance(diffs["forge_vs_hffp32_q"], dtype) and
                    passes_tolerance(diffs["forge_vs_hffp32_k"], dtype) and
                    not torch.isnan(q_f).any().item() and
                    not torch.isnan(k_f).any().item()
                )
                cases.append({
                    "shape_label": label,
                    "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                              "seq_len": seq_len, "head_dim": head_dim},
                    "dtype": str(dtype),
                    "cos_batched": cos_batched,
                    "tolerance": TOLERANCES[str(dtype)],
                    "diffs": diffs,
                    "has_nan_q": bool(torch.isnan(q_f).any().item()),
                    "has_nan_k": bool(torch.isnan(k_f).any().item()),
                    "passed": bool(passed),
                })

    n_passed = sum(1 for c in cases if c["passed"])
    return {
        "summary": {"total": len(cases), "passed": n_passed, "failed": len(cases) - n_passed},
        "reference": "HF apply_rotary_pos_emb computed in fp32, cast to input dtype",
        "cases": cases,
    }


def test_backward_correctness() -> dict[str, Any]:
    """Backward gradients: Forge V1 vs HF autograd computed in fp32.

    Reference autograd runs in fp32 (q/k/cos/sin upcast), gradients cast back to
    input dtype for fair comparison. Without this, the bf16/fp16 diff is dominated
    by HF's lower-precision arithmetic, not Forge's actual error.
    """
    cases = []
    shape_subset = [
        ("demo_tiny",      2,  4,  2,   16,  64),
        ("qwen3_8b_short", 4, 32,  8,  512, 128),
        ("mqa_extreme",    2,  8,  1, 1024, 128),
    ]
    for label, batch, n_q, n_kv, seq_len, head_dim in shape_subset:
        for dtype in [torch.bfloat16, torch.float32]:  # skip fp16 backward for speed
            q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype)
            grad_q = torch.randn(batch, n_q,  seq_len, head_dim, device="cuda", dtype=dtype)
            grad_k = torch.randn(batch, n_kv, seq_len, head_dim, device="cuda", dtype=dtype)

            # HF reference in fp32, then cast back to input dtype for comparison
            q_hf = q.float().clone().requires_grad_(True)
            k_hf = k.float().clone().requires_grad_(True)
            cos_hf = cos.float()
            sin_hf = sin.float()
            out_q_hf, out_k_hf = hf_apply_rotary_pos_emb(q_hf, k_hf, cos_hf, sin_hf)
            loss_hf = (out_q_hf * grad_q.float()).sum() + (out_k_hf * grad_k.float()).sum()
            loss_hf.backward()
            out_q_hf_cast = out_q_hf.detach().to(dtype)
            out_k_hf_cast = out_k_hf.detach().to(dtype)
            dq_hf_cast = q_hf.grad.to(dtype)
            dk_hf_cast = k_hf.grad.to(dtype)

            # Forge V1 in input dtype (fp32 accumulation inside the kernel)
            q_fg = q.clone().requires_grad_(True)
            k_fg = k.clone().requires_grad_(True)
            out_q_f, out_k_f = forge_apply_rope(q_fg, k_fg, cos, sin)
            loss_f = (out_q_f * grad_q).sum() + (out_k_f * grad_k).sum()
            loss_f.backward()

            fwd_diff_q = max_abs(out_q_f, out_q_hf_cast)
            fwd_diff_k = max_abs(out_k_f, out_k_hf_cast)
            bwd_diff_dq = max_abs(q_fg.grad, dq_hf_cast)
            bwd_diff_dk = max_abs(k_fg.grad, dk_hf_cast)

            passed = all(passes_tolerance(d, dtype) for d in
                         [fwd_diff_q, fwd_diff_k, bwd_diff_dq, bwd_diff_dk])

            cases.append({
                "shape_label": label,
                "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                          "seq_len": seq_len, "head_dim": head_dim},
                "dtype": str(dtype),
                "tolerance": TOLERANCES[str(dtype)],
                "forward_diff": {"q": fwd_diff_q, "k": fwd_diff_k},
                "backward_diff": {"dq": bwd_diff_dq, "dk": bwd_diff_dk},
                "passed": bool(passed),
            })

    n_passed = sum(1 for c in cases if c["passed"])
    return {
        "summary": {"total": len(cases), "passed": n_passed, "failed": len(cases) - n_passed},
        "reference": "HF apply_rotary_pos_emb + autograd in fp32, grads cast to input dtype",
        "cases": cases,
    }


def test_gradcheck() -> dict[str, Any]:
    """torch.autograd.gradcheck on a small fp64 input.

    Tolerances are loose because Forge's kernel accumulates in fp32 internally
    (per FORGE_CONTEXT.md), so the analytical gradient is fp32-accurate even
    for fp64 inputs. With eps=1e-3, finite-diff signal survives fp32 quantization.

    A direct manual check (one-hot dy → expected cos/-sin) is also performed as
    a math sanity check independent of gradcheck.
    """
    batch, n_q, n_kv, seq_len, head_dim = 2, 4, 2, 8, 32
    q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, torch.float64)

    # --- Direct math check: one-hot dy → exact cos / -sin gradient ---
    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    out_q, out_k = forge_apply_rope(q1, k1, cos, sin)
    pos_s, pos_d = 1, 0
    grad_q_oh = torch.zeros_like(out_q); grad_q_oh[0, 0, pos_s, pos_d] = 1.0
    grad_k_oh = torch.zeros_like(out_k)
    (out_q * grad_q_oh).sum().add_((out_k * grad_k_oh).sum()).backward()
    expected_cos = cos[0, pos_s, pos_d].item()
    expected_neg_sin = -sin[0, pos_s, pos_d].item()
    got_lo = q1.grad[0, 0, pos_s, pos_d].item()
    got_hi = q1.grad[0, 0, pos_s, pos_d + head_dim // 2].item()
    manual_check = {
        "one_hot_position": {"s": pos_s, "d": pos_d},
        "expected_cos":     expected_cos,
        "expected_neg_sin": expected_neg_sin,
        "got_lo":           got_lo,
        "got_hi":           got_hi,
        "lo_diff": abs(got_lo - expected_cos),
        "hi_diff": abs(got_hi - expected_neg_sin),
        "passed":  abs(got_lo - expected_cos) < 1e-5 and abs(got_hi - expected_neg_sin) < 1e-5,
    }

    # --- torch.autograd.gradcheck with relaxed tolerances for Triton/fp32 internal compute ---
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    try:
        passed = torch.autograd.gradcheck(
            ForgeRoPEv1Function.apply,
            (q2, k2, cos, sin),
            eps=1e-3,   # larger eps to survive fp32 quantization in the kernel
            atol=1e-2, rtol=1e-2,
            nondet_tol=1e-3,
            check_undefined_grad=False,
            check_batched_grad=False,
        )
        return {
            "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                      "seq_len": seq_len, "head_dim": head_dim},
            "dtype": "torch.float64",
            "gradcheck_settings": {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2, "nondet_tol": 1e-3},
            "manual_check": manual_check,
            "passed": bool(passed),
            "error": None,
        }
    except Exception as e:
        return {
            "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                      "seq_len": seq_len, "head_dim": head_dim},
            "dtype": "torch.float64",
            "gradcheck_settings": {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2, "nondet_tol": 1e-3},
            "manual_check": manual_check,
            "passed": False,
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        }


def test_forward_timing() -> dict[str, Any]:
    """Forward-only timing: Forge V1 vs PyTorch / Liger / Unsloth-default / Unsloth-QK.

    Uses triton.testing.do_bench (handles warmup, CUDA sync, returns median ms).
    """
    timing_shapes = [
        ("qwen25_0p5b_short", 1, 14, 2,  512,  64),
        ("qwen3_8b_short",    4, 32, 8,  512, 128),
        ("qwen3_8b_train",    2, 32, 8, 2048, 128),
    ]
    timing_dtypes = [torch.bfloat16, torch.float16]

    cases = []
    for label, batch, n_q, n_kv, seq_len, head_dim in timing_shapes:
        for dtype in timing_dtypes:
            q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype)

            # Each callable must accept no args and re-clone q/k since some kernels mutate inputs.
            def call_pytorch():
                return hf_apply_rotary_pos_emb(q, k, cos, sin)

            def call_liger():
                return liger.apply_rope(q.clone(), k.clone(), cos, sin)

            def call_unsloth():
                return unsloth.apply_rope(q.clone(), k.clone(), cos, sin)

            def call_unsloth_qk():
                return unsloth.apply_rope_qk_fused(q.clone(), k.clone(), cos, sin)

            def call_forge():
                return forge_apply_rope(q, k, cos, sin)  # forge is out-of-place, no clone needed

            timings_ms = {}
            for name, fn in [
                ("pytorch_ref",       call_pytorch),
                ("liger",             call_liger),
                ("unsloth_default",   call_unsloth),
                ("unsloth_fused_qk",  call_unsloth_qk),
                ("forge_v1",          call_forge),
            ]:
                try:
                    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
                    timings_ms[name] = float(ms)
                except Exception as e:
                    timings_ms[name] = None
                    timings_ms[f"{name}_error"] = f"{type(e).__name__}: {e}"

            # Speedups (vs pytorch_ref)
            ref = timings_ms.get("pytorch_ref")
            speedups = {}
            if ref:
                for k_name in ("liger", "unsloth_default", "unsloth_fused_qk", "forge_v1"):
                    t = timings_ms.get(k_name)
                    speedups[k_name] = (ref / t) if t else None

            # HBM bandwidth utilization for forge_v1 (rough)
            # bytes per output element approx 2*read + 2*write per token+head + cos/sin
            element_size = torch.tensor([], dtype=dtype).element_size()
            q_bytes = batch * n_q  * seq_len * head_dim * element_size * 2  # read + write
            k_bytes = batch * n_kv * seq_len * head_dim * element_size * 2
            cos_sin_bytes = 2 * seq_len * head_dim * element_size  # 1x batch broadcast, read only
            total_bytes = q_bytes + k_bytes + cos_sin_bytes
            forge_ms = timings_ms.get("forge_v1")
            bandwidth_gbps = (total_bytes / 1e9) / (forge_ms / 1000.0) if forge_ms else None

            cases.append({
                "shape_label": label,
                "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                          "seq_len": seq_len, "head_dim": head_dim},
                "dtype": str(dtype),
                "median_ms": timings_ms,
                "speedup_vs_pytorch": speedups,
                "total_hbm_traffic_bytes": total_bytes,
                "forge_v1_bandwidth_gbps": bandwidth_gbps,
            })

    return {"cases": cases}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_environment() -> dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda_avail,
    }
    if cuda_avail:
        idx = torch.cuda.current_device()
        env["cuda_device_name"] = torch.cuda.get_device_name(idx)
        env["cuda_device_capability"] = list(torch.cuda.get_device_capability(idx))
        env["cuda_runtime_version"] = torch.version.cuda
    return env


def kernel_config() -> dict[str, Any]:
    return {
        "version": "v1",
        "grid": "(batch * seq_len, n_q_heads)",
        "block_size": "next_power_of_2(head_dim // 2)",
        "num_warps": 4,
        "backward_strategy": "shared kernel via BACKWARD_PASS constexpr; sin negated after load",
        "fp32_accumulation": True,
        "gqa_handling": "if head_pos < n_kv: do K work (Unsloth-QK mask trick)",
        "save_for_backward": ["cos", "sin"],
        "out_of_place": True,
        "autotune": False,
        "head_grouping": False,
    }


def write_markdown_summary(results: dict[str, Any], path: Path) -> None:
    env = results["environment"]
    fwd = results["results"].get("forward_correctness", {}).get("summary",
                                                                {"passed": 0, "total": 0, "failed": 0})
    bwd = results["results"].get("backward_correctness", {}).get("summary",
                                                                 {"passed": 0, "total": 0, "failed": 0})
    grad = results["results"].get("gradcheck_fp64", {"passed": False, "error": "missing"})
    timing_cases = results["results"].get("forward_timing", {}).get("cases", [])

    lines = []
    lines.append(f"# ForgeRoPE V1 — Test Results Summary")
    lines.append(f"")
    lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} "
                 f"(compute {env.get('cuda_device_capability', 'n/a')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}")
    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    lines.append(f"| Suite | Passed | Total |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Forward correctness | {fwd['passed']} | {fwd['total']} |")
    lines.append(f"| Backward correctness | {bwd['passed']} | {bwd['total']} |")
    lines.append(f"| Gradcheck (fp64) | {'✓' if grad['passed'] else '✗'} | 1 |")
    lines.append("")
    if grad.get("error"):
        lines.append(f"**Gradcheck error:** `{grad['error']}`")
        lines.append("")
    lines.append("## Forward timing (median, lower = better)")
    lines.append("")
    lines.append("| Shape | dtype | PyTorch (ms) | Liger (ms) | Unsloth-default (ms) | Unsloth-fused-QK (ms) | **Forge V1 (ms)** | Forge speedup vs PT |")
    lines.append("|---|---|---|---|---|---|---|---|")
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
    for c in timing_cases:
        t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v1")
        sh = c["shape"]
        sp_str = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "n/a"
        lines.append(
            f"| {c['shape_label']} ({sh['batch']}×{sh['n_q']}/{sh['n_kv']}×{sh['seq_len']}×{sh['head_dim']}) "
            f"| {c['dtype']} "
            f"| {fmt(t.get('pytorch_ref'))} "
            f"| {fmt(t.get('liger'))} "
            f"| {fmt(t.get('unsloth_default'))} "
            f"| {fmt(t.get('unsloth_fused_qk'))} "
            f"| **{fmt(t.get('forge_v1'))}** "
            f"| {sp_str} |"
        )
    lines.append("")
    lines.append("## HBM bandwidth utilization (Forge V1)")
    lines.append("")
    lines.append("| Shape | dtype | Total traffic (MB) | Forge V1 time (ms) | Achieved bandwidth (GB/s) |")
    lines.append("|---|---|---|---|---|")
    for c in timing_cases:
        bw = c["forge_v1_bandwidth_gbps"]
        bw_str = f"{bw:.0f}" if isinstance(bw, (int, float)) else "n/a"
        lines.append(
            f"| {c['shape_label']} | {c['dtype']} "
            f"| {c['total_hbm_traffic_bytes'] / 1e6:.1f} "
            f"| {fmt(c['median_ms'].get('forge_v1'))} "
            f"| {bw_str} |"
        )

    path.write_text("\n".join(lines))


def main():
    results: dict[str, Any] = {
        "version": "v1",
        "implementation_file": "kernels/rope/forge_rope_v1.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        "kernel_config": kernel_config(),
        "results": {},
        "errors": [],
    }

    # Import smoke
    results["results"]["import_smoke"] = {"passed": True, "error": None}

    # Forward correctness
    print("[1/4] Forward correctness ...", flush=True)
    try:
        results["results"]["forward_correctness"] = test_forward_correctness()
    except Exception as e:
        results["results"]["forward_correctness"] = {"error": str(e),
                                                     "traceback": traceback.format_exc()}
        results["errors"].append({"phase": "forward_correctness", "error": str(e)})

    # Backward correctness
    print("[2/4] Backward correctness ...", flush=True)
    try:
        results["results"]["backward_correctness"] = test_backward_correctness()
    except Exception as e:
        results["results"]["backward_correctness"] = {"error": str(e),
                                                      "traceback": traceback.format_exc()}
        results["errors"].append({"phase": "backward_correctness", "error": str(e)})

    # Gradcheck
    print("[3/4] Gradcheck (fp64) ...", flush=True)
    try:
        results["results"]["gradcheck_fp64"] = test_gradcheck()
    except Exception as e:
        results["results"]["gradcheck_fp64"] = {"passed": False, "error": str(e),
                                                "traceback": traceback.format_exc()}
        results["errors"].append({"phase": "gradcheck_fp64", "error": str(e)})

    # Timing
    print("[4/4] Forward timing ...", flush=True)
    try:
        results["results"]["forward_timing"] = test_forward_timing()
    except Exception as e:
        results["results"]["forward_timing"] = {"error": str(e),
                                                "traceback": traceback.format_exc()}
        results["errors"].append({"phase": "forward_timing", "error": str(e)})

    # Write artifacts
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "v1_results.json"
    md_path = out_dir / "v1_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    try:
        write_markdown_summary(results, md_path)
    except Exception as e:
        results["errors"].append({"phase": "summary_md", "error": str(e),
                                  "traceback": traceback.format_exc()})
        # Re-dump JSON with the appended error
        json_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")

    # Print one-screen summary
    print("\n=== ForgeRoPE V1 — short summary ===")
    fwd = results["results"].get("forward_correctness", {}).get("summary", {})
    bwd = results["results"].get("backward_correctness", {}).get("summary", {})
    grad = results["results"].get("gradcheck_fp64", {})
    print(f"  Forward correctness:  {fwd.get('passed', 0)}/{fwd.get('total', 0)} passed")
    print(f"  Backward correctness: {bwd.get('passed', 0)}/{bwd.get('total', 0)} passed")
    print(f"  Gradcheck fp64:       {'PASS' if grad.get('passed') else 'FAIL'}")
    if results["errors"]:
        print(f"  Errors during run:    {len(results['errors'])}")
        for err in results["errors"]:
            print(f"    - {err['phase']}: {err['error']}")
    return 0 if not results["errors"] and grad.get("passed") and fwd.get("failed", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""ForgeRoPE V2 — comprehensive correctness + benchmark runner.

Mirrors the structure of tests/test_v1.py but adds:
  - V2-specific correctness checks (Forge V2 vs HF + cross-check vs V1)
  - Side-by-side timing: Forge V2 vs Forge V1 vs Liger vs Unsloth-default vs Unsloth-fused-QK vs PyTorch
  - Backward timing (not just forward)
  - HBM bandwidth utilization analysis
  - V1 vs V2 delta summary (the headline V2 question)

Emits:
  - kernels/rope/benchmarks/results/v2_results.json   (machine-readable)
  - kernels/rope/benchmarks/results/v2_summary.md     (human-readable)

Usage:
  python -m kernels.rope.benchmarks.bench_v2
  # or
  python kernels/rope/benchmarks/bench_v2.py
"""

from __future__ import annotations

import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import triton
import triton.testing

from kernels.rope.forge_rope_v1 import apply_rope as forge_v1_apply, ForgeRoPEv1Function
from kernels.rope.forge_rope_v2 import apply_rope as forge_v2_apply, ForgeRoPEv2Function
from kernels.rope.baselines import liger, unsloth


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
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
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, n_q,  seq_len, head_dim, device=device, dtype=dtype, generator=g)
    k = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=dtype, generator=g)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)

    cos_2d = emb.cos().to(dtype)
    sin_2d = emb.sin().to(dtype)
    if cos_batched:
        cos = cos_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()
        sin = sin_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()
    else:
        cos = cos_2d.unsqueeze(0)
        sin = sin_2d.unsqueeze(0)
    return q, k, cos, sin


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Tolerances per dtype (must match V1's runner for apples-to-apples)
# ---------------------------------------------------------------------------

TOLERANCES = {
    "torch.bfloat16": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float16":  {"atol": 5e-2, "rtol": 5e-2},
    "torch.float32":  {"atol": 1e-5, "rtol": 1e-5},
    "torch.float64":  {"atol": 1e-7, "rtol": 1e-7},
}


def passes_tolerance(diff: float, dtype: torch.dtype) -> bool:
    tol = TOLERANCES.get(str(dtype), {"atol": 1e-3})
    return diff <= tol["atol"]


# All shapes here have n_q % n_kv == 0 (V2 requirement)
SHAPES = [
    ("demo_tiny",         2,  4,  2,   16,  64),    # G=2
    ("qwen3_8b_short",    4, 32,  8,  512, 128),    # G=4 (Qwen3-8B GQA ratio)
    ("qwen3_8b_train",    2, 32,  8, 2048, 128),    # G=4
    ("mha_no_gqa",        2, 16, 16, 1024, 128),    # G=1 (no GQA — degenerate to V1-like)
    ("mqa_extreme",       2,  8,  1, 1024, 128),    # G=8 (all heads in one program)
]

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_forward_correctness() -> dict[str, Any]:
    """Forge V2 vs HF (fp32 reference) and cross-check vs Forge V1."""
    cases = []
    for label, batch, n_q, n_kv, seq_len, head_dim in SHAPES:
        for dtype in DTYPES:
            for cos_batched in [False, True]:
                q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim,
                                            dtype, cos_batched=cos_batched)

                # HF reference in fp32 then cast back to input dtype
                q_hf_fp32, k_hf_fp32 = hf_apply_rotary_pos_emb(
                    q.float(), k.float(), cos.float(), sin.float()
                )
                q_hf = q_hf_fp32.to(dtype)
                k_hf = k_hf_fp32.to(dtype)

                # Forge V2 (the kernel under test)
                q_v2, k_v2 = forge_v2_apply(q.clone(), k.clone(), cos, sin)

                # Forge V1 cross-check
                q_v1, k_v1 = forge_v1_apply(q.clone(), k.clone(), cos, sin)

                diffs = {
                    "v2_vs_hffp32_q": max_abs(q_v2, q_hf),
                    "v2_vs_hffp32_k": max_abs(k_v2, k_hf),
                    "v2_vs_v1_q":     max_abs(q_v2, q_v1),
                    "v2_vs_v1_k":     max_abs(k_v2, k_v1),
                }
                passed = (
                    passes_tolerance(diffs["v2_vs_hffp32_q"], dtype) and
                    passes_tolerance(diffs["v2_vs_hffp32_k"], dtype) and
                    not torch.isnan(q_v2).any().item() and
                    not torch.isnan(k_v2).any().item()
                )
                cases.append({
                    "shape_label": label,
                    "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                              "seq_len": seq_len, "head_dim": head_dim,
                              "G_ratio": n_q // n_kv},
                    "dtype": str(dtype),
                    "cos_batched": cos_batched,
                    "tolerance": TOLERANCES[str(dtype)],
                    "diffs": diffs,
                    "has_nan_q": bool(torch.isnan(q_v2).any().item()),
                    "has_nan_k": bool(torch.isnan(k_v2).any().item()),
                    "passed": bool(passed),
                })

    n_passed = sum(1 for c in cases if c["passed"])
    return {
        "summary": {"total": len(cases), "passed": n_passed, "failed": len(cases) - n_passed},
        "reference": "HF apply_rotary_pos_emb in fp32, cast to input dtype",
        "cases": cases,
    }


def test_backward_correctness() -> dict[str, Any]:
    """Forge V2 backward gradients vs HF fp32 autograd."""
    cases = []
    shape_subset = [
        ("demo_tiny",      2,  4,  2,   16,  64),
        ("qwen3_8b_short", 4, 32,  8,  512, 128),
        ("mqa_extreme",    2,  8,  1, 1024, 128),
        ("mha_no_gqa",     2, 16, 16, 1024, 128),
    ]
    for label, batch, n_q, n_kv, seq_len, head_dim in shape_subset:
        for dtype in [torch.bfloat16, torch.float32]:
            q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype)
            grad_q = torch.randn(batch, n_q,  seq_len, head_dim, device="cuda", dtype=dtype)
            grad_k = torch.randn(batch, n_kv, seq_len, head_dim, device="cuda", dtype=dtype)

            # HF in fp32
            q_hf = q.float().clone().requires_grad_(True)
            k_hf = k.float().clone().requires_grad_(True)
            out_q_hf, out_k_hf = hf_apply_rotary_pos_emb(q_hf, k_hf, cos.float(), sin.float())
            loss_hf = (out_q_hf * grad_q.float()).sum() + (out_k_hf * grad_k.float()).sum()
            loss_hf.backward()
            out_q_hf_cast = out_q_hf.detach().to(dtype)
            out_k_hf_cast = out_k_hf.detach().to(dtype)
            dq_hf_cast = q_hf.grad.to(dtype)
            dk_hf_cast = k_hf.grad.to(dtype)

            # Forge V2
            q_v2 = q.clone().requires_grad_(True)
            k_v2 = k.clone().requires_grad_(True)
            out_q_v2, out_k_v2 = forge_v2_apply(q_v2, k_v2, cos, sin)
            loss_v2 = (out_q_v2 * grad_q).sum() + (out_k_v2 * grad_k).sum()
            loss_v2.backward()

            fwd_diff_q  = max_abs(out_q_v2,  out_q_hf_cast)
            fwd_diff_k  = max_abs(out_k_v2,  out_k_hf_cast)
            bwd_diff_dq = max_abs(q_v2.grad, dq_hf_cast)
            bwd_diff_dk = max_abs(k_v2.grad, dk_hf_cast)

            passed = all(passes_tolerance(d, dtype)
                         for d in [fwd_diff_q, fwd_diff_k, bwd_diff_dq, bwd_diff_dk])

            cases.append({
                "shape_label": label,
                "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                          "seq_len": seq_len, "head_dim": head_dim,
                          "G_ratio": n_q // n_kv},
                "dtype": str(dtype),
                "tolerance": TOLERANCES[str(dtype)],
                "forward_diff":  {"q":  fwd_diff_q,  "k":  fwd_diff_k},
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
    """torch.autograd.gradcheck on fp64 for V2 + manual one-hot math check."""
    batch, n_q, n_kv, seq_len, head_dim = 2, 4, 2, 8, 32
    q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, torch.float64)

    # Manual one-hot check
    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    out_q, out_k = forge_v2_apply(q1, k1, cos, sin)
    pos_s, pos_d = 1, 0
    grad_q_oh = torch.zeros_like(out_q); grad_q_oh[0, 0, pos_s, pos_d] = 1.0
    grad_k_oh = torch.zeros_like(out_k)
    ((out_q * grad_q_oh).sum() + (out_k * grad_k_oh).sum()).backward()
    manual_check = {
        "one_hot_position": {"s": pos_s, "d": pos_d},
        "expected_cos":     cos[0, pos_s, pos_d].item(),
        "expected_neg_sin": -sin[0, pos_s, pos_d].item(),
        "got_lo":           q1.grad[0, 0, pos_s, pos_d].item(),
        "got_hi":           q1.grad[0, 0, pos_s, pos_d + head_dim // 2].item(),
    }
    manual_check["lo_diff"] = abs(manual_check["got_lo"] - manual_check["expected_cos"])
    manual_check["hi_diff"] = abs(manual_check["got_hi"] - manual_check["expected_neg_sin"])
    manual_check["passed"]  = manual_check["lo_diff"] < 1e-5 and manual_check["hi_diff"] < 1e-5

    # gradcheck with Triton-friendly tolerances
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    try:
        passed = torch.autograd.gradcheck(
            ForgeRoPEv2Function.apply,
            (q2, k2, cos, sin),
            eps=1e-3, atol=1e-2, rtol=1e-2, nondet_tol=1e-3,
            check_undefined_grad=False, check_batched_grad=False,
        )
        return {
            "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                      "seq_len": seq_len, "head_dim": head_dim, "G_ratio": 2},
            "dtype": "torch.float64",
            "gradcheck_settings": {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2, "nondet_tol": 1e-3},
            "manual_check": manual_check,
            "passed": bool(passed),
            "error": None,
        }
    except Exception as e:
        return {
            "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                      "seq_len": seq_len, "head_dim": head_dim, "G_ratio": 2},
            "dtype": "torch.float64",
            "gradcheck_settings": {"eps": 1e-3, "atol": 1e-2, "rtol": 1e-2, "nondet_tol": 1e-3},
            "manual_check": manual_check,
            "passed": False,
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

TIMING_SHAPES = [
    ("qwen3_8b_short", 4, 32, 8,  512, 128),
    ("qwen3_8b_train", 2, 32, 8, 2048, 128),
    ("mqa_extreme",    2,  8, 1, 1024, 128),
    ("mha_no_gqa",     2, 16, 16, 1024, 128),
]
TIMING_DTYPES = [torch.bfloat16, torch.float16]


def test_forward_timing() -> dict[str, Any]:
    """Forward-only timing across all kernels."""
    cases = []
    for label, batch, n_q, n_kv, seq_len, head_dim in TIMING_SHAPES:
        for dtype in TIMING_DTYPES:
            q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype)

            kernels = {
                "pytorch_ref":      lambda: hf_apply_rotary_pos_emb(q, k, cos, sin),
                "liger":            lambda: liger.apply_rope(q.clone(), k.clone(), cos, sin),
                "unsloth_default":  lambda: unsloth.apply_rope(q.clone(), k.clone(), cos, sin),
                "unsloth_fused_qk": lambda: unsloth.apply_rope_qk_fused(q.clone(), k.clone(), cos, sin),
                "forge_v1":         lambda: forge_v1_apply(q, k, cos, sin),
                "forge_v2":         lambda: forge_v2_apply(q, k, cos, sin),
            }

            timings_ms = {}
            for name, fn in kernels.items():
                try:
                    timings_ms[name] = float(triton.testing.do_bench(fn, warmup=25, rep=100))
                except Exception as e:
                    timings_ms[name] = None
                    timings_ms[f"{name}_error"] = f"{type(e).__name__}: {e}"

            ref = timings_ms.get("pytorch_ref")
            speedups = {}
            for k_name in ("liger", "unsloth_default", "unsloth_fused_qk", "forge_v1", "forge_v2"):
                t = timings_ms.get(k_name)
                speedups[k_name] = (ref / t) if (ref and t) else None

            # V2 vs V1 ratio
            v1 = timings_ms.get("forge_v1")
            v2 = timings_ms.get("forge_v2")
            v2_vs_v1 = (v1 / v2) if (v1 and v2) else None

            # HBM bandwidth for V2
            element_size = torch.tensor([], dtype=dtype).element_size()
            q_bytes = batch * n_q  * seq_len * head_dim * element_size * 2
            k_bytes = batch * n_kv * seq_len * head_dim * element_size * 2
            cos_sin_bytes = 2 * seq_len * head_dim * element_size
            total_bytes = q_bytes + k_bytes + cos_sin_bytes
            v2_bw_gbps = (total_bytes / 1e9) / (v2 / 1000.0) if v2 else None

            cases.append({
                "shape_label": label,
                "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                          "seq_len": seq_len, "head_dim": head_dim, "G_ratio": n_q // n_kv},
                "dtype": str(dtype),
                "median_ms": timings_ms,
                "speedup_vs_pytorch": speedups,
                "v2_speedup_vs_v1": v2_vs_v1,
                "total_hbm_traffic_bytes": total_bytes,
                "forge_v2_bandwidth_gbps": v2_bw_gbps,
            })
    return {"cases": cases}


def test_backward_timing() -> dict[str, Any]:
    """Backward-only timing for kernels that expose autograd integration.

    PyTorch and Liger and Forge V1/V2 are tested via autograd. Unsloth has different
    backward semantics (in-place) — we use the QK kernel path which has a proper bwd.
    """
    cases = []
    for label, batch, n_q, n_kv, seq_len, head_dim in TIMING_SHAPES:
        for dtype in TIMING_DTYPES:
            q, k, cos, sin = gen_inputs(batch, n_q, n_kv, seq_len, head_dim, dtype)

            def bench_bwd(apply_fn):
                """Time backward + zero_grad for an autograd-wrapped apply_fn."""
                qg = q.clone().detach().requires_grad_(True)
                kg = k.clone().detach().requires_grad_(True)
                out_q, out_k = apply_fn(qg, kg, cos, sin)
                grad_q = torch.ones_like(out_q)
                grad_k = torch.ones_like(out_k)
                def step():
                    qg.grad = None; kg.grad = None
                    torch.autograd.backward((out_q, out_k), (grad_q, grad_k),
                                            retain_graph=True)
                return float(triton.testing.do_bench(step, warmup=25, rep=100))

            timings_ms: dict[str, Any] = {}
            try:
                timings_ms["forge_v1"] = bench_bwd(forge_v1_apply)
            except Exception as e:
                timings_ms["forge_v1"] = None
                timings_ms["forge_v1_error"] = str(e)
            try:
                timings_ms["forge_v2"] = bench_bwd(forge_v2_apply)
            except Exception as e:
                timings_ms["forge_v2"] = None
                timings_ms["forge_v2_error"] = str(e)
            try:
                timings_ms["liger"] = bench_bwd(lambda qq, kk, c, s: liger.apply_rope(qq, kk, c, s))
            except Exception as e:
                timings_ms["liger"] = None
                timings_ms["liger_error"] = str(e)
            try:
                timings_ms["pytorch_ref"] = bench_bwd(hf_apply_rotary_pos_emb)
            except Exception as e:
                timings_ms["pytorch_ref"] = None
                timings_ms["pytorch_ref_error"] = str(e)

            v1 = timings_ms.get("forge_v1")
            v2 = timings_ms.get("forge_v2")
            v2_vs_v1 = (v1 / v2) if (v1 and v2) else None

            cases.append({
                "shape_label": label,
                "shape": {"batch": batch, "n_q": n_q, "n_kv": n_kv,
                          "seq_len": seq_len, "head_dim": head_dim, "G_ratio": n_q // n_kv},
                "dtype": str(dtype),
                "median_ms": timings_ms,
                "v2_speedup_vs_v1": v2_vs_v1,
            })
    return {"cases": cases}


# ---------------------------------------------------------------------------
# Env / config
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
        "version": "v2",
        "grid": "(batch * seq_len, n_kv_heads)",
        "head_grouping": "G = n_q // n_kv (exact division required)",
        "tile_shape_q": "(G_BLOCK, BLOCK_SIZE)",
        "tile_shape_k": "(BLOCK_SIZE,) single head per program",
        "block_size": "next_power_of_2(head_dim // 2)",
        "g_block": "next_power_of_2(G)",
        "num_warps": 4,
        "backward_strategy": "shared kernel via BACKWARD_PASS constexpr; sin negated after load",
        "fp32_accumulation": True,
        "gqa_handling": "implicit via G grouping — no mask branch",
        "save_for_backward": ["cos", "sin"],
        "out_of_place": True,
        "autotune": False,
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_markdown_summary(results: dict[str, Any], path: Path) -> None:
    env = results["environment"]
    fwd = results["results"].get("forward_correctness", {}).get("summary", {})
    bwd = results["results"].get("backward_correctness", {}).get("summary", {})
    grad = results["results"].get("gradcheck_fp64", {})
    timing_fwd = results["results"].get("forward_timing", {}).get("cases", [])
    timing_bwd = results["results"].get("backward_timing", {}).get("cases", [])

    def fmt(v, places=4):
        return f"{v:.{places}f}" if isinstance(v, (int, float)) else "n/a"

    lines = []
    lines.append("# ForgeRoPE V2 — Test + Benchmark Results")
    lines.append("")
    lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} "
                 f"(compute {env.get('cuda_device_capability', 'n/a')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}")
    lines.append(f"**Kernel design:** grid `(b·s, n_kv)`, G=n_q//n_kv Q heads per program, "
                 "1 K head per program, cos/sin loaded once and reused")
    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    lines.append("| Suite | Passed | Total |")
    lines.append("|---|---|---|")
    lines.append(f"| Forward correctness | {fwd.get('passed', 0)} | {fwd.get('total', 0)} |")
    lines.append(f"| Backward correctness | {bwd.get('passed', 0)} | {bwd.get('total', 0)} |")
    lines.append(f"| Gradcheck (fp64) | {'PASS' if grad.get('passed') else 'FAIL'} | 1 |")
    lines.append("")
    if grad.get("error"):
        lines.append(f"**Gradcheck error:** `{grad['error']}`")
        lines.append("")

    lines.append("## Forward timing (median ms)")
    lines.append("")
    lines.append("| Shape (b×nq/nkv×s×hd, G) | dtype | PyTorch | Liger | UnslDef | UnslQK | **V1** | **V2** | V2/V1 | V2 vs PT |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in timing_fwd:
        t = c["median_ms"]
        sh = c["shape"]
        sp_v2 = c["speedup_vs_pytorch"].get("forge_v2")
        v2v1 = c["v2_speedup_vs_v1"]
        lines.append(
            f"| {c['shape_label']} ({sh['batch']}×{sh['n_q']}/{sh['n_kv']}×{sh['seq_len']}×{sh['head_dim']}, G={sh['G_ratio']}) "
            f"| {c['dtype']} "
            f"| {fmt(t.get('pytorch_ref'))} "
            f"| {fmt(t.get('liger'))} "
            f"| {fmt(t.get('unsloth_default'))} "
            f"| {fmt(t.get('unsloth_fused_qk'))} "
            f"| {fmt(t.get('forge_v1'))} "
            f"| **{fmt(t.get('forge_v2'))}** "
            f"| {fmt(v2v1, 2)}× "
            f"| {fmt(sp_v2, 2)}× |"
        )
    lines.append("")

    lines.append("## Backward timing (median ms)")
    lines.append("")
    lines.append("| Shape (G) | dtype | PyTorch | Liger | **V1** | **V2** | V2/V1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in timing_bwd:
        t = c["median_ms"]
        sh = c["shape"]
        v2v1 = c["v2_speedup_vs_v1"]
        lines.append(
            f"| {c['shape_label']} (G={sh['G_ratio']}) "
            f"| {c['dtype']} "
            f"| {fmt(t.get('pytorch_ref'))} "
            f"| {fmt(t.get('liger'))} "
            f"| {fmt(t.get('forge_v1'))} "
            f"| **{fmt(t.get('forge_v2'))}** "
            f"| {fmt(v2v1, 2)}× |"
        )
    lines.append("")

    lines.append("## HBM bandwidth utilization (Forge V2)")
    lines.append("")
    lines.append("| Shape | dtype | Traffic (MB) | V2 time (ms) | Achieved BW (GB/s) |")
    lines.append("|---|---|---|---|---|")
    for c in timing_fwd:
        bw = c["forge_v2_bandwidth_gbps"]
        lines.append(
            f"| {c['shape_label']} | {c['dtype']} "
            f"| {c['total_hbm_traffic_bytes'] / 1e6:.1f} "
            f"| {fmt(c['median_ms'].get('forge_v2'))} "
            f"| {fmt(bw, 0)} |"
        )

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results: dict[str, Any] = {
        "version": "v2",
        "implementation_file": "kernels/rope/forge_rope_v2.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        "kernel_config": kernel_config(),
        "results": {},
        "errors": [],
    }

    results["results"]["import_smoke"] = {"passed": True, "error": None}

    sections = [
        ("forward_correctness",  test_forward_correctness),
        ("backward_correctness", test_backward_correctness),
        ("gradcheck_fp64",       test_gradcheck),
        ("forward_timing",       test_forward_timing),
        ("backward_timing",      test_backward_timing),
    ]
    for i, (name, fn) in enumerate(sections, start=1):
        print(f"[{i}/{len(sections)}] {name} ...", flush=True)
        try:
            results["results"][name] = fn()
        except Exception as e:
            results["results"][name] = {"error": str(e), "traceback": traceback.format_exc()}
            results["errors"].append({"phase": name, "error": str(e)})

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "v2_results.json"
    md_path = out_dir / "v2_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    try:
        write_markdown_summary(results, md_path)
    except Exception as e:
        results["errors"].append({"phase": "summary_md", "error": str(e),
                                  "traceback": traceback.format_exc()})
        json_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")

    print("\n=== ForgeRoPE V2 — short summary ===")
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

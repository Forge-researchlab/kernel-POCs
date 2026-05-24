"""ForgeRMSNorm V2 test runner.

Runs:
  1. Environment fingerprint
  2. Import smoke
  3. Forward correctness (Forge V2 vs torch_rmsnorm_reference_v2 oracle + Liger + Unsloth)
     - Covers both Llama path (offset=0, casting_mode='llama') and Gemma path
       (offset=1, casting_mode='gemma') across Qwen3 + Gemma + edge shapes.
  4. Backward correctness (Forge V2 dx, dw vs autograd of oracle)
  5. Gradcheck on fp64 — BOTH offset=0.0 (llama) AND offset=1.0 (gemma)
  6. Forward + backward timing vs PyTorch eager + Liger + Unsloth + Forge V1

Emits:
  - kernels/rmsnorm/tests/results/v2_results.json   (machine-readable)
  - kernels/rmsnorm/tests/results/v2_summary.md     (human-readable)

Usage:
  python kernels/rmsnorm/tests/test_v2.py
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

from kernels.rmsnorm.forge_rmsnorm_v2 import (
    apply_rmsnorm_v2,
    ForgeRMSNormv2Function,
    torch_rmsnorm_reference_v2,
)
from kernels.rmsnorm.forge_rmsnorm_v1 import rmsnorm_v1
from kernels.rmsnorm.baselines import liger, unsloth


# ----------------------------------------------------------------------------
# Tolerances per dtype
# ----------------------------------------------------------------------------
TOLERANCES = {
    # atol covers small-magnitude values; rtol scales tolerance with the
    # reference magnitude — backward dW reduces over n_rows so values reach
    # O(sqrt(n_rows)) ≈ 45 at qwen3_8b_short, where bf16 ULP ≈ 0.4 ≈ 8×atol.
    "torch.bfloat16": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float16":  {"atol": 2e-2, "rtol": 2e-2},
    "torch.float32":  {"atol": 1e-4, "rtol": 1e-5},
    "torch.float64":  {"atol": 1e-7, "rtol": 1e-7},
}

def passes(diff: float, ref_max: float, dtype: torch.dtype) -> bool:
    tol = TOLERANCES.get(str(dtype), {"atol": 1e-3, "rtol": 1e-3})
    return diff <= tol["atol"] + tol["rtol"] * abs(ref_max)


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def ref_max(t):
    return t.float().abs().max().item()


# ----------------------------------------------------------------------------
# Shape matrix — (label, batch, seq_len, hidden)
# ----------------------------------------------------------------------------
SHAPES = [
    ("tiny",                1,    8,    64),     # gradcheck-friendly
    ("qwen25_0p5b",         2,  512,   896),     # Qwen2.5-0.5B hidden
    ("qwen3_8b_short",      4,  512,  4096),
    ("qwen3_8b_train",      2, 2048,  4096),     # headline shape
    ("gemma2_2b",           2, 2048,  2304),
    ("gemma2_9b",           2, 2048,  3584),
    ("non_pow2",            4,  128,  4097),     # masked tail
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# (offset, casting_mode, weight_init_strategy) — pair offsets with the casting
# mode HF actually uses for that family. "near_zero" weights mimic Gemma init.
OFFSET_CASES = [
    (0.0, "llama", "ones_plus_noise"),
    (1.0, "gemma", "near_zero"),
]


def _make_inputs(batch, seq, hidden, dtype, device, weight_init, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype, generator=g)
    if weight_init == "ones_plus_noise":
        w = torch.ones(hidden, device=device, dtype=dtype) + 0.1 * torch.randn(
            hidden, device=device, dtype=dtype, generator=g
        )
    elif weight_init == "near_zero":
        w = 0.05 * torch.randn(hidden, device=device, dtype=dtype, generator=g)
    else:
        w = torch.randn(hidden, device=device, dtype=dtype, generator=g)
    return x.contiguous(), w.contiguous()


# ----------------------------------------------------------------------------
# (3) Forward correctness
# ----------------------------------------------------------------------------
def test_forward_correctness() -> dict[str, Any]:
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode, init in OFFSET_CASES:
                if h > 16384:
                    continue
                x, w = _make_inputs(b, s, h, dtype, "cuda", init)

                y_forge = apply_rmsnorm_v2(x, w, eps=1e-6, offset=offset, casting_mode=mode)
                y_oracle = torch_rmsnorm_reference_v2(x, w, 1e-6, offset, mode)
                y_liger = liger.apply_rmsnorm(x, w, 1e-6, offset, mode)
                y_unsloth = unsloth.apply_rmsnorm(x, w, 1e-6, offset)

                diffs = {
                    "forge_vs_oracle":  max_abs(y_forge, y_oracle),
                    "forge_vs_liger":   max_abs(y_forge, y_liger),
                    "forge_vs_unsloth": max_abs(y_forge, y_unsloth),
                }
                passed = (
                    passes(diffs["forge_vs_oracle"], ref_max(y_oracle), dtype)
                    and not torch.isnan(y_forge).any().item()
                    and y_forge.shape == x.shape
                    and y_forge.dtype == dtype
                )
                cases.append({
                    "shape_label": label,
                    "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype),
                    "offset": offset,
                    "casting_mode": mode,
                    "diffs": diffs,
                    "tolerance": TOLERANCES[str(dtype)],
                    "passed": bool(passed),
                })
    n_pass = sum(1 for c in cases if c["passed"])
    return {
        "summary": {"total": len(cases), "passed": n_pass, "failed": len(cases) - n_pass},
        "reference": "torch_rmsnorm_reference_v2 (fp32 reduction, offset-aware)",
        "cases": cases,
    }


# ----------------------------------------------------------------------------
# (4) Backward correctness — Forge V2 vs autograd of oracle
# ----------------------------------------------------------------------------
def test_backward_correctness() -> dict[str, Any]:
    cases = []
    shape_subset = [
        ("tiny",            1,    8,    64),
        ("qwen3_8b_short",  4,  512, 4096),
        ("gemma2_2b",       2, 2048, 2304),
    ]
    for label, b, s, h in shape_subset:
        for dtype in [torch.bfloat16, torch.float32]:
            for offset, mode, init in OFFSET_CASES:
                x, w = _make_inputs(b, s, h, dtype, "cuda", init, seed=1)
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)

                # Forge
                xf = x.clone().requires_grad_(True)
                wf = w.clone().requires_grad_(True)
                y_f = apply_rmsnorm_v2(xf, wf, 1e-6, offset, mode)
                y_f.backward(grad)

                # Oracle autograd in fp32 reference
                xo = x.float().clone().requires_grad_(True)
                wo = w.float().clone().requires_grad_(True)
                y_o = torch_rmsnorm_reference_v2(xo, wo, 1e-6, offset, mode)
                y_o.backward(grad.float())

                dx_diff = max_abs(xf.grad, xo.grad.to(dtype))
                dw_diff = max_abs(wf.grad, wo.grad.to(dtype))
                fwd_diff = max_abs(y_f, y_o.to(dtype))

                passed = (passes(dx_diff,  ref_max(xo.grad), dtype)
                          and passes(dw_diff,  ref_max(wo.grad), dtype)
                          and passes(fwd_diff, ref_max(y_o),     dtype))
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype),
                    "offset": offset, "casting_mode": mode,
                    "fwd_diff": fwd_diff,
                    "dx_diff": dx_diff,
                    "dw_diff": dw_diff,
                    "tolerance": TOLERANCES[str(dtype)],
                    "passed": bool(passed),
                })
    n_pass = sum(1 for c in cases if c["passed"])
    return {
        "summary": {"total": len(cases), "passed": n_pass, "failed": len(cases) - n_pass},
        "cases": cases,
    }


# ----------------------------------------------------------------------------
# (5) Gradcheck at fp64 — both offsets
# ----------------------------------------------------------------------------
def test_gradcheck_fp64() -> dict[str, Any]:
    out = {}
    for offset, mode, init in OFFSET_CASES:
        try:
            torch.manual_seed(42)
            xs = torch.randn(2, 8, 32, device="cuda", dtype=torch.float64, requires_grad=True)
            if init == "near_zero":
                ws_data = 0.5 * (torch.rand(32, device="cuda", dtype=torch.float64) - 0.5)
            else:
                ws_data = torch.randn(32, device="cuda", dtype=torch.float64) * 0.5
            ws = ws_data.detach().clone().requires_grad_(True)
            ok = torch.autograd.gradcheck(
                lambda xx, ww: ForgeRMSNormv2Function.apply(xx, ww, 1e-6, offset, mode),
                (xs, ws),
                eps=1e-6, atol=1e-5, rtol=1e-4,
                fast_mode=True,
            )
            out[f"offset={offset}_mode={mode}"] = {"passed": bool(ok)}
        except Exception as e:
            out[f"offset={offset}_mode={mode}"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    out["passed"] = all(v.get("passed", False) for v in out.values())
    return out


# ----------------------------------------------------------------------------
# (6) Forward + backward timing
# ----------------------------------------------------------------------------
def test_timing() -> dict[str, Any]:
    cases = []
    for label, b, s, h in SHAPES:
        if h > 16384:
            continue
        for dtype in [torch.bfloat16, torch.float16]:
            for offset, mode, init in OFFSET_CASES:
                x, w = _make_inputs(b, s, h, dtype, "cuda", init, seed=2)

                # Forward closures
                call_pytorch = lambda x=x, w=w: torch_rmsnorm_reference_v2(x, w, 1e-6, offset, mode)
                call_liger   = lambda x=x, w=w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode)
                call_unsloth = lambda x=x, w=w: unsloth.apply_rmsnorm(x, w, 1e-6, offset)
                call_v1      = lambda x=x, w=w: rmsnorm_v1(x, w, 1e-6) if offset == 0.0 else None
                call_v2      = lambda x=x, w=w: apply_rmsnorm_v2(x, w, 1e-6, offset, mode)

                timings = {}
                for name, fn in [
                    ("pytorch_ref",  call_pytorch),
                    ("liger",        call_liger),
                    ("unsloth",      call_unsloth),
                    ("forge_v1",     call_v1),
                    ("forge_v2",     call_v2),
                ]:
                    if fn is None:
                        continue
                    try:
                        ms = triton.testing.do_bench(fn, warmup=10, rep=50)
                        timings[name] = float(ms)
                    except Exception as e:
                        timings[name] = None
                        timings[f"{name}_error"] = f"{type(e).__name__}: {e}"

                ref = timings.get("pytorch_ref")
                speedups = {
                    k: (ref / t) if (t and ref) else None
                    for k, t in timings.items() if not k.endswith("_error")
                }
                # HBM bandwidth (read x + read w + write y; rstd is small)
                el = torch.tensor([], dtype=dtype).element_size()
                total_bytes = b * s * h * el * 2 + h * el
                forge_ms = timings.get("forge_v2")
                bw = (total_bytes / 1e9) / (forge_ms / 1000.0) if forge_ms else None
                cases.append({
                    "shape_label": label,
                    "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype),
                    "offset": offset, "casting_mode": mode,
                    "median_ms": timings,
                    "speedup_vs_pytorch": speedups,
                    "forge_v2_bandwidth_gbps": bw,
                })
    return {"cases": cases}


# ----------------------------------------------------------------------------
# Environment + main
# ----------------------------------------------------------------------------
def collect_environment() -> dict[str, Any]:
    cuda_ok = torch.cuda.is_available()
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda_ok,
    }
    if cuda_ok:
        idx = torch.cuda.current_device()
        env["cuda_device_name"] = torch.cuda.get_device_name(idx)
        env["cuda_device_capability"] = list(torch.cuda.get_device_capability(idx))
        env["cuda_runtime_version"] = torch.version.cuda
        env["sm_count"] = torch.cuda.get_device_properties(idx).multi_processor_count
    return env


def kernel_config() -> dict[str, Any]:
    return {
        "version": "v2",
        "grid": "(n_rows,)",
        "block_size": "next_power_of_2(n_cols), capped at 131072",
        "num_warps_heuristic": "4/8/16/32 by BLOCK_SIZE bucket",
        "offset": "tl.constexpr",
        "casting_mode": "tl.constexpr (0=LLAMA, 1=GEMMA, 2=NONE)",
        "acc_dtype": "fp32 for bf16/fp16/fp32 inputs; fp64 for fp64 (gradcheck)",
        "backward_strategy": "SM-proportional dW partials + Python sum(0)",
        "save_for_backward": ["x_2d", "weight_1d", "rstd"],
        "autotune": False,
    }


def write_markdown_summary(results: dict[str, Any], path: Path) -> None:
    env = results["environment"]
    fwd = results["results"].get("forward_correctness", {}).get("summary", {"passed": 0, "total": 0})
    bwd = results["results"].get("backward_correctness", {}).get("summary", {"passed": 0, "total": 0})
    gc = results["results"].get("gradcheck_fp64", {})
    timing_cases = results["results"].get("timing", {}).get("cases", [])

    lines = []
    lines.append("# ForgeRMSNorm V2 — Test Results Summary")
    lines.append("")
    lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} "
                 f"(compute {env.get('cuda_device_capability', 'n/a')}, SMs={env.get('sm_count', '?')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}")
    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    lines.append("| Suite | Passed | Total |")
    lines.append("|---|---|---|")
    lines.append(f"| Forward correctness  | {fwd['passed']} | {fwd['total']} |")
    lines.append(f"| Backward correctness | {bwd['passed']} | {bwd['total']} |")
    lines.append(
        f"| fp64 gradcheck (llama)| {'✓' if gc.get('offset=0.0_mode=llama', {}).get('passed') else '✗'} | 1 |"
    )
    lines.append(
        f"| fp64 gradcheck (gemma)| {'✓' if gc.get('offset=1.0_mode=gemma', {}).get('passed') else '✗'} | 1 |"
    )
    lines.append("")
    lines.append("## Forward + backward timing (median, lower = better)")
    lines.append("")
    lines.append("| Shape | dtype | offset | PT (ms) | Liger (ms) | Unsloth (ms) | V1 (ms) | **V2 (ms)** | V2 speedup vs PT | V2 BW (GB/s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
    for c in timing_cases:
        sh = c["shape"]
        t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v2")
        sp_str = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
        bw = c.get("forge_v2_bandwidth_gbps")
        bw_str = f"{bw:.0f}" if isinstance(bw, (int, float)) else "—"
        lines.append(
            f"| {c['shape_label']} ({sh['batch']}×{sh['seq']}×{sh['hidden']}) | "
            f"{c['dtype']} | {c['offset']} | {fmt(t.get('pytorch_ref'))} | "
            f"{fmt(t.get('liger'))} | {fmt(t.get('unsloth'))} | "
            f"{fmt(t.get('forge_v1'))} | **{fmt(t.get('forge_v2'))}** | {sp_str} | {bw_str} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    out_dir = _REPO_ROOT / "kernels" / "rmsnorm" / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = collect_environment()
    if not env["cuda_available"]:
        print("CUDA not available — Triton kernels require a GPU. Aborting.")
        return 1

    print("=" * 70)
    print("ForgeRMSNorm V2 test runner")
    print("=" * 70)
    print(f"Device: {env.get('cuda_device_name', '?')} (SMs={env.get('sm_count', '?')})")
    print(f"Torch / Triton: {env['torch']} / {env['triton']}")
    print()

    results = {
        "kernel": kernel_config(),
        "environment": env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": {},
    }

    suites = [
        ("forward_correctness",  test_forward_correctness),
        ("backward_correctness", test_backward_correctness),
        ("gradcheck_fp64",       test_gradcheck_fp64),
        ("timing",               test_timing),
    ]
    overall_ok = True
    for name, fn in suites:
        try:
            print(f">> running {name} ...", flush=True)
            results["results"][name] = fn()
            s = results["results"][name].get("summary")
            if s:
                print(f"   {name}: {s['passed']}/{s['total']} passed")
                if s.get("failed"):
                    overall_ok = False
            elif name == "gradcheck_fp64":
                gc_ok = results["results"][name].get("passed", False)
                print(f"   gradcheck_fp64: {'PASS' if gc_ok else 'FAIL'}")
                if not gc_ok:
                    overall_ok = False
        except Exception as e:
            results["results"][name] = {"error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc()}
            print(f"   ERROR in {name}: {e}", flush=True)
            overall_ok = False

    json_path = out_dir / "v2_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    md_path = out_dir / "v2_summary.md"
    write_markdown_summary(results, md_path)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

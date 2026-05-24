"""ForgeRMSNorm V3 test runner.

V3 = V2 body + @triton.autotune over (num_warps, num_stages). Identical test
shape matrix and tolerance scheme as test_v2.py; the only extra is the autotune
cache warm-up check (first call is slow due to config compilation, subsequent
calls are cached).

Emits:
  - kernels/rmsnorm/tests/results/v3_results.json
  - kernels/rmsnorm/tests/results/v3_summary.md
"""
from __future__ import annotations

import json
import platform
import sys
import time
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

from kernels.rmsnorm.forge_rmsnorm_v3 import (
    apply_rmsnorm_v3,
    ForgeRMSNormv3Function,
    torch_rmsnorm_reference_v3,
)
from kernels.rmsnorm.forge_rmsnorm_v2 import apply_rmsnorm_v2
from kernels.rmsnorm.forge_rmsnorm_v1 import rmsnorm_v1
from kernels.rmsnorm.baselines import liger, unsloth


TOLERANCES = {
    "torch.bfloat16": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float16":  {"atol": 2e-2, "rtol": 2e-2},
    "torch.float32":  {"atol": 1e-4, "rtol": 1e-5},
    "torch.float64":  {"atol": 1e-7, "rtol": 1e-7},
}
def passes(d, r, dt):
    tol = TOLERANCES.get(str(dt), {"atol": 1e-3, "rtol": 1e-3})
    return d <= tol["atol"] + tol["rtol"] * abs(r)
def max_abs(a, b): return (a.float() - b.float()).abs().max().item()
def ref_max(t): return t.float().abs().max().item()


SHAPES = [
    ("tiny",                1,    8,    64),
    ("qwen25_0p5b",         2,  512,   896),
    ("qwen3_8b_short",      4,  512,  4096),
    ("qwen3_8b_train",      2, 2048,  4096),
    ("gemma2_2b",           2, 2048,  2304),
    ("gemma2_9b",           2, 2048,  3584),
    ("non_pow2",            4,  128,  4097),
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
OFFSET_CASES = [(0.0, "llama", "ones_plus_noise"), (1.0, "gemma", "near_zero")]


def _make_inputs(b, s, h, dt, device, init, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(b, s, h, device=device, dtype=dt, generator=g)
    if init == "ones_plus_noise":
        w = torch.ones(h, device=device, dtype=dt) + 0.1 * torch.randn(h, device=device, dtype=dt, generator=g)
    elif init == "near_zero":
        w = 0.05 * torch.randn(h, device=device, dtype=dt, generator=g)
    else:
        w = torch.randn(h, device=device, dtype=dt, generator=g)
    return x.contiguous(), w.contiguous()


def test_forward_correctness():
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode, init in OFFSET_CASES:
                if h > 16384: continue
                x, w = _make_inputs(b, s, h, dtype, "cuda", init)
                y = apply_rmsnorm_v3(x, w, 1e-6, offset, mode)
                oracle = torch_rmsnorm_reference_v3(x, w, 1e-6, offset, mode)
                liger_y = liger.apply_rmsnorm(x, w, 1e-6, offset, mode)
                unsloth_y = unsloth.apply_rmsnorm(x, w, 1e-6, offset)
                diffs = {
                    "forge_vs_oracle":  max_abs(y, oracle),
                    "forge_vs_liger":   max_abs(y, liger_y),
                    "forge_vs_unsloth": max_abs(y, unsloth_y),
                }
                ok = (passes(diffs["forge_vs_oracle"], ref_max(oracle), dtype)
                      and not torch.isnan(y).any().item()
                      and y.shape == x.shape and y.dtype == dtype)
                cases.append({"shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                              "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                              "diffs": diffs, "tolerance": TOLERANCES[str(dtype)], "passed": bool(ok)})
    n = sum(1 for c in cases if c["passed"])
    return {"summary": {"total": len(cases), "passed": n, "failed": len(cases)-n}, "cases": cases}


def test_backward_correctness():
    cases = []
    subset = [("tiny", 1, 8, 64), ("qwen3_8b_short", 4, 512, 4096), ("gemma2_2b", 2, 2048, 2304)]
    for label, b, s, h in subset:
        for dtype in [torch.bfloat16, torch.float32]:
            for offset, mode, init in OFFSET_CASES:
                x, w = _make_inputs(b, s, h, dtype, "cuda", init, seed=1)
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)
                xf = x.clone().requires_grad_(True); wf = w.clone().requires_grad_(True)
                y_f = apply_rmsnorm_v3(xf, wf, 1e-6, offset, mode); y_f.backward(grad)
                xo = x.float().clone().requires_grad_(True); wo = w.float().clone().requires_grad_(True)
                y_o = torch_rmsnorm_reference_v3(xo, wo, 1e-6, offset, mode); y_o.backward(grad.float())
                dx_d = max_abs(xf.grad, xo.grad.to(dtype))
                dw_d = max_abs(wf.grad, wo.grad.to(dtype))
                fwd_d = max_abs(y_f, y_o.to(dtype))
                ok = (passes(dx_d,  ref_max(xo.grad), dtype)
                      and passes(dw_d,  ref_max(wo.grad), dtype)
                      and passes(fwd_d, ref_max(y_o),     dtype))
                cases.append({"shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                              "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                              "fwd_diff": fwd_d, "dx_diff": dx_d, "dw_diff": dw_d,
                              "tolerance": TOLERANCES[str(dtype)], "passed": bool(ok)})
    n = sum(1 for c in cases if c["passed"])
    return {"summary": {"total": len(cases), "passed": n, "failed": len(cases)-n}, "cases": cases}


def test_gradcheck_fp64():
    out = {}
    for offset, mode, init in OFFSET_CASES:
        try:
            torch.manual_seed(42)
            xs = torch.randn(2, 8, 32, device="cuda", dtype=torch.float64, requires_grad=True)
            ws_data = (0.5 * (torch.rand(32, device="cuda", dtype=torch.float64) - 0.5)
                       if init == "near_zero"
                       else torch.randn(32, device="cuda", dtype=torch.float64) * 0.5)
            ws = ws_data.detach().clone().requires_grad_(True)
            ok = torch.autograd.gradcheck(
                lambda xx, ww: ForgeRMSNormv3Function.apply(xx, ww, 1e-6, offset, mode),
                (xs, ws), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True,
            )
            out[f"offset={offset}_mode={mode}"] = {"passed": bool(ok)}
        except Exception as e:
            out[f"offset={offset}_mode={mode}"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    out["passed"] = all(v.get("passed", False) for v in out.values())
    return out


def test_autotune_cache():
    """First call compiles + benchmarks 6 configs; subsequent calls hit cache."""
    torch.cuda.synchronize()
    # Use a distinct shape we haven't called before in this process.
    x = torch.randn(2, 64, 3200, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3200, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y1 = apply_rmsnorm_v3(x, w, 1e-6, 0.0, "llama")
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        y2 = apply_rmsnorm_v3(x, w, 1e-6, 0.0, "llama")
    torch.cuda.synchronize()
    cached_avg_ms = (time.perf_counter() - t0) * 1000 / 20

    speedup = first_ms / cached_avg_ms if cached_avg_ms else None
    # First call should be at least 20× slower than cached steady state.
    passed = bool(speedup is not None and speedup >= 20.0)
    return {
        "first_call_ms": first_ms,
        "cached_avg_ms": cached_avg_ms,
        "speedup_first_over_cached": speedup,
        "passed": passed,
        "note": "If passed=False on a clean run, autotune cache may not be active.",
    }


def test_timing():
    cases = []
    for label, b, s, h in SHAPES:
        if h > 16384: continue
        for dtype in [torch.bfloat16, torch.float16]:
            for offset, mode, init in OFFSET_CASES:
                x, w = _make_inputs(b, s, h, dtype, "cuda", init, seed=2)
                call_pt      = lambda x=x, w=w: torch_rmsnorm_reference_v3(x, w, 1e-6, offset, mode)
                call_liger   = lambda x=x, w=w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode)
                call_unsloth = lambda x=x, w=w: unsloth.apply_rmsnorm(x, w, 1e-6, offset)
                call_v1      = (lambda x=x, w=w: rmsnorm_v1(x, w, 1e-6)) if offset == 0.0 else None
                call_v2      = lambda x=x, w=w: apply_rmsnorm_v2(x, w, 1e-6, offset, mode)
                call_v3      = lambda x=x, w=w: apply_rmsnorm_v3(x, w, 1e-6, offset, mode)
                timings = {}
                for name, fn in [("pytorch_ref", call_pt), ("liger", call_liger),
                                 ("unsloth", call_unsloth), ("forge_v1", call_v1),
                                 ("forge_v2", call_v2), ("forge_v3", call_v3)]:
                    if fn is None: continue
                    try:
                        timings[name] = float(triton.testing.do_bench(fn, warmup=10, rep=50))
                    except Exception as e:
                        timings[name] = None
                        timings[f"{name}_error"] = f"{type(e).__name__}: {e}"
                ref = timings.get("pytorch_ref")
                speedups = {k: (ref / t) if (t and ref) else None
                            for k, t in timings.items() if not k.endswith("_error")}
                el = torch.tensor([], dtype=dtype).element_size()
                total_bytes = b * s * h * el * 2 + h * el
                forge_ms = timings.get("forge_v3")
                bw = (total_bytes / 1e9) / (forge_ms / 1000.0) if forge_ms else None
                cases.append({"shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                              "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                              "median_ms": timings, "speedup_vs_pytorch": speedups,
                              "forge_v3_bandwidth_gbps": bw})
    return {"cases": cases}


def collect_environment():
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "triton": triton.__version__, "platform": platform.platform(),
           "cuda_available": torch.cuda.is_available()}
    if env["cuda_available"]:
        idx = torch.cuda.current_device()
        env["cuda_device_name"] = torch.cuda.get_device_name(idx)
        env["cuda_device_capability"] = list(torch.cuda.get_device_capability(idx))
        env["sm_count"] = torch.cuda.get_device_properties(idx).multi_processor_count
    return env


def kernel_config():
    return {"version": "v3", "grid": "(n_rows,)",
            "autotune": True, "autotune_key": ["n_cols", "ACC_DTYPE"],
            "autotune_configs": [{"num_warps": nw, "num_stages": ns}
                                 for nw in (4, 8, 16) for ns in (2, 3)],
            "offset": "tl.constexpr", "casting_mode": "tl.constexpr",
            "backward_strategy": "SM-proportional dW partials (same as v2)"}


def write_md(results, path):
    env = results["environment"]
    fwd = results["results"]["forward_correctness"]["summary"]
    bwd = results["results"]["backward_correctness"]["summary"]
    gc  = results["results"]["gradcheck_fp64"]
    at  = results["results"]["autotune_cache"]
    timing = results["results"]["timing"]["cases"]
    lines = []
    lines.append("# ForgeRMSNorm V3 — Test Results Summary")
    lines.append(""); lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} (SMs={env.get('sm_count', '?')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}"); lines.append("")
    lines.append("## Correctness"); lines.append("")
    lines.append("| Suite | Passed | Total |"); lines.append("|---|---|---|")
    lines.append(f"| Forward correctness   | {fwd['passed']} | {fwd['total']} |")
    lines.append(f"| Backward correctness  | {bwd['passed']} | {bwd['total']} |")
    lines.append(f"| fp64 gradcheck (llama)| {'✓' if gc.get('offset=0.0_mode=llama', {}).get('passed') else '✗'} | 1 |")
    lines.append(f"| fp64 gradcheck (gemma)| {'✓' if gc.get('offset=1.0_mode=gemma', {}).get('passed') else '✗'} | 1 |")
    lines.append(""); lines.append("## Autotune Cache"); lines.append("")
    lines.append(f"- First call (cold compile): {at['first_call_ms']:.1f} ms")
    lines.append(f"- Cached steady state: {at['cached_avg_ms']:.4f} ms")
    if at.get("speedup_first_over_cached"):
        lines.append(f"- Cold/cached speedup ratio: {at['speedup_first_over_cached']:.1f}× ({'PASS' if at['passed'] else 'FAIL'})")
    lines.append(""); lines.append("## Forward + backward timing"); lines.append("")
    lines.append("| Shape | dtype | offset | PT (ms) | Liger (ms) | Unsloth (ms) | V1 (ms) | V2 (ms) | **V3 (ms)** | V3 spd vs PT | V3 BW (GB/s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    def fmt(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
    for c in timing:
        sh = c["shape"]; t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v3")
        sp_str = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
        bw = c.get("forge_v3_bandwidth_gbps")
        bw_str = f"{bw:.0f}" if isinstance(bw, (int, float)) else "—"
        lines.append(f"| {c['shape_label']} ({sh['batch']}×{sh['seq']}×{sh['hidden']}) | {c['dtype']} | "
                     f"{c['offset']} | {fmt(t.get('pytorch_ref'))} | {fmt(t.get('liger'))} | "
                     f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v1'))} | {fmt(t.get('forge_v2'))} | "
                     f"**{fmt(t.get('forge_v3'))}** | {sp_str} | {bw_str} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    out_dir = _REPO_ROOT / "kernels" / "rmsnorm" / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    if not env["cuda_available"]:
        print("CUDA not available — aborting."); return 1
    print("=" * 70); print("ForgeRMSNorm V3 test runner"); print("=" * 70)
    print(f"Device: {env.get('cuda_device_name', '?')} (SMs={env.get('sm_count', '?')})"); print()
    results = {"kernel": kernel_config(), "environment": env,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "results": {}}
    overall_ok = True
    for name, fn in [("forward_correctness", test_forward_correctness),
                      ("backward_correctness", test_backward_correctness),
                      ("gradcheck_fp64", test_gradcheck_fp64),
                      ("autotune_cache", test_autotune_cache),
                      ("timing", test_timing)]:
        try:
            print(f">> {name} ...", flush=True)
            r = fn(); results["results"][name] = r
            s = r.get("summary")
            if s:
                print(f"   {name}: {s['passed']}/{s['total']}")
                if s.get("failed"): overall_ok = False
            elif name == "gradcheck_fp64":
                gc_ok = r.get("passed", False)
                print(f"   gradcheck_fp64: {'PASS' if gc_ok else 'FAIL'}")
                if not gc_ok: overall_ok = False
            elif name == "autotune_cache":
                print(f"   autotune_cache: {'PASS' if r.get('passed') else 'FAIL'} "
                      f"(first={r['first_call_ms']:.1f}ms cached={r['cached_avg_ms']:.4f}ms)")
                # Don't fail overall on autotune cache (depends on Triton internals).
        except Exception as e:
            results["results"][name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            print(f"   ERROR in {name}: {e}")
            overall_ok = False

    json_path = out_dir / "v3_results.json"; md_path = out_dir / "v3_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    write_md(results, md_path)
    print(); print(f"Wrote: {json_path}"); print(f"Wrote: {md_path}")
    print(); print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""ForgeRMSNorm V4 test runner.

V4 = V3 + in_place dY → dX backward (Unsloth-style memory saver).

Runs:
  1. Environment fingerprint
  2. Forward correctness — v4 vs torch_rmsnorm_reference_v4 oracle + Liger + Unsloth
     (covers in_place ∈ {True, False} × offset ∈ {0.0, 1.0} × dtypes)
  3. In-place equivalence — v4(in_place=True) and v4(in_place=False) must produce
     bit-identical fwd/dx/dw. This is the headline correctness check for v4.
  4. Backward correctness — vs autograd of oracle (in_place=False path, since
     in_place=True is gradcheck-non-reentrant by design)
  5. fp64 gradcheck — in_place=False ONLY (in_place=True is fundamentally not
     reentrant: each backward call modifies dy; the SECOND gradcheck call sees
     the corrupted buffer. This is a property of the optimization, not a bug.
     The math equivalence check in (3) covers in_place=True correctness.)
  6. Forward + backward timing — v4(in_place=True) vs v4(False) vs v3 vs Liger
     vs Unsloth

Emits:
  - kernels/rmsnorm/tests/results/v4_results.json
  - kernels/rmsnorm/tests/results/v4_summary.md
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

from kernels.rmsnorm.forge_rmsnorm_v4 import (
    apply_rmsnorm_v4,
    ForgeRMSNormv4Function,
    torch_rmsnorm_reference_v4,
)
from kernels.rmsnorm.forge_rmsnorm_v3 import apply_rmsnorm_v3
from kernels.rmsnorm.forge_rmsnorm_v1 import rmsnorm_v1
from kernels.rmsnorm.baselines import liger, unsloth


TOLERANCES = {
    "torch.bfloat16": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float16":  {"atol": 2e-2, "rtol": 2e-2},
    "torch.float32":  {"atol": 1e-4, "rtol": 1e-5},
    "torch.float64":  {"atol": 1e-7, "rtol": 1e-7},
}
def passes(diff, ref_max, dtype):
    tol = TOLERANCES.get(str(dtype), {"atol": 1e-3, "rtol": 1e-3})
    return diff <= tol["atol"] + tol["rtol"] * abs(ref_max)
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
                # Both in_place settings (forward path is identical, but we run
                # both to confirm in_place is a backward-only knob).
                for in_place in (True, False):
                    y = apply_rmsnorm_v4(x, w, 1e-6, offset, mode, in_place=in_place)
                    oracle = torch_rmsnorm_reference_v4(x, w, 1e-6, offset, mode)
                    diff = max_abs(y, oracle)
                    ok = (passes(diff, ref_max(oracle), dtype)
                          and not torch.isnan(y).any().item()
                          and y.shape == x.shape and y.dtype == dtype)
                    cases.append({
                        "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                        "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                        "in_place": in_place, "diff_vs_oracle": diff,
                        "tolerance": TOLERANCES[str(dtype)], "passed": bool(ok),
                    })
    n = sum(1 for c in cases if c["passed"])
    return {"summary": {"total": len(cases), "passed": n, "failed": len(cases)-n}, "cases": cases}


def test_in_place_equivalence():
    """v4(in_place=True) and v4(in_place=False) must produce bit-identical fwd, dx, dw.

    Headline correctness check for v4. If this fails, the in_place optimization
    silently corrupts gradients.
    """
    cases = []
    subset = [("tiny", 1, 8, 64), ("qwen3_8b_short", 4, 512, 4096),
              ("gemma2_2b", 2, 2048, 2304), ("qwen3_8b_train", 2, 2048, 4096)]
    for label, b, s, h in subset:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for offset, mode, init in OFFSET_CASES:
                x_data = _make_inputs(b, s, h, dtype, "cuda", init, seed=1)[0]
                w_data = _make_inputs(b, s, h, dtype, "cuda", init, seed=1)[1]
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)

                # in_place=True path
                x_t = x_data.clone().requires_grad_(True)
                w_t = w_data.clone().requires_grad_(True)
                y_t = apply_rmsnorm_v4(x_t, w_t, 1e-6, offset, mode, in_place=True)
                y_t.backward(grad.clone())

                # in_place=False path
                x_f = x_data.clone().requires_grad_(True)
                w_f = w_data.clone().requires_grad_(True)
                y_f = apply_rmsnorm_v4(x_f, w_f, 1e-6, offset, mode, in_place=False)
                y_f.backward(grad.clone())

                fwd_d = max_abs(y_t, y_f)
                dx_d  = max_abs(x_t.grad, x_f.grad)
                dw_d  = max_abs(w_t.grad, w_f.grad)
                # Bit-identical required: zero tolerance.
                ok = (fwd_d == 0.0 and dx_d == 0.0 and dw_d == 0.0)
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "fwd_diff": fwd_d, "dx_diff": dx_d, "dw_diff": dw_d,
                    "passed": bool(ok),
                })
    n = sum(1 for c in cases if c["passed"])
    return {"summary": {"total": len(cases), "passed": n, "failed": len(cases)-n},
            "note": "Strict bit-identical between in_place=True and in_place=False.",
            "cases": cases}


def test_backward_correctness():
    """v4 backward vs autograd of oracle. in_place=False (in_place=True
    correctness is covered by the equivalence test above)."""
    cases = []
    subset = [("tiny", 1, 8, 64), ("qwen3_8b_short", 4, 512, 4096),
              ("gemma2_2b", 2, 2048, 2304)]
    for label, b, s, h in subset:
        for dtype in [torch.bfloat16, torch.float32]:
            for offset, mode, init in OFFSET_CASES:
                x, w = _make_inputs(b, s, h, dtype, "cuda", init, seed=2)
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)
                xf = x.clone().requires_grad_(True); wf = w.clone().requires_grad_(True)
                y_f = apply_rmsnorm_v4(xf, wf, 1e-6, offset, mode, in_place=False)
                y_f.backward(grad.clone())
                xo = x.float().clone().requires_grad_(True); wo = w.float().clone().requires_grad_(True)
                y_o = torch_rmsnorm_reference_v4(xo, wo, 1e-6, offset, mode)
                y_o.backward(grad.float())
                dx_d  = max_abs(xf.grad, xo.grad.to(dtype))
                dw_d  = max_abs(wf.grad, wo.grad.to(dtype))
                fwd_d = max_abs(y_f, y_o.to(dtype))
                ok = (passes(dx_d, ref_max(xo.grad), dtype)
                      and passes(dw_d, ref_max(wo.grad), dtype)
                      and passes(fwd_d, ref_max(y_o), dtype))
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "fwd_diff": fwd_d, "dx_diff": dx_d, "dw_diff": dw_d,
                    "tolerance": TOLERANCES[str(dtype)], "passed": bool(ok),
                })
    n = sum(1 for c in cases if c["passed"])
    return {"summary": {"total": len(cases), "passed": n, "failed": len(cases)-n}, "cases": cases}


def test_gradcheck_fp64():
    """fp64 gradcheck — in_place=False ONLY. in_place=True is non-reentrant by design."""
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
                lambda xx, ww: ForgeRMSNormv4Function.apply(xx, ww, 1e-6, offset, mode, False),
                (xs, ws), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True,
            )
            out[f"offset={offset}_mode={mode}_in_place=False"] = {"passed": bool(ok)}
        except Exception as e:
            out[f"offset={offset}_mode={mode}_in_place=False"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    out["passed"] = all(v.get("passed", False) for v in out.values() if isinstance(v, dict))
    out["note"] = ("in_place=True is non-reentrant by design (each backward modifies dy in-place); "
                   "correctness is verified via test_in_place_equivalence instead.")
    return out


def test_timing():
    """Forward + backward timing — full-dW (weight.requires_grad=True) so all
    backends compute dw. Unsloth still skips dw internally (returns None), so
    its numbers represent its frozen-base-LoRA design point."""
    cases = []
    for label, b, s, h in SHAPES:
        if h > 16384: continue
        for dtype in [torch.bfloat16, torch.float16]:
            for offset, mode, init in OFFSET_CASES:
                x_data, w_data = _make_inputs(b, s, h, dtype, "cuda", init, seed=3)
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)

                def make_step(fn, w_grad=True):
                    def step():
                        x = x_data.clone().requires_grad_(True)
                        w = w_data.clone().requires_grad_(w_grad)
                        y = fn(x, w)
                        y.backward(grad.clone())
                    return step

                steps = {
                    "pytorch_ref": make_step(lambda x, w: torch_rmsnorm_reference_v4(x, w, 1e-6, offset, mode)),
                    "liger":       make_step(lambda x, w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode)),
                    "unsloth":     make_step(lambda x, w: unsloth.apply_rmsnorm(x, w, 1e-6, offset)),
                    "forge_v3":    make_step(lambda x, w: apply_rmsnorm_v3(x, w, 1e-6, offset, mode)),
                    "forge_v4_in_place":  make_step(lambda x, w: apply_rmsnorm_v4(x, w, 1e-6, offset, mode, True)),
                    "forge_v4_out_place": make_step(lambda x, w: apply_rmsnorm_v4(x, w, 1e-6, offset, mode, False)),
                }
                if offset == 0.0:
                    steps["forge_v1"] = make_step(lambda x, w: rmsnorm_v1(x, w, 1e-6))
                # Warm autotune cache
                for fn in steps.values(): fn()
                timings = {}
                for name, fn in steps.items():
                    try:
                        timings[name] = float(triton.testing.do_bench(fn, warmup=10, rep=50))
                    except Exception as e:
                        timings[name] = None
                        timings[f"{name}_error"] = f"{type(e).__name__}: {e}"
                ref = timings.get("pytorch_ref")
                speedups = {k: (ref / t) if (isinstance(t, (int, float)) and ref) else None
                            for k, t in timings.items() if not k.endswith("_error")}
                v3_ms = timings.get("forge_v3"); v4_ip = timings.get("forge_v4_in_place")
                v4_vs_v3 = (v3_ms / v4_ip) if (isinstance(v3_ms, (int, float)) and isinstance(v4_ip, (int, float))) else None
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "median_ms": timings, "speedup_vs_pytorch": speedups,
                    "v4_inplace_speedup_vs_v3": v4_vs_v3,
                })
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
    return {"version": "v4", "grid": "(n_rows,)",
            "autotune": True, "autotune_key": ["n_cols", "ACC_DTYPE"],
            "autotune_restore_value": ["dX_ptr", "dY_ptr"],
            "offset": "tl.constexpr", "casting_mode": "tl.constexpr",
            "in_place_default": True,
            "in_place_notes": "Backward writes dx into dy buffer when True. Saves alloc + HBM. "
                              "Closure factory defaults to True for Qwen/Llama (offset=0) and "
                              "False for Gemma (offset=1, residual-paired pattern)."}


def write_md(results, path):
    env = results["environment"]
    fwd = results["results"]["forward_correctness"]["summary"]
    eqv = results["results"]["in_place_equivalence"]["summary"]
    bwd = results["results"]["backward_correctness"]["summary"]
    gc  = results["results"]["gradcheck_fp64"]
    timing = results["results"]["timing"]["cases"]
    lines = []
    lines.append("# ForgeRMSNorm V4 — Test Results Summary"); lines.append("")
    lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} (SMs={env.get('sm_count', '?')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}"); lines.append("")
    lines.append("## Correctness"); lines.append("")
    lines.append("| Suite | Passed | Total |"); lines.append("|---|---|---|")
    lines.append(f"| Forward correctness            | {fwd['passed']} | {fwd['total']} |")
    lines.append(f"| in_place=True ↔ False bit-identical | {eqv['passed']} | {eqv['total']} |")
    lines.append(f"| Backward correctness (out-of-place) | {bwd['passed']} | {bwd['total']} |")
    lines.append(f"| fp64 gradcheck (llama, out-of-place) | {'✓' if gc.get('offset=0.0_mode=llama_in_place=False', {}).get('passed') else '✗'} | 1 |")
    lines.append(f"| fp64 gradcheck (gemma, out-of-place) | {'✓' if gc.get('offset=1.0_mode=gemma_in_place=False', {}).get('passed') else '✗'} | 1 |")
    lines.append("")
    lines.append(f"> {gc.get('note', '')}"); lines.append("")
    lines.append("## Forward + backward timing (median ms; smaller = better)"); lines.append("")
    lines.append("| Shape | dtype | offset | PT | Liger | Unsloth(no-dW) | V1 | V3 | V4(ip) | V4(op) | V4(ip) vs Liger | V4(ip) vs V3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    def fmt(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
    for c in timing:
        sh = c["shape"]; t = c["median_ms"]
        v4ip = t.get("forge_v4_in_place"); lg = t.get("liger"); v3 = t.get("forge_v3")
        vs_l = f"{lg/v4ip:.2f}×" if isinstance(v4ip, (int, float)) and isinstance(lg, (int, float)) else "—"
        vs_v3 = f"{v3/v4ip:.2f}×" if isinstance(v4ip, (int, float)) and isinstance(v3, (int, float)) else "—"
        lines.append(f"| {c['shape_label']}({sh['batch']}×{sh['seq']}×{sh['hidden']}) | {c['dtype']} | "
                     f"{c['offset']} | {fmt(t.get('pytorch_ref'))} | {fmt(t.get('liger'))} | "
                     f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v1'))} | {fmt(t.get('forge_v3'))} | "
                     f"**{fmt(t.get('forge_v4_in_place'))}** | {fmt(t.get('forge_v4_out_place'))} | "
                     f"{vs_l} | {vs_v3} |")
    lines.append("")
    lines.append("> Note: Unsloth's backward returns `None` for dW (designed for frozen-base+LoRA "
                 "training). Its timing represents that design point — not a fair perf comparison "
                 "for full fine-tuning workloads where dW is computed (Liger, Forge v1-v4).")
    path.write_text("\n".join(lines))


def main():
    out_dir = _REPO_ROOT / "kernels" / "rmsnorm" / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    if not env["cuda_available"]:
        print("CUDA not available — aborting."); return 1
    print("=" * 70); print("ForgeRMSNorm V4 test runner"); print("=" * 70)
    print(f"Device: {env.get('cuda_device_name', '?')} (SMs={env.get('sm_count', '?')})"); print()
    results = {"kernel": kernel_config(), "environment": env,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "results": {}}
    overall_ok = True
    for name, fn in [("forward_correctness", test_forward_correctness),
                      ("in_place_equivalence", test_in_place_equivalence),
                      ("backward_correctness", test_backward_correctness),
                      ("gradcheck_fp64", test_gradcheck_fp64),
                      ("timing", test_timing)]:
        try:
            print(f">> {name} ...", flush=True)
            r = fn(); results["results"][name] = r
            s = r.get("summary")
            if s:
                print(f"   {name}: {s['passed']}/{s['total']}")
                if s.get("failed"): overall_ok = False
            elif name == "gradcheck_fp64":
                ok = r.get("passed", False)
                print(f"   gradcheck_fp64 (out-of-place only): {'PASS' if ok else 'FAIL'}")
                if not ok: overall_ok = False
        except Exception as e:
            results["results"][name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            print(f"   ERROR in {name}: {e}")
            overall_ok = False

    json_path = out_dir / "v4_results.json"; md_path = out_dir / "v4_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    write_md(results, md_path)
    print(); print(f"Wrote: {json_path}"); print(f"Wrote: {md_path}")
    print(); print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

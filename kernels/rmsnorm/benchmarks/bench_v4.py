"""Benchmark runner for ForgeRMSNorm v4.

Head-to-head V4(in_place=True) vs V4(in_place=False) vs V3 vs Liger vs Unsloth.

Headline question: does in-place dY→dX close the Unsloth backward gap?
Honest answer (see v4_summary.md): for full-dW workloads, V4(in_place=True)
beats Liger by ~1.4× and lands within ~1.2-1.6× of Unsloth. The remaining
Unsloth gap is feature-completeness — Unsloth's backward returns None for dW
(frozen-base-only LoRA design), Forge computes the full dW.

Emits:
  - kernels/rmsnorm/benchmarks/results/v4_results.json
  - kernels/rmsnorm/benchmarks/results/v4_summary.md
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

from kernels.rmsnorm.forge_rmsnorm_v4 import apply_rmsnorm_v4, torch_rmsnorm_reference_v4
from kernels.rmsnorm.forge_rmsnorm_v3 import apply_rmsnorm_v3
from kernels.rmsnorm.forge_rmsnorm_v1 import rmsnorm_v1
from kernels.rmsnorm.baselines import liger, unsloth


SHAPES = [
    ("qwen25_0p5b",         2,  512,   896),
    ("qwen3_8b_short",      4,  512,  4096),
    ("qwen3_8b_train",      2, 2048,  4096),
    ("gemma2_2b",           2, 2048,  2304),
    ("gemma2_9b",           2, 2048,  3584),
    ("gemma2_27b",          1, 2048,  4608),
    ("llama3_70b",          1,  512,  8192),
]
DTYPES = [torch.bfloat16, torch.float16]
OFFSET_CASES = [(0.0, "llama"), (1.0, "gemma")]


def _mk(b, s, h, dt, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(b, s, h, device="cuda", dtype=dt, generator=g)
    w = torch.ones(h, device="cuda", dtype=dt) + 0.1 * torch.randn(h, device="cuda", dtype=dt, generator=g)
    return x.contiguous(), w.contiguous()


def _bench(fn, warmup=20, rep=100):
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def bench_forward():
    """Forward-only — confirms all 4 Triton kernels (Liger, Unsloth, v3, v4)
    are at HBM-peak parity (in_place is a backward-only knob)."""
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode in OFFSET_CASES:
                x, w = _mk(b, s, h, dtype)
                refs = {
                    "pytorch_eager":  lambda x=x, w=w: torch_rmsnorm_reference_v4(x, w, 1e-6, offset, mode),
                    "liger":          lambda x=x, w=w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode),
                    "unsloth":        lambda x=x, w=w: unsloth.apply_rmsnorm(x, w, 1e-6, offset),
                    "forge_v3":       lambda x=x, w=w: apply_rmsnorm_v3(x, w, 1e-6, offset, mode),
                    "forge_v4":       lambda x=x, w=w: apply_rmsnorm_v4(x, w, 1e-6, offset, mode, True),
                }
                for fn in refs.values():
                    try: fn()
                    except Exception: pass
                timings = {name: _bench(fn) for name, fn in refs.items()}
                ref = timings.get("pytorch_eager")
                ref_ms = ref if isinstance(ref, (int, float)) else None
                speedups = {k: (ref_ms / t) if (isinstance(t, (int, float)) and ref_ms) else None
                            for k, t in timings.items()}
                el = torch.tensor([], dtype=dtype).element_size()
                bytes_per_call = b * s * h * el * 2 + h * el
                v4_ms = timings.get("forge_v4")
                bw = (bytes_per_call / 1e9) / (v4_ms / 1000.0) if isinstance(v4_ms, (int, float)) else None
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "median_ms": timings, "speedup_vs_pytorch": speedups,
                    "forge_v4_bandwidth_gbps": bw,
                })
    return {"cases": cases}


def bench_forward_backward():
    """Forward+backward combined — the headline v4 comparison.

    Two scenarios:
      - "full_dW": weight.requires_grad=True. All backends compute dw EXCEPT
                   Unsloth (which always returns None for dw). Unsloth's
                   numbers here represent its frozen-base-LoRA design point.
      - "frozen_w": weight.requires_grad=False. Now-fair comparison vs Unsloth.
    """
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode in OFFSET_CASES:
                x_data, w_data = _mk(b, s, h, dtype, seed=1)
                grad = torch.randn(b, s, h, device="cuda", dtype=dtype)

                def make_step(apply_fn, w_grad):
                    def step():
                        x = x_data.clone().requires_grad_(True)
                        w = w_data.clone().requires_grad_(w_grad)
                        y = apply_fn(x, w)
                        y.backward(grad.clone())
                    return step

                fns = {
                    "pytorch_eager":  lambda x, w: torch_rmsnorm_reference_v4(x, w, 1e-6, offset, mode),
                    "liger":          lambda x, w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode),
                    "unsloth":        lambda x, w: unsloth.apply_rmsnorm(x, w, 1e-6, offset),
                    "forge_v3":       lambda x, w: apply_rmsnorm_v3(x, w, 1e-6, offset, mode),
                    "forge_v4_in_place":  lambda x, w: apply_rmsnorm_v4(x, w, 1e-6, offset, mode, True),
                    "forge_v4_out_place": lambda x, w: apply_rmsnorm_v4(x, w, 1e-6, offset, mode, False),
                }
                if offset == 0.0:
                    fns["forge_v1"] = lambda x, w: rmsnorm_v1(x, w, 1e-6)

                results_full = {}; results_frozen = {}
                for name, fn in fns.items():
                    step_full = make_step(fn, True)
                    step_frozen = make_step(fn, False)
                    try: step_full(); step_frozen()
                    except Exception: pass
                    results_full[name] = _bench(step_full, warmup=10, rep=50)
                    results_frozen[name] = _bench(step_frozen, warmup=10, rep=50)

                def speedups(ts, ref_name="pytorch_eager"):
                    r = ts.get(ref_name)
                    return {k: (r / t) if (isinstance(t, (int, float)) and isinstance(r, (int, float))) else None
                            for k, t in ts.items()}

                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "full_dW": {
                        "median_ms": results_full,
                        "speedup_vs_pytorch": speedups(results_full),
                    },
                    "frozen_w": {
                        "median_ms": results_frozen,
                        "speedup_vs_pytorch": speedups(results_frozen),
                    },
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


def write_md(results, path):
    env = results["environment"]
    fwd = results["results"]["bench_forward"]["cases"]
    fb  = results["results"]["bench_forward_backward"]["cases"]
    lines = []
    lines.append("# ForgeRMSNorm V4 — Benchmark Results")
    lines.append(""); lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} (SMs={env.get('sm_count', '?')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}"); lines.append("")
    lines.append("## Forward-only (median ms; smaller = better)"); lines.append("")
    lines.append("| Shape | dt | off | PT | Liger | Unsloth | V3 | **V4** | V4 spd vs PT | V4 BW GB/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    for c in fwd:
        sh = c["shape"]; t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v4")
        sp_s = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
        bw = c.get("forge_v4_bandwidth_gbps"); bw_s = f"{bw:.0f}" if isinstance(bw, (int, float)) else "—"
        lines.append(f"| {c['shape_label']}({sh['batch']}×{sh['seq']}×{sh['hidden']}) | "
                     f"{c['dtype'].replace('torch.','')} | {c['offset']} | "
                     f"{fmt(t.get('pytorch_eager'))} | {fmt(t.get('liger'))} | "
                     f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v3'))} | "
                     f"**{fmt(t.get('forge_v4'))}** | {sp_s} | {bw_s} |")
    lines.append("")
    lines.append("## Forward + backward, **weight.requires_grad=True** (full dW — Liger/Forge fair)")
    lines.append("")
    lines.append("> Unsloth returns None for dW (frozen-base+LoRA design); its number is for reference.")
    lines.append("")
    lines.append("| Shape | dt | off | PT | Liger | Unsl(no-dW) | V3 | V4(ip) | V4(op) | V4(ip)/Liger | V4(ip)/V3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in fb:
        sh = c["shape"]; t = c["full_dW"]["median_ms"]
        v4ip = t.get("forge_v4_in_place"); lg = t.get("liger"); v3 = t.get("forge_v3")
        vs_l = f"{lg/v4ip:.2f}×" if isinstance(v4ip, (int, float)) and isinstance(lg, (int, float)) else "—"
        vs_v3 = f"{v3/v4ip:.2f}×" if isinstance(v4ip, (int, float)) and isinstance(v3, (int, float)) else "—"
        lines.append(f"| {c['shape_label']}({sh['batch']}×{sh['seq']}×{sh['hidden']}) | "
                     f"{c['dtype'].replace('torch.','')} | {c['offset']} | "
                     f"{fmt(t.get('pytorch_eager'))} | {fmt(t.get('liger'))} | "
                     f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v3'))} | "
                     f"**{fmt(t.get('forge_v4_in_place'))}** | {fmt(t.get('forge_v4_out_place'))} | "
                     f"{vs_l} | {vs_v3} |")
    lines.append("")
    lines.append("## Forward + backward, **weight.requires_grad=False** (Unsloth-fair frozen-w)")
    lines.append("")
    lines.append("| Shape | dt | off | PT | Liger | Unsloth | V3 | V4(ip) | V4(op) | V4(ip)/Unsl |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in fb:
        sh = c["shape"]; t = c["frozen_w"]["median_ms"]
        v4ip = t.get("forge_v4_in_place"); us = t.get("unsloth")
        vs_u = f"{us/v4ip:.2f}×" if isinstance(v4ip, (int, float)) and isinstance(us, (int, float)) else "—"
        lines.append(f"| {c['shape_label']}({sh['batch']}×{sh['seq']}×{sh['hidden']}) | "
                     f"{c['dtype'].replace('torch.','')} | {c['offset']} | "
                     f"{fmt(t.get('pytorch_eager'))} | {fmt(t.get('liger'))} | "
                     f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v3'))} | "
                     f"**{fmt(t.get('forge_v4_in_place'))}** | {fmt(t.get('forge_v4_out_place'))} | "
                     f"{vs_u} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    out_dir = _REPO_ROOT / "kernels" / "rmsnorm" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    if not env["cuda_available"]:
        print("CUDA not available — aborting."); return 1
    print("=" * 70); print("ForgeRMSNorm V4 benchmark runner"); print("=" * 70)
    print(f"Device: {env.get('cuda_device_name', '?')} (SMs={env.get('sm_count', '?')})"); print()
    results = {"environment": env,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "results": {}}
    for name, fn in [("bench_forward", bench_forward),
                      ("bench_forward_backward", bench_forward_backward)]:
        print(f">> {name} ...", flush=True)
        try:
            results["results"][name] = fn()
            print(f"   {name}: {len(results['results'][name]['cases'])} cases")
        except Exception as e:
            results["results"][name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            print(f"   ERROR: {e}")
    json_path = out_dir / "v4_results.json"; md_path = out_dir / "v4_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    write_md(results, md_path)
    print(); print(f"Wrote: {json_path}"); print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

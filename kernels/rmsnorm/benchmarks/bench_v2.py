"""Benchmark runner for ForgeRMSNorm v2.

Standalone timing harness (no correctness checks — those live in tests/test_v2.py).
Compares Forge v2 against pytorch_eager, Liger, Unsloth, and Forge v1 at production
shapes for both Llama (offset=0) and Gemma (offset=1) paths.

Emits:
  - kernels/rmsnorm/benchmarks/results/v2_results.json
  - kernels/rmsnorm/benchmarks/results/v2_summary.md
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

from kernels.rmsnorm.forge_rmsnorm_v2 import apply_rmsnorm_v2, torch_rmsnorm_reference_v2
from kernels.rmsnorm.forge_rmsnorm_v1 import rmsnorm_v1
from kernels.rmsnorm.baselines import liger, unsloth


# (label, batch, seq, hidden) — production-relevant shapes
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
# (offset, casting_mode)
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


def bench_forward() -> dict[str, Any]:
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode in OFFSET_CASES:
                x, w = _mk(b, s, h, dtype)
                refs = {
                    "pytorch_eager":    lambda x=x, w=w: torch_rmsnorm_reference_v2(x, w, 1e-6, offset, mode),
                    "liger":            lambda x=x, w=w: liger.apply_rmsnorm(x, w, 1e-6, offset, mode),
                    "unsloth":          lambda x=x, w=w: unsloth.apply_rmsnorm(x, w, 1e-6, offset),
                    "forge_v2":         lambda x=x, w=w: apply_rmsnorm_v2(x, w, 1e-6, offset, mode),
                }
                if offset == 0.0:
                    refs["forge_v1"] = lambda x=x, w=w: rmsnorm_v1(x, w, 1e-6)
                timings = {name: _bench(fn) for name, fn in refs.items()}
                ref = timings.get("pytorch_eager")
                ref_ms = ref if isinstance(ref, (int, float)) else None
                speedups = {}
                for k, t in timings.items():
                    if isinstance(t, (int, float)) and ref_ms:
                        speedups[k] = ref_ms / t
                el = torch.tensor([], dtype=dtype).element_size()
                bytes_per_call = b * s * h * el * 2 + h * el  # x read + y write + w read
                forge_ms = timings.get("forge_v2")
                bw = (bytes_per_call / 1e9) / (forge_ms / 1000.0) if isinstance(forge_ms, (int, float)) else None
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "median_ms": timings, "speedup_vs_pytorch": speedups,
                    "forge_v2_bandwidth_gbps": bw, "bytes_per_call": bytes_per_call,
                })
    return {"cases": cases}


def bench_forward_backward() -> dict[str, Any]:
    """Forward+backward combined latency — exercises the SM-proportional dW partials path."""
    cases = []
    for label, b, s, h in SHAPES:
        for dtype in DTYPES:
            for offset, mode in OFFSET_CASES:
                x, w = _mk(b, s, h, dtype, seed=1)
                grad = torch.randn_like(x.detach())

                def call(fn, offset=offset, mode=mode):
                    def step():
                        xx = x.clone().requires_grad_(True)
                        ww = w.clone().requires_grad_(True)
                        y = fn(xx, ww, offset, mode)
                        y.backward(grad)
                    return step

                # Wrappers that ignore the offset/mode args if the impl doesn't take them.
                pt   = lambda xx, ww, o, m: torch_rmsnorm_reference_v2(xx, ww, 1e-6, o, m)
                lg   = lambda xx, ww, o, m: liger.apply_rmsnorm(xx, ww, 1e-6, o, m)
                us   = lambda xx, ww, o, m: unsloth.apply_rmsnorm(xx, ww, 1e-6, o)
                v1   = lambda xx, ww, o, m: rmsnorm_v1(xx, ww, 1e-6)
                v2   = lambda xx, ww, o, m: apply_rmsnorm_v2(xx, ww, 1e-6, o, m)

                refs = {
                    "pytorch_eager":  call(pt),
                    "liger":          call(lg),
                    "unsloth":        call(us),
                    "forge_v2":       call(v2),
                }
                if offset == 0.0:
                    refs["forge_v1"] = call(v1)
                timings = {name: _bench(fn, warmup=10, rep=50) for name, fn in refs.items()}
                ref = timings.get("pytorch_eager")
                ref_ms = ref if isinstance(ref, (int, float)) else None
                speedups = {k: (ref_ms / t) if (isinstance(t, (int, float)) and ref_ms) else None
                            for k, t in timings.items()}
                cases.append({
                    "shape_label": label, "shape": {"batch": b, "seq": s, "hidden": h},
                    "dtype": str(dtype), "offset": offset, "casting_mode": mode,
                    "median_ms": timings, "speedup_vs_pytorch": speedups,
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
    lines.append("# ForgeRMSNorm V2 — Benchmark Results"); lines.append("")
    lines.append(f"**Run:** {results['timestamp_utc']}")
    lines.append(f"**Device:** {env.get('cuda_device_name', 'CPU')} (SMs={env.get('sm_count', '?')})")
    lines.append(f"**Torch / Triton:** {env['torch']} / {env['triton']}"); lines.append("")
    lines.append("## Forward-only (median ms, lower = better)"); lines.append("")
    lines.append("| Shape | dtype | offset | PT | Liger | Unsloth | V1 | **V2** | V2 spd vs PT | V2 BW GB/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    for c in fwd:
        sh = c["shape"]; t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v2")
        sp_s = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
        bw = c.get("forge_v2_bandwidth_gbps"); bw_s = f"{bw:.0f}" if isinstance(bw, (int, float)) else "—"
        lines.append(
            f"| {c['shape_label']} ({sh['batch']}×{sh['seq']}×{sh['hidden']}) | {c['dtype']} | "
            f"{c['offset']} | {fmt(t.get('pytorch_eager'))} | {fmt(t.get('liger'))} | "
            f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v1'))} | "
            f"**{fmt(t.get('forge_v2'))}** | {sp_s} | {bw_s} |"
        )
    lines.append("")
    lines.append("## Forward + backward combined (median ms, lower = better)"); lines.append("")
    lines.append("| Shape | dtype | offset | PT | Liger | Unsloth | V1 | **V2** | V2 spd vs PT |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in fb:
        sh = c["shape"]; t = c["median_ms"]
        sp = c["speedup_vs_pytorch"].get("forge_v2")
        sp_s = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
        lines.append(
            f"| {c['shape_label']} ({sh['batch']}×{sh['seq']}×{sh['hidden']}) | {c['dtype']} | "
            f"{c['offset']} | {fmt(t.get('pytorch_eager'))} | {fmt(t.get('liger'))} | "
            f"{fmt(t.get('unsloth'))} | {fmt(t.get('forge_v1'))} | "
            f"**{fmt(t.get('forge_v2'))}** | {sp_s} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    out_dir = _REPO_ROOT / "kernels" / "rmsnorm" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    if not env["cuda_available"]:
        print("CUDA not available — aborting."); return 1
    print("=" * 70); print("ForgeRMSNorm V2 benchmark runner"); print("=" * 70)
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
            results["results"][name] = {"error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc()}
            print(f"   ERROR: {e}")

    json_path = out_dir / "v2_results.json"; md_path = out_dir / "v2_summary.md"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    write_md(results, md_path)
    print(); print(f"Wrote: {json_path}"); print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

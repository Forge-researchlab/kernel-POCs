"""Benchmark Qwen providers: vanilla Torch/HF, Forge patching, and Liger.

Run one provider/mode/model shape per process so monkey patches do not leak
between providers. Results are appended as JSONL for later report generation.

Example:
  PYTHONPATH=forge:. .venv/bin/python forge/benchmarks/bench_qwen_provider.py \
    --provider forge --model Qwen/Qwen2.5-0.5B --mode train_backward \
    --batch 1 --seq-len 128 --output artifacts/qwen_perf/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Qwen Torch vs Forge vs Liger")
    p.add_argument("--provider", choices=["torch", "forge", "liger"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--mode", choices=["logits_forward", "loss_forward", "train_backward"], required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="Append JSONL result here")
    return p.parse_args()


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return float("nan")
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def apply_provider(provider, model):
    if provider == "torch":
        return {}
    if provider == "forge":
        import forge

        forge.patch(model)
        return dict(getattr(model, "_forge_patched_counts", {}))
    if provider == "liger":
        model_type = getattr(model.config, "model_type", None)
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen2
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3

        if model_type == "qwen2":
            apply_liger_kernel_to_qwen2(
                model=model,
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
            )
        elif model_type == "qwen3":
            apply_liger_kernel_to_qwen3(
                model=model,
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
            )
        else:
            raise ValueError(f"Liger Qwen patch does not support model_type={model_type!r}")
        return {
            "rope": True,
            "fused_linear_ce": True,
            "rmsnorm": True,
            "swiglu": True,
        }
    raise AssertionError(provider)


def main() -> int:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.config.use_cache = False
    load_s = time.perf_counter() - load_start

    # Fused linear CE paths are activated by training=True in both Forge/Liger.
    if args.mode == "logits_forward":
        model.eval()
    else:
        model.train()

    patch_start = time.perf_counter()
    patched = apply_provider(args.provider, model)
    patch_s = time.perf_counter() - patch_start

    input_ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device="cuda")
    labels = input_ids.clone()

    def zero_grads():
        for p in model.parameters():
            p.grad = None

    def run_once():
        if args.mode == "logits_forward":
            with torch.no_grad():
                out = model(input_ids=input_ids)
                result = out.logits
                # Touch the tensor so lazy errors surface.
                return float(result[..., 0].float().mean().detach().cpu())
        if args.mode == "loss_forward":
            with torch.no_grad():
                out = model(input_ids=input_ids, labels=labels)
                return float(out.loss.detach().float().cpu())
        if args.mode == "train_backward":
            zero_grads()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            return float(loss.detach().float().cpu())
        raise AssertionError(args.mode)

    # Warmup compiles Triton kernels and fills CUDA caches.
    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    base_alloc_mb = torch.cuda.memory_allocated() / 1024**2
    times_ms = []
    outputs = []
    peak_alloc_mb = base_alloc_mb
    peak_reserved_mb = torch.cuda.memory_reserved() / 1024**2

    for _ in range(args.iters):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs.append(run_once())
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        peak_alloc_mb = max(peak_alloc_mb, torch.cuda.max_memory_allocated() / 1024**2)
        peak_reserved_mb = max(peak_reserved_mb, torch.cuda.max_memory_reserved() / 1024**2)

    median_ms = statistics.median(times_ms)
    tokens = args.batch * args.seq_len
    result = {
        "provider": args.provider,
        "model": args.model,
        "model_type": getattr(model.config, "model_type", None),
        "mode": args.mode,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "tokens": tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "warmup": args.warmup,
        "iters": args.iters,
        "times_ms": times_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(times_ms),
        "p20_ms": percentile(times_ms, 20),
        "p80_ms": percentile(times_ms, 80),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "tokens_per_s_median": tokens / (median_ms / 1000.0),
        "base_allocated_mb_after_warmup": base_alloc_mb,
        "peak_allocated_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "load_s": load_s,
        "patch_s": patch_s,
        "patched": patched,
        "output_sample_mean": statistics.mean(outputs),
        "env": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    try:
        import liger_kernel

        result["env"]["liger_kernel_module"] = getattr(liger_kernel, "__file__", None)
    except Exception:
        pass

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "times_ms"}, indent=2))
    print("times_ms=", ", ".join(f"{x:.3f}" for x in times_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

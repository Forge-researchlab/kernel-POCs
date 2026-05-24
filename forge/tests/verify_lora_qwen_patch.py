"""Forward/backward correctness verifier for Forge-patched Qwen LoRA training.

Compares a PEFT LoRA Qwen model before and after forge.patch on the same random
batch. It checks:
  - forward loss closeness
  - adapter gradient closeness for every trainable LoRA parameter
  - finite grads/losses
  - unpatch restores the original forwards
  - optional JSON artifact with per-parameter gradient errors

Run on CUDA/Linux, for example:

  PYTHONPATH=forge:. UV_CACHE_DIR=/workspace/.uv-cache uv run python \
    forge/tests/verify_lora_qwen_patch.py \
    --model Qwen/Qwen2.5-0.5B --seq-len 16 --batch 1 \
    --artifact artifacts/verify_lora_qwen2_5.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Forge LoRA Qwen forward/backward correctness.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["lora_mlp", "lora_qkv", "fused_linear_ce"],
        help="Patch kernels to compare against unpatched PEFT. Keep this focused for tight grad diffs.",
    )
    parser.add_argument("--loss-atol", type=float, default=5e-3)
    parser.add_argument("--grad-max-atol", type=float, default=1.0)
    parser.add_argument("--grad-mean-atol", type=float, default=5e-3)
    parser.add_argument("--grad-cos-min", type=float, default=0.98)
    parser.add_argument("--artifact", default=None)
    return parser.parse_args()


def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return model


def zero_trainable_grads(model):
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None


def collect_lora_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                grads[name] = None
            else:
                grads[name] = param.grad.detach().float().cpu().clone()
    return grads


def run_loss_backward(model, input_ids):
    import torch

    zero_trainable_grads(model)
    out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"missing or non-finite loss: {loss}")
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach().float().cpu()), collect_lora_grads(model)


def compare_grads(base_grads, patched_grads):
    import torch

    rows = []
    max_abs = 0.0
    max_mean = 0.0
    min_cos = 1.0
    missing = []
    for name, base in base_grads.items():
        patched = patched_grads.get(name)
        if base is None or patched is None:
            if base is not patched:
                missing.append(name)
            continue
        diff = (patched - base).abs()
        abs_max = float(diff.max()) if diff.numel() else 0.0
        abs_mean = float(diff.mean()) if diff.numel() else 0.0
        base_flat = base.flatten()
        patched_flat = patched.flatten()
        base_norm = base_flat.norm()
        patched_norm = patched_flat.norm()
        denom = base_norm * patched_norm
        if not base_flat.numel() or (float(base_norm) < 1e-12 and float(patched_norm) < 1e-12):
            cos = 1.0
        else:
            cos = float((base_flat @ patched_flat) / (denom + 1e-12))
        max_abs = max(max_abs, abs_max)
        max_mean = max(max_mean, abs_mean)
        min_cos = min(min_cos, cos)
        rows.append({
            "name": name,
            "shape": list(base.shape),
            "max_abs_diff": abs_max,
            "mean_abs_diff": abs_mean,
            "cosine": cos,
            "base_norm": float(base_norm),
            "patched_norm": float(patched_norm),
        })
    return rows, {"max_abs_diff": max_abs, "max_mean_abs_diff": max_mean, "min_cosine": min_cos, "missing": missing}


def main() -> int:
    args = parse_args()

    import torch
    import forge

    try:
        import peft  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Install PEFT first: uv pip install peft") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    model = build_model(args)
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device="cuda")

    base_loss, base_grads = run_loss_backward(model, input_ids)

    forge.patch(model, kernels=args.kernels)
    patched_counts = dict(model._forge_patched_counts)
    patched_loss, patched_grads = run_loss_backward(model, input_ids)
    forge.unpatch(model)

    restored = not getattr(model, "_forge_patched", False)
    rows, summary = compare_grads(base_grads, patched_grads)
    loss_diff = abs(patched_loss - base_loss)

    ok = (
        restored
        and loss_diff <= args.loss_atol
        and not summary["missing"]
        and summary["max_abs_diff"] <= args.grad_max_atol
        and summary["max_mean_abs_diff"] <= args.grad_mean_atol
        and summary["min_cosine"] >= args.grad_cos_min
    )

    artifact = {
        "model": args.model,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "rank": args.rank,
        "alpha": args.alpha,
        "kernels": args.kernels,
        "patched_counts": patched_counts,
        "base_loss": base_loss,
        "patched_loss": patched_loss,
        "loss_diff": loss_diff,
        "summary": summary,
        "grads": rows,
        "restored": restored,
        "passed": ok,
    }

    print(json.dumps({k: artifact[k] for k in artifact if k != "grads"}, indent=2))
    if args.artifact:
        path = Path(args.artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2))
        print(f"saved artifact to {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end forward/backward verifier for Forge-patched Qwen models.

This is complementary to ``verify_patch_qwen3.py``:
  - ``verify_patch_qwen3.py`` checks forward logits/unpatch behavior.
  - this script checks loss forward + backward gradients on selected params.

It runs the same model twice on the same batch: unpatched baseline first, then
Forge-patched. Gradients are compared for representative parameters while
avoiding huge full CPU copies of embedding/lm_head matrices.
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
    parser = argparse.ArgumentParser(description="Verify Qwen Forge patch forward+backward correctness.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["fused_linear_ce", "swiglu", "rmsnorm"],
        help="Kernels to patch. Use focused sets for strict gradient comparisons.",
    )
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-max-atol", type=float, default=2e-3)
    parser.add_argument("--grad-mean-atol", type=float, default=2e-5)
    parser.add_argument("--grad-cos-min", type=float, default=0.999)
    parser.add_argument("--sample-elements", type=int, default=4096)
    parser.add_argument("--artifact", default=None)
    return parser.parse_args()


def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.config.use_cache = False
    model.train()
    return model


def zero_grads(model):
    for param in model.parameters():
        param.grad = None


def selected_param_names(model):
    """Representative params touched by patched paths, excluding giant matrices."""
    names = []
    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    layer_ids = {0, max(0, num_layers // 2), max(0, num_layers - 1)}
    needles = [
        "model.norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("model.norm.weight"):
            names.append(name)
            continue
        layer_ok = False
        for layer_id in layer_ids:
            if f"layers.{layer_id}." in name:
                layer_ok = True
                break
        if layer_ok and any(needle in name for needle in needles):
            names.append(name)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(names))


def collect_grad_samples(model, names, sample_elements):
    import torch

    params = dict(model.named_parameters())
    out = {}
    for name in names:
        grad = params[name].grad
        if grad is None:
            out[name] = None
            continue
        flat = grad.detach().float().flatten()
        if flat.numel() > sample_elements:
            # Deterministic prefix+suffix sample catches common projection/layout errors
            half = sample_elements // 2
            flat = torch.cat([flat[:half], flat[-(sample_elements - half):]])
        out[name] = flat.cpu().clone()
    return out


def run_loss_backward(model, input_ids, names, sample_elements):
    import torch

    zero_grads(model)
    out = model(input_ids=input_ids, labels=input_ids)
    loss = out.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"missing or non-finite loss: {loss}")
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach().float().cpu()), collect_grad_samples(model, names, sample_elements)


def compare_grads(base, patched):
    rows = []
    max_abs = 0.0
    max_mean = 0.0
    min_cos = 1.0
    missing = []
    for name, base_grad in base.items():
        patched_grad = patched.get(name)
        if base_grad is None or patched_grad is None:
            if base_grad is not patched_grad:
                missing.append(name)
            continue
        diff = (patched_grad - base_grad).abs()
        abs_max = float(diff.max()) if diff.numel() else 0.0
        abs_mean = float(diff.mean()) if diff.numel() else 0.0
        base_norm = base_grad.norm()
        patched_norm = patched_grad.norm()
        denom = base_norm * patched_norm
        if not base_grad.numel() or (float(base_norm) < 1e-12 and float(patched_norm) < 1e-12):
            cos = 1.0
        else:
            cos = float((base_grad @ patched_grad) / (denom + 1e-12))
        max_abs = max(max_abs, abs_max)
        max_mean = max(max_mean, abs_mean)
        min_cos = min(min_cos, cos)
        rows.append(
            {
                "name": name,
                "num_sampled": int(base_grad.numel()),
                "max_abs_diff": abs_max,
                "mean_abs_diff": abs_mean,
                "cosine": cos,
                "base_norm": float(base_norm),
                "patched_norm": float(patched_norm),
            }
        )
    return rows, {"max_abs_diff": max_abs, "max_mean_abs_diff": max_mean, "min_cosine": min_cos, "missing": missing}


def main() -> int:
    args = parse_args()

    import torch
    import forge

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    model = build_model(args)
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device="cuda")
    names = selected_param_names(model)
    if not names:
        raise RuntimeError("no representative parameters selected")

    base_loss, base_grads = run_loss_backward(model, input_ids, names, args.sample_elements)

    forge.patch(model, kernels=args.kernels)
    patched_counts = dict(model._forge_patched_counts)
    patched_loss, patched_grads = run_loss_backward(model, input_ids, names, args.sample_elements)
    forge.unpatch(model)

    rows, summary = compare_grads(base_grads, patched_grads)
    loss_diff = abs(patched_loss - base_loss)
    restored = not getattr(model, "_forge_patched", False)
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
        "kernels": args.kernels,
        "patched_counts": patched_counts,
        "selected_param_count": len(names),
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

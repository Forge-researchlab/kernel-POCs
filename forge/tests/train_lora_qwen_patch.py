"""Tiny LoRA training smoke test for Forge-patched Qwen.

This is intentionally artifact-producing: it saves a JSON metrics file, per-step
JSONL losses, and the trained PEFT adapter so a run can be inspected after the
fact. It is meant for CUDA/Linux validation, e.g. on the runpod:

  PYTHONPATH=forge:. UV_CACHE_DIR=/workspace/.uv-cache uv run python \
    forge/tests/train_lora_qwen_patch.py \
    --model Qwen/Qwen2.5-0.5B --steps 2 --seq-len 32 --batch 1 \
    --output-dir artifacts/forge_lora_qwen2_5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny LoRA Qwen run with forge.patch enabled.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=[
            "embedding",
            "rmsnorm",
            "swiglu",
            "rope",
            "cross_entropy",
            "fused_linear_ce",
            "lora_mlp",
            "lora_qkv",
        ],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    import forge
    from transformers import AutoModelForCausalLM

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install PEFT first: uv pip install peft") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training smoke test")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    output_dir = Path(args.output_dir or f"artifacts/forge_lora_qwen_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    forge.patch(model, kernels=args.kernels)
    patched_counts = dict(getattr(model, "_forge_patched_counts", {}))
    print(f"patched_counts = {patched_counts}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    losses_path = output_dir / "losses.jsonl"
    step_records = []

    started = time.time()
    with losses_path.open("w") as f:
        for step in range(args.steps):
            input_ids = torch.randint(
                low=0,
                high=model.config.vocab_size,
                size=(args.batch, args.seq_len),
                device="cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite or missing loss at step {step}: {loss}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            torch.cuda.synchronize()

            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
            step_records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(record)

    elapsed_s = time.time() - started
    forge.unpatch(model)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)

    metrics = {
        "model": args.model,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "steps": args.steps,
        "lr": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "kernels": args.kernels,
        "patched_counts": patched_counts,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "elapsed_s": elapsed_s,
        "losses": step_records,
        "adapter_dir": str(adapter_dir),
        "losses_jsonl": str(losses_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LoRA fine-tune Qwen2.5-0.5B on the pirate-speak dataset under FSDP2 + forge.patch.

Demo for the Forge team: real Qwen/Qwen2.5-0.5B weights, LoRA r=16, our §7-fixed
forge.patch(kernels=["lora_qkv","lora_mlp"]) wired through PEFT submodules and
FSDP2 fully_shard. Trains for ~200 steps, logs loss / peak VRAM / step time per
step to JSONL on rank 0, saves the LoRA adapter at the end.

Plotting + inference are separate scripts (plot_artifacts.py, run_inference_qwen.py).

Run:
    HF_HOME=/workspace/.hf-cache \
      torchrun --nproc-per-node=2 --standalone \
      forge/demos/train_lora_qwen_forge.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
sys.path.insert(0, _HERE)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

import forge
from pirate_dataset import TRAIN_PAIRS, build_full_example


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ARTIFACTS_DIR = Path("/workspace/kernel-POCs/artifacts/lora_demo_qwen")
LOG_PATH = ARTIFACTS_DIR / "train_log.jsonl"
RUN_META_PATH = ARTIFACTS_DIR / "run_meta.json"
ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"

SEED = 0
LORA_R = 16
LORA_ALPHA = 32
TRAIN_STEPS = 200
LR = 2e-4
BATCH = 2
MICRO_BATCHES = 4
MAX_SEQ_LEN = 96


# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------

def encode_example(tokenizer, english: str, pirate: str, max_len: int):
    prompt, completion = build_full_example(english, pirate)
    full = prompt + completion + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_ids = tokenizer(full, add_special_tokens=True).input_ids
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    n_prompt = min(len(prompt_ids), len(full_ids))

    labels = list(full_ids)
    for i in range(n_prompt):
        labels[i] = -100

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while len(full_ids) < max_len:
        full_ids.append(pad)
        labels.append(-100)
    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def make_batch(tokenizer, rng, batch_size, max_len):
    examples = [rng.choice(TRAIN_PAIRS) for _ in range(batch_size)]
    encoded = [encode_example(tokenizer, en, pi, max_len) for en, pi in examples]
    ids = torch.stack([e[0] for e in encoded])
    labels = torch.stack([e[1] for e in encoded])
    return ids, labels


# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------

def build_peft_model(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="eager",
    )
    model.config.use_cache = False
    model = model.to(device=device, dtype=dtype)

    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_cfg)
    peft_model = peft_model.to(dtype=dtype)
    return peft_model, tokenizer


def _inner_model(peft_model):
    return peft_model.base_model.model.model


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    is_rank0 = (rank == 0)
    if is_rank0:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print(f"Forge LoRA fine-tune: Qwen2.5-0.5B + forge.patch + FSDP2")
        print("=" * 70)
        print(f"  world={world}  device0={torch.cuda.get_device_name(0)}")
        print(f"  model={MODEL_ID}  lora_r={LORA_R} alpha={LORA_ALPHA}")
        print(f"  steps={TRAIN_STEPS}  lr={LR}  batch={BATCH}x{MICRO_BATCHES}={BATCH*MICRO_BATCHES}")
        print()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    t0 = time.time()
    peft_model, tokenizer = build_peft_model(device, dtype)
    if is_rank0:
        print(f"  loaded {MODEL_ID} in {time.time()-t0:.1f}s")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for layer in _inner_model(peft_model).layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(_inner_model(peft_model), mp_policy=mp_policy)

    forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
    if is_rank0:
        print(f"  forge.patch: {peft_model._forge_patched_counts}")
        n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in peft_model.parameters())
        print(f"  trainable params: {n_train:,} / {n_total:,} "
              f"({100*n_train/n_total:.2f}%)")
        print()

    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )

    torch.cuda.reset_peak_memory_stats()

    if is_rank0 and LOG_PATH.exists():
        LOG_PATH.unlink()

    peft_model.train()
    rng = random.Random(SEED + rank)
    losses = []

    for step in range(1, TRAIN_STEPS + 1):
        step_t0 = time.time()
        opt.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        for _ in range(MICRO_BATCHES):
            ids, labels = make_batch(tokenizer, rng, BATCH, MAX_SEQ_LEN)
            ids = ids.to(device)
            labels = labels.to(device)
            out = peft_model(input_ids=ids, labels=labels)
            (out.loss / MICRO_BATCHES).backward()
            step_loss_sum += float(out.loss.detach())
        opt.step()
        torch.cuda.synchronize()
        step_t = time.time() - step_t0
        loss = step_loss_sum / MICRO_BATCHES
        losses.append(loss)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        peaks = [None] * world
        dist.all_gather_object(peaks, peak_gb)

        if is_rank0:
            entry = {
                "step": step,
                "loss": loss,
                "step_time_s": step_t,
                "peak_vram_gb_per_rank": peaks,
            }
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            if step <= 5 or step % 10 == 0 or step == TRAIN_STEPS:
                print(f"  step {step:4d}  loss={loss:.4f}  "
                      f"time={step_t*1000:.0f}ms  peak_vram={max(peaks):.2f}GB")

    if is_rank0:
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        gathered = {}
        for name, p in peft_model.state_dict().items():
            if "lora_" not in name:
                continue
            if hasattr(p, "full_tensor"):
                gathered[name] = p.full_tensor().detach().cpu().clone()
            else:
                gathered[name] = p.detach().cpu().clone()
    else:
        for name, p in peft_model.state_dict().items():
            if "lora_" not in name:
                continue
            if hasattr(p, "full_tensor"):
                _ = p.full_tensor()

    if is_rank0:
        torch.save(gathered, ADAPTER_DIR / "lora_weights.pt")
        from peft import LoraConfig
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        cfg.save_pretrained(str(ADAPTER_DIR))

        final_peaks = [None] * world
        dist.all_gather_object(final_peaks, torch.cuda.max_memory_allocated() / 1e9)

        meta = {
            "model": MODEL_ID,
            "world_size": world,
            "device_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "steps": TRAIN_STEPS, "lr": LR,
            "batch_per_micro": BATCH, "micro_batches": MICRO_BATCHES,
            "effective_batch": BATCH * MICRO_BATCHES,
            "max_seq_len": MAX_SEQ_LEN,
            "forge_kernels": ["lora_qkv", "lora_mlp"],
            "patched_counts": dict(peft_model._forge_patched_counts),
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "final_peak_vram_gb_per_rank": final_peaks,
        }
        with RUN_META_PATH.open("w") as f:
            json.dump(meta, f, indent=2)
        print()
        print(f"  done. adapter -> {ADAPTER_DIR}")
        print(f"  log     -> {LOG_PATH}")
        print(f"  meta    -> {RUN_META_PATH}")
        print(f"  first loss {losses[0]:.4f}  final loss {losses[-1]:.4f}  "
              f"min loss {min(losses):.4f}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

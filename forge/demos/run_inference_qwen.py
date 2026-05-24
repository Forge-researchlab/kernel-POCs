"""Run inference on held-out pirate prompts, with and without the trained LoRA.

Loads Qwen2.5-0.5B on a single GPU. For each held-out English prompt:
  * Generate from BASE only (no adapter, no forge.patch).
  * Generate from BASE + trained LoRA adapter.
Writes results to artifacts/lora_demo_qwen/inference_samples.md.

Run after training completes:
    HF_HOME=/workspace/.hf-cache python forge/demos/run_inference_qwen.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
sys.path.insert(0, _HERE)

import torch

from pirate_dataset import HELD_OUT_PROMPTS, build_prompt


MODEL_ID = "Qwen/Qwen2.5-0.5B"
ARTIFACTS_DIR = Path("/workspace/kernel-POCs/artifacts/lora_demo_qwen")
ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"
OUT_PATH = ARTIFACTS_DIR / "inference_samples.md"

MAX_NEW_TOKENS = 32


def load_base(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="eager",
    ).to(device=device, dtype=dtype)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def attach_lora_adapter(base_model, device, dtype):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig.from_pretrained(str(ADAPTER_DIR))
    peft_model = get_peft_model(base_model, cfg)
    state = torch.load(ADAPTER_DIR / "lora_weights.pt", map_location=device,
                       weights_only=True)
    own_keys = set(peft_model.state_dict().keys())
    aligned = {}
    for k, v in state.items():
        if k in own_keys:
            aligned[k] = v
        else:
            print(f"  WARN: skipping unknown key {k}")
    missing, unexpected = peft_model.load_state_dict(aligned, strict=False)
    lora_missing = [m for m in missing if "lora_" in m]
    if lora_missing:
        print(f"  WARN: {len(lora_missing)} lora keys missing from load")
    peft_model = peft_model.to(device=device, dtype=dtype)
    peft_model.eval()
    return peft_model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)
    return decoded.strip()


def main():
    device = "cuda:0"
    dtype = torch.bfloat16

    print(f"Loading base {MODEL_ID} on {device} ...")
    base_model, tokenizer = load_base(device, dtype)

    print("Running BASE-model generations on held-out prompts ...")
    base_outputs = []
    for prompt_eng in HELD_OUT_PROMPTS:
        full = build_prompt(prompt_eng)
        gen = generate(base_model, tokenizer, full, device)
        base_outputs.append((prompt_eng, gen))
        print(f"  [base] {prompt_eng!r} -> {gen!r}")

    del base_model
    torch.cuda.empty_cache()
    print()
    print("Loading base again + attaching trained LoRA adapter ...")
    base_for_adapter, _ = load_base(device, dtype)
    peft_model = attach_lora_adapter(base_for_adapter, device, dtype)

    print("Running LoRA-tuned generations on held-out prompts ...")
    lora_outputs = []
    for prompt_eng in HELD_OUT_PROMPTS:
        full = build_prompt(prompt_eng)
        gen = generate(peft_model, tokenizer, full, device)
        lora_outputs.append((prompt_eng, gen))
        print(f"  [lora] {prompt_eng!r} -> {gen!r}")

    print(f"\nWriting {OUT_PATH} ...")
    with OUT_PATH.open("w") as f:
        f.write("# Forge LoRA demo (Qwen) — held-out inference samples\n\n")
        f.write(f"Model: `{MODEL_ID}` + LoRA adapter from "
                f"`{ADAPTER_DIR.relative_to(ARTIFACTS_DIR.parent)}`\n\n")
        f.write("All prompts are HELD OUT — they did **not** appear in training. "
                "Greedy decoding (`do_sample=False`), 32 new tokens.\n\n")
        f.write("| # | English | Base model | LoRA-tuned |\n")
        f.write("|---|---------|------------|------------|\n")
        for i, ((eng, base_out), (_, lora_out)) in enumerate(zip(base_outputs, lora_outputs), 1):
            base_clean = base_out.replace("|", "\\|").replace("\n", " ")
            lora_clean = lora_out.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {i} | {eng} | {base_clean} | {lora_clean} |\n")
        f.write("\n*Looking for pirate-speak style transfer in the LoRA column.*\n")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

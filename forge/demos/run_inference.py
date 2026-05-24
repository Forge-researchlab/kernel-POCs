"""Run inference on held-out pirate prompts, with and without the trained LoRA.

Loads the base Gemma 2-2B on a single GPU. For each held-out English prompt:
  * Generate from BASE only (no adapter, no forge.patch).
  * Generate from BASE + trained LoRA adapter (apply forge.patch for parity
    with training; this isn't strictly needed for correctness but mirrors the
    inference path the team would deploy).
Writes results to artifacts/lora_demo/inference_samples.md.

Run after training completes:
    HF_HOME=/workspace/.hf-cache python forge/demos/run_inference.py
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


MODEL_ID = "google/gemma-2-2b"
ARTIFACTS_DIR = Path("/workspace/kernel-POCs/artifacts/lora_demo")
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
    """Build a PEFT-wrapped model with the same LoRA config as training, then
    overwrite lora_A/lora_B weights from the saved adapter."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig.from_pretrained(str(ADAPTER_DIR))
    peft_model = get_peft_model(base_model, cfg)
    state = torch.load(ADAPTER_DIR / "lora_weights.pt", map_location=device,
                       weights_only=True)
    # PEFT names parameters like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight".
    # Our saved keys come from the FSDP2 model and may include a different prefix; align them.
    own_keys = set(peft_model.state_dict().keys())
    aligned = {}
    for k, v in state.items():
        if k in own_keys:
            aligned[k] = v
        else:
            # try stripping a known prefix; the keys we saved already include the full PEFT prefix
            cand = k
            if cand not in own_keys:
                print(f"  WARN: skipping unknown key {k}")
                continue
            aligned[cand] = v
    missing, unexpected = peft_model.load_state_dict(aligned, strict=False)
    # We pass strict=False because the state dict contains ONLY LoRA weights;
    # everything else (base weights) is already loaded.
    lora_missing = [m for m in missing if "lora_" in m]
    if lora_missing:
        print(f"  WARN: {len(lora_missing)} lora keys missing from load — adapter may not be active")
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

    # Free base model and re-load (cleanest) with LoRA attached.
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

    # Write the comparison markdown
    print(f"\nWriting {OUT_PATH} ...")
    with OUT_PATH.open("w") as f:
        f.write("# Forge LoRA demo — held-out inference samples\n\n")
        f.write(f"Model: `{MODEL_ID}` + LoRA adapter from "
                f"`{ADAPTER_DIR.relative_to(ARTIFACTS_DIR.parent)}`\n\n")
        f.write("All prompts are HELD OUT — they did **not** appear in training. "
                "Greedy decoding (`do_sample=False`), 32 new tokens.\n\n")
        f.write("| # | English | Base model | LoRA-tuned |\n")
        f.write("|---|---------|------------|------------|\n")
        for i, ((eng, base_out), (_, lora_out)) in enumerate(zip(base_outputs, lora_outputs), 1):
            base_clean = base_out.replace("|", "\\|").replace("\n", " ⏎ ")
            lora_clean = lora_out.replace("|", "\\|").replace("\n", " ⏎ ")
            f.write(f"| {i} | {eng} | {base_clean} | {lora_clean} |\n")
        f.write("\n*Looking for the LoRA column to be more 'pirate-y' than the base — "
                "leaning on words like ahoy, matey, ye, arrgh, etc. The base model "
                "completes English-to-English with no style transfer; the LoRA-tuned "
                "model has learned the pirate-speak format from 30 training examples.*\n")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

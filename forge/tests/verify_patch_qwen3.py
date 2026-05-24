"""Real-model verification for forge.patch on Qwen2.5/Qwen3-style models.

Runs the bisection pattern from design_details.html#wiring:

    baseline -> patch(kernels=["embedding"]) -> compare
    baseline -> patch(kernels=["rope"])      -> compare    # the brittle one
    baseline -> patch(kernels=["rmsnorm"])   -> compare    # Qwen2RMSNorm path
    baseline -> patch(kernels=["swiglu"])    -> compare    # Qwen2MLP/Qwen3MLP path
    baseline -> patch()                       -> compare
    baseline -> unpatch                       -> must match baseline EXACTLY

Each comparison reports:
    - NaN in patched output? (must be False)
    - shape matches?         (must be True)
    - max abs diff in logits (expect < 1.0 for bf16, typically << 0.1)
    - cosine similarity      (expect > 0.999)

Run:    python forge/tests/verify_patch_qwen3.py
Or:     python forge/tests/verify_patch_qwen3.py --model Qwen/Qwen2.5-0.5B

Exit code: 0 on PASS, 1 on any FAIL (so it's CI-pinnable).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# Make the forge package importable when this file is run as a script from
# anywhere (e.g., `python forge/tests/verify_patch_qwen3.py`).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _summary(out_orig, out_new, label):
    nan_count = int(out_new.isnan().sum().item())
    shape_ok = (out_new.shape == out_orig.shape)
    max_diff = (out_new - out_orig).abs().max().item()
    flat_orig = out_orig.float().flatten()
    flat_new = out_new.float().flatten()
    cos = float(
        (flat_orig @ flat_new) /
        (flat_orig.norm() * flat_new.norm() + 1e-12)
    )
    print(f"  [{label}] shape={tuple(out_new.shape)} "
          f"NaN={nan_count} max_diff={max_diff:.4e} cos_sim={cos:.6f}")
    return nan_count == 0 and shape_ok and max_diff < 1.0 and cos > 0.999


def _run_patch_case(model, ids, out_orig, label, kernels):
    import forge

    print(f"\nforge.patch(model, kernels={kernels!r}) ...")
    try:
        forge.patch(model, kernels=kernels)
        print(f"  patched_counts = {model._forge_patched_counts}")
        if kernels == ["rope"]:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
            print(f"  qwen2.apply_rotary_pos_emb = {qwen2_mod.apply_rotary_pos_emb.__name__}")
            try:
                import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
                print(f"  qwen3.apply_rotary_pos_emb = {qwen3_mod.apply_rotary_pos_emb.__name__}")
            except Exception:
                pass
        with __import__("torch").no_grad():
            out_new = model(ids).logits
        return _summary(out_orig, out_new, label)
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import forge

    print(f"\n=== forge.patch verification on {args.model} ===")
    print(f"forge version: {forge.__version__}")
    print(f"torch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM
    import transformers
    print(f"transformers: {transformers.__version__}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(args.seed)
    print(f"loading {args.model} (dtype={dtype}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  model_type = {model.config.model_type!r}")
    print(f"  hidden_size = {model.config.hidden_size}, "
          f"num_layers = {getattr(model.config, 'num_hidden_layers', '?')}")

    # Module class census — confirms our mapping keys actually match
    from collections import Counter
    cls_census = Counter(type(m).__name__ for _, m in model.named_modules())
    from forge.patching.qwen3 import QWEN3_MAPPING
    print("\n  module class census (filtered to mapping-relevant):")
    for k in sorted(QWEN3_MAPPING):
        found = cls_census.get(k, 0)
        print(f"    {k:20s}  count={found}  ({'mapped' if found else 'NOT FOUND'})")

    ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device=device)

    # === Baseline ===
    print("\n[1/7] Baseline (unpatched) forward ...")
    with torch.no_grad():
        out_orig = model(ids).logits.clone()
    print(f"  baseline logits: shape={tuple(out_orig.shape)} "
          f"dtype={out_orig.dtype} NaN={int(out_orig.isnan().sum())}")

    results = {}

    # === Embedding only ===
    print("\n[2/7] forge.patch(model, kernels=['embedding']) ...")
    try:
        forge.patch(model, kernels=["embedding"])
        print(f"  patched_counts = {model._forge_patched_counts}")
        with torch.no_grad():
            out_emb = model(ids).logits
        results["embedding"] = _summary(out_orig, out_emb, "embedding")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["embedding"] = False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)

    # === RoPE only (the brittle one — verifies the cos/sin shape contract) ===
    print("\n[3/7] forge.patch(model, kernels=['rope']) ...")
    try:
        forge.patch(model, kernels=["rope"])
        print(f"  patched_counts = {model._forge_patched_counts}  "
              f"(module-level RoPE swap active)")
        # Confirm the swap actually happened
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        print(f"  apply_rotary_pos_emb = {qwen2_mod.apply_rotary_pos_emb.__name__}")
        with torch.no_grad():
            out_rope = model(ids).logits
        results["rope"] = _summary(out_orig, out_rope, "rope")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["rope"] = False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)

    # === RMSNorm only (Qwen2/3RMSNorm — offset=0.0, casting_mode='llama') ===
    print("\n[4/7] forge.patch(model, kernels=['rmsnorm']) ...")
    try:
        forge.patch(model, kernels=["rmsnorm"])
        print(f"  patched_counts = {model._forge_patched_counts}")
        with torch.no_grad():
            out_rmsnorm = model(ids).logits
        results["rmsnorm"] = _summary(out_orig, out_rmsnorm, "rmsnorm")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["rmsnorm"] = False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)

    # === SwiGLU only (Qwen2MLP/Qwen3MLP — replaces full gate/up/down with
    # gate_proj + up_proj + Forge swiglu(silu(g)*u) + down_proj). ===
    print("\n[5/7] forge.patch(model, kernels=['swiglu']) ...")
    try:
        forge.patch(model, kernels=["swiglu"])
        print(f"  patched_counts = {model._forge_patched_counts}")
        with torch.no_grad():
            out_swiglu = model(ids).logits
        results["swiglu"] = _summary(out_orig, out_swiglu, "swiglu")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["swiglu"] = False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)

    # === Both (default: everything that's wired) ===
    print("\n[6/7] forge.patch(model)  (all wired kernels) ...")
    try:
        forge.patch(model)
        print(f"  patched_counts = {model._forge_patched_counts}")
        with torch.no_grad():
            out_all = model(ids).logits
        results["all"] = _summary(out_orig, out_all, "all")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["all"] = False
    finally:
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)

    # === Unpatch must restore EXACTLY ===
    print("\n[7/7] unpatch restoration check ...")
    with torch.no_grad():
        out_restored = model(ids).logits
    exact_restore = bool(torch.equal(out_restored, out_orig))
    print(f"  bit-exact restore = {exact_restore}")
    results["unpatch_exact"] = exact_restore

    # === Verdict ===
    print("\n=== VERDICT ===")
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {name:18s}  {'PASS' if ok else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

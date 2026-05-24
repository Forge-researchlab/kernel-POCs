"""End-to-end LoRA-PEFT integration test for forge.patch on Gemma 2.

Tested against PEFT 0.19.x. Builds a tiny random-init Gemma 2, wraps it in
PEFT LoRA, and verifies that the existing forge.patch LoRA factories (in
forge/forge/patching/kernels/lora.py) intercept the attention and MLP blocks
correctly — with Gemma-specific branches:

  * make_lora_mlp_forward routes through the GeGLU kernel (Gemma 2 uses
    `gelu_pytorch_tanh`, not SiLU); we can't reuse LoRAMLPv3 here because it
    hardcodes SiLU. The Gemma path leans on PEFT's projection forwards
    (which already include LoRA-A/B) and only fuses the GeGLU activation.
  * make_lora_qkv_forward forwards `softcap=` to the attention interface
    (Gemma 2 sets `attn_logit_softcapping`; Qwen 3 does not), and accepts
    both `past_key_value` (Qwen) and `past_key_values` (Gemma 2) naming.

Test sections:
  1. PEFT availability + version
  2. Patching analysis printout (LoRA mapping entries + Gemma routing)
  3. Module class census (Gemma2Attention, Gemma2MLP, PEFT-wrapped Linears)
  4. Baseline forward (PEFT-wrapped, unpatched)
  5. Per-kernel bisection: lora_qkv only / lora_mlp only / both
  6. Backward parity on q_proj and gate_proj LoRA params
  7. ForgeSkipPatch smoke test (disable_adapter_layers)
  8. Bit-exact unpatch
  9. Negative tests (double-patch, unknown kernel)
 10. Mini-convergence (20 SGD steps, patched vs PEFT-only)

Run:    python forge/tests/verify_patch_lora_gemma.py
Exit 0 on PASS, 1 on any FAIL.
"""
from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import forge  # noqa: F401, E402  -- triggers POC-root sys.path injection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cos_sim(a, b):
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _summary(out_orig, out_new, label, max_diff_thresh=5e-2, cos_thresh=0.999):
    nan = int(out_new.isnan().sum().item())
    shape_ok = (out_new.shape == out_orig.shape)
    max_diff = (out_new - out_orig).abs().max().item()
    cos = _cos_sim(out_orig, out_new)
    ok = (nan == 0 and shape_ok and max_diff < max_diff_thresh and cos > cos_thresh)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{label:14s}] shape={tuple(out_new.shape)} NaN={nan} "
          f"max_diff={max_diff:.3e} cos={cos:.6f}  {flag}")
    return ok


def _grad_summary(g_orig, g_new, label, rel_max_diff_thresh=5e-2, cos_thresh=0.9999):
    """Gradient parity check.

    Cosine similarity is undefined (cos = 0) when either side is the zero
    vector. This is the EXPECTED case for `lora_A` gradients on a freshly-
    initialized PEFT model: PEFT inits `lora_B = 0` so the chain rule zeros
    the LoRA-A gradient until B starts moving. In that regime the only
    meaningful check is that BOTH sides agree the gradient is zero — which
    `max_diff == 0` (or below noise floor) captures cleanly.
    """
    nan = int(g_new.isnan().sum().item())
    shape_ok = (g_new.shape == g_orig.shape)
    max_diff = (g_new - g_orig).abs().max().item()
    base_max = g_orig.abs().max().item()
    new_max = g_new.abs().max().item()
    rel = max_diff / (base_max + 1e-12)
    cos = _cos_sim(g_orig, g_new)

    # Trivial-zero case: both sides are (near-)zero — agreement, not failure.
    trivial_zero = (base_max < 1e-6 and new_max < 1e-6 and max_diff < 1e-6)

    if trivial_zero:
        ok = (nan == 0 and shape_ok)
        note = "  (both zero — trivial agree)"
    else:
        ok = (nan == 0 and shape_ok and rel < rel_max_diff_thresh and cos > cos_thresh)
        note = ""
    flag = "PASS" if ok else "FAIL"
    print(f"  [grad: {label:36s}] NaN={nan} base_max={base_max:.2e} "
          f"new_max={new_max:.2e} max_diff={max_diff:.2e} rel={rel:.2e} "
          f"cos={cos:.6f}  {flag}{note}")
    return ok


def _build_tiny_peft_gemma(device, dtype, seed=0):
    """Build a tiny random-init Gemma 2 + wrap with PEFT LoRA."""
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = Gemma2Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,           # >=2 so we hit full + sliding-window alternation
        num_attention_heads=4,
        num_key_value_heads=2,         # GQA
        head_dim=32,
        max_position_embeddings=256,
        hidden_activation="gelu_pytorch_tanh",
    )
    torch.manual_seed(seed)
    model = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
    )
    peft_model = get_peft_model(model, peft_cfg)
    return peft_model


def _inner_model(peft_model):
    """Reach the inner Gemma2Model (past PEFT wrappers) to walk modules."""
    return peft_model.base_model.model.model


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def _print_patching_analysis():
    from forge.patching.gemma import GEMMA_MAPPING
    from forge.patching.kernels import FORWARD_MAKERS

    print("=" * 78)
    print("PATCHING STRATEGY — forge.patch on a PEFT-wrapped Gemma 2 model")
    print("=" * 78)
    print()
    print("Architecture detection routes through GEMMA_MAPPING. Per-class specs")
    print("are tried in list order — the first one whose factory doesn't raise")
    print("ForgeSkipPatch wins (LoRA factory raises when no PEFT adapter exists,")
    print("falling through to the plain GeGLU adapter).")
    print()
    for cls_name, specs in GEMMA_MAPPING.items():
        spec_list = [specs] if isinstance(specs, tuple) else specs
        for kernel, cfg in spec_list:
            maker = FORWARD_MAKERS.get(kernel)
            stub = getattr(maker, "__forge_stub__", False) if maker else True
            status = "STUB" if stub else "real"
            print(f"  {cls_name:18s} -> kernel={kernel!r:14s} cfg={cfg!r:24s} [{status}]")
    print()


def _module_census(peft_model):
    import peft
    from collections import Counter
    cls_census = Counter(type(m).__name__ for _, m in _inner_model(peft_model).named_modules())
    print("Module class census:")
    for k in ("Gemma2Attention", "Gemma2MLP", "Gemma2RMSNorm", "Embedding"):
        print(f"    {k:18s}  count={cls_census.get(k, 0)}")
    # Confirm PEFT wrapped the projections
    sample_q = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    print(f"  q_proj is PEFT Linear: {type(sample_q).__name__} "
          f"(has lora_A={hasattr(sample_q, 'lora_A')})")
    print()


def _section_bisection(peft_model, ids, out_orig):
    """Per-kernel bisection. Each run is a fresh patch + unpatch."""
    results = {}
    for label, kernels in [
        ("lora_qkv only", ["lora_qkv"]),
        ("lora_mlp only", ["lora_mlp"]),
        ("both",          ["lora_qkv", "lora_mlp"]),
    ]:
        print(f"\n  forge.patch(model, kernels={kernels!r}) ...")
        try:
            forge.patch(peft_model, kernels=kernels)
            print(f"    patched_counts = {peft_model._forge_patched_counts}")
            import torch
            with torch.no_grad():
                out = peft_model(ids).logits
            results[label] = _summary(out_orig, out, label)
        except Exception as e:
            print(f"    EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[label] = False
        finally:
            if getattr(peft_model, "_forge_patched", False):
                forge.unpatch(peft_model)
    return results


def _section_backward(peft_model, ids):
    """Backward parity on representative LoRA params (q_proj + gate_proj A/B)."""
    import torch

    # Both runs need grads enabled
    for p in peft_model.parameters():
        if p.requires_grad is False and hasattr(p, "_was_trainable"):
            continue
    peft_model.train()

    # Names go through the PEFT wrapper chain
    target_names = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight",
    ]
    name_to_param = dict(peft_model.named_parameters())
    missing = [n for n in target_names if n not in name_to_param]
    if missing:
        print(f"  SKIP: missing params: {missing}")
        return False

    def _run():
        peft_model.zero_grad(set_to_none=True)
        out = peft_model(ids).logits
        out.float().sum().backward()
        return {n: name_to_param[n].grad.detach().clone() for n in target_names}

    print("\n  computing PEFT-baseline gradients (unpatched) ...")
    base = _run()
    print(f"    grads computed: {[n.rsplit('.', 4)[-3:] for n in target_names]}")

    print("\n  patching kernels=['lora_qkv','lora_mlp'] and recomputing ...")
    forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
    print(f"    patched_counts = {peft_model._forge_patched_counts}")
    try:
        new = _run()
    finally:
        forge.unpatch(peft_model)

    ok = True
    for n in target_names:
        short = n.replace("base_model.model.model.", "").replace(".weight", "")
        ok &= _grad_summary(base[n], new[n], short)
    return ok


def _section_skip_patch_when_adapters_disabled(peft_model, ids):
    import torch
    print("  disabling PEFT adapters and patching ...")
    peft_model.disable_adapter_layers()
    try:
        forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
        counts = peft_model._forge_patched_counts
        print(f"    patched_counts = {counts}")
        # Both must be 0 — the factory should raise ForgeSkipPatch when no active adapter
        ok = (counts.get("lora_qkv", 0) == 0 and counts.get("lora_mlp", 0) == 0)
        print(f"  -> {'PASS' if ok else 'FAIL'} (expected 0 patches when adapters disabled)")
    finally:
        if getattr(peft_model, "_forge_patched", False):
            forge.unpatch(peft_model)
        peft_model.enable_adapter_layers()
    return ok


def _section_unpatch_bit_exact(peft_model, ids, out_orig):
    import torch
    forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
    forge.unpatch(peft_model)
    with torch.no_grad():
        out = peft_model(ids).logits
    exact = bool(torch.equal(out, out_orig))
    print(f"  bit-exact restore = {exact}  {'PASS' if exact else 'FAIL'}")
    return exact


def _section_negative_tests(peft_model):
    ok = True
    # double-patch
    print("  double-patch must raise RuntimeError ...")
    forge.patch(peft_model, kernels=["lora_mlp"])
    try:
        try:
            forge.patch(peft_model, kernels=["lora_mlp"])
            print("    no exception — FAIL")
            ok = False
        except RuntimeError as e:
            print(f"    RuntimeError (expected): {e}")
    finally:
        forge.unpatch(peft_model)

    # unknown kernel
    print("  unknown kernel must raise KeyError ...")
    try:
        forge.patch(peft_model, kernels=["nope_kernel"])
        if getattr(peft_model, "_forge_patched", False):
            forge.unpatch(peft_model)
        print("    no exception — FAIL")
        ok = False
    except KeyError as e:
        print(f"    KeyError (expected): {e}")
    return ok


def _section_convergence(device, dtype, steps=20, seed=0):
    """20 SGD steps on a fixed batch. Two identical PEFT-wrapped models —
    one patched, one not — must trace the same loss curve."""
    import torch

    print(f"  building two identical PEFT-wrapped Gemma2 models (seed={seed}) ...")
    model_a = _build_tiny_peft_gemma(device, dtype, seed=seed)
    model_b = _build_tiny_peft_gemma(device, dtype, seed=seed)

    # Sanity: identical trainable params at init
    a_params = dict(model_a.named_parameters())
    b_params = dict(model_b.named_parameters())
    for name in a_params:
        pa, pb = a_params[name], b_params[name]
        if pa.requires_grad:
            assert torch.equal(pa, pb), f"init mismatch on {name}"

    forge.patch(model_b, kernels=["lora_qkv", "lora_mlp"])
    print(f"  model B patched: {model_b._forge_patched_counts}")

    opt_a = torch.optim.SGD([p for p in model_a.parameters() if p.requires_grad],
                            lr=1e-3, momentum=0.9)
    opt_b = torch.optim.SGD([p for p in model_b.parameters() if p.requires_grad],
                            lr=1e-3, momentum=0.9)

    torch.manual_seed(seed + 100)
    ids = torch.randint(0, _inner_model(model_a).config.vocab_size,
                        (2, 32), device=device)
    labels = ids.clone()

    print(f"\n  {'step':>5} {'loss_A':>10} {'loss_B':>10} {'rel_diff':>10}")
    rel_diffs = []
    losses_a, losses_b = [], []
    fail_step = None
    for step in range(1, steps + 1):
        opt_a.zero_grad(set_to_none=True)
        out_a = model_a(ids, labels=labels).loss
        out_a.backward(); opt_a.step()

        opt_b.zero_grad(set_to_none=True)
        out_b = model_b(ids, labels=labels).loss
        out_b.backward(); opt_b.step()

        la = float(out_a.detach())
        lb = float(out_b.detach())
        rel = abs(la - lb) / (abs(la) + 1e-12)
        rel_diffs.append(rel); losses_a.append(la); losses_b.append(lb)
        if step <= 5 or step % 5 == 0 or step == steps:
            print(f"  {step:5d} {la:10.4f} {lb:10.4f} {rel:10.4f}")
        if rel > 0.02 and fail_step is None:
            fail_step = step

    forge.unpatch(model_b)

    converging_a = losses_a[-1] < losses_a[0]
    converging_b = losses_b[-1] < losses_b[0]
    ok = (fail_step is None and converging_a and converging_b
          and rel_diffs[-1] < 0.02)
    print(f"\n  rel_diff max={max(rel_diffs):.4e}  final={rel_diffs[-1]:.4e}")
    print(f"  converging? A={converging_a}  B={converging_b}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch
    if not torch.cuda.is_available():
        print("CUDA required"); return 1

    # === [1] PEFT availability ===
    print("=" * 78)
    print("[1] PEFT availability")
    print("=" * 78)
    try:
        import peft
        print(f"  peft version: {peft.__version__}")
    except ImportError:
        print("  PEFT not installed — pip install peft. SKIP.")
        return 1

    print(f"\n  forge {forge.__version__} on {torch.cuda.get_device_name(0)}")
    print(f"  torch {torch.__version__}\n")

    # === [2] Patching analysis ===
    _print_patching_analysis()

    device = "cuda"
    dtype = torch.bfloat16

    # === [3] Module census ===
    print("=" * 78)
    print("[3] Module class census on PEFT-wrapped Gemma 2")
    print("=" * 78)
    peft_model = _build_tiny_peft_gemma(device, dtype, seed=0)
    _module_census(peft_model)

    ids = torch.randint(0, _inner_model(peft_model).config.vocab_size,
                        (2, 32), device=device)

    # === [4] Baseline ===
    print("=" * 78)
    print("[4] Baseline forward (PEFT-wrapped, unpatched)")
    print("=" * 78)
    peft_model.eval()
    with torch.no_grad():
        out_orig = peft_model(ids).logits.clone()
    print(f"  logits shape={tuple(out_orig.shape)} NaN={int(out_orig.isnan().sum())}")

    results = {}

    # === [5] Bisection ===
    print("\n" + "=" * 78)
    print("[5] Per-kernel bisection")
    print("=" * 78)
    results.update(_section_bisection(peft_model, ids, out_orig))

    # === [6] Backward parity ===
    print("\n" + "=" * 78)
    print("[6] Backward gradient parity")
    print("=" * 78)
    # Fresh model so eval()/grad state doesn't leak from section [5]
    peft_model_grad = _build_tiny_peft_gemma(device, dtype, seed=1)
    ids_grad = torch.randint(0, _inner_model(peft_model_grad).config.vocab_size,
                             (2, 32), device=device)
    results["backward"] = _section_backward(peft_model_grad, ids_grad)
    del peft_model_grad

    # === [7] ForgeSkipPatch smoke ===
    print("\n" + "=" * 78)
    print("[7] ForgeSkipPatch when adapters disabled")
    print("=" * 78)
    peft_model.eval()
    results["skip_patch_when_disabled"] = _section_skip_patch_when_adapters_disabled(peft_model, ids)

    # === [8] Bit-exact unpatch ===
    print("\n" + "=" * 78)
    print("[8] Bit-exact unpatch restoration")
    print("=" * 78)
    peft_model.eval()
    results["unpatch_exact"] = _section_unpatch_bit_exact(peft_model, ids, out_orig)

    # === [9] Negative tests ===
    print("\n" + "=" * 78)
    print("[9] Negative tests")
    print("=" * 78)
    results["negative"] = _section_negative_tests(peft_model)
    del peft_model

    # === [10] Mini-convergence ===
    print("\n" + "=" * 78)
    print("[10] Mini-convergence (20 SGD steps)")
    print("=" * 78)
    results["convergence"] = _section_convergence(device, dtype)

    # === Verdict ===
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for name, ok in results.items():
        print(f"  {name:30s}  {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  OVERALL ({sum(results.values())}/{len(results)}): "
          f"{'PASS' if all_ok else 'FAIL'}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

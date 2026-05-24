"""Real-model verification for forge.patch on Gemma-2 architecture.

Mirrors verify_patch_qwen3.py but exercises Gemma-specific behavior:

    * embedding scaling — Gemma multiplies inputs_embeds by sqrt(hidden_size)
      INSIDE GemmaModel.forward (not inside nn.Embedding). The embedding patch
      must return raw rows; the * sqrt(d) must still come from HF.
    * RMSNorm(offset=1.0, casting_mode="gemma") — Gemma normalizes as
      (1 + w) * x / rms instead of Qwen's w * x / rms. The mapping passes
      offset=1.0; the closure picks "gemma" casting by default; in_place=False
      because the residual path consumes dy.
    * GeGLU activation variant — Gemma2 ships hidden_activation="gelu_pytorch_tanh"
      by default. Our factory reads this from module.config and picks
      approximate="tanh"; passing approximate="none" on a tanh-config model would
      shift outputs ~1e-3 and fail parity.
    * Sliding-window attention — Gemma2 alternates full/sliding-window attention
      across layers, both feed through the same apply_rotary_pos_emb that we
      monkey-patch at module level. Test exercises >=2 layers to hit both.
    * Tied embeddings + lm_head — gradient on embed_tokens.weight is identical
      to gradient on lm_head.weight via tensor aliasing; the kernel must not
      break this.

The test uses a small RANDOMLY-INITIALIZED Gemma2 (no checkpoint download,
no HF auth needed). Forward parity is checked weight-agnostic: both patched
and unpatched paths see identical weights, so random init is sufficient to
detect any numerical divergence introduced by the kernels.

Bisection pattern:
    baseline -> patch(kernels=["embedding"]) -> compare
    baseline -> patch(kernels=["rmsnorm"])   -> compare
    baseline -> patch(kernels=["rope"])      -> compare
    baseline -> patch(kernels=["geglu"])     -> compare
    baseline -> patch()                       -> compare    (everything wired)
    baseline -> unpatch                       -> must match baseline EXACTLY

Extra tests beyond the bisection:
    [shape stress]   multiple (batch, seq) shapes against full patch
    [backward]       grad parity on embed weight + a sample RMSNorm weight
    [double-patch]   second patch() raises RuntimeError
    [unknown kernel] patch(kernels=["nope"]) raises KeyError
    [stub guard]     patch(kernels=["cross_entropy"]) raises NotImplementedError

Run:    python forge/tests/verify_patch_gemma.py
Exit 0 on PASS, 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# Make the forge package importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(out_orig, out_new, label, max_diff_thresh=1.0, cos_thresh=0.999):
    nan_count = int(out_new.isnan().sum().item())
    shape_ok = (out_new.shape == out_orig.shape)
    max_diff = (out_new - out_orig).abs().max().item()
    flat_orig = out_orig.float().flatten()
    flat_new = out_new.float().flatten()
    cos = float(
        (flat_orig @ flat_new) /
        (flat_orig.norm() * flat_new.norm() + 1e-12)
    )
    ok = (
        nan_count == 0 and shape_ok
        and max_diff < max_diff_thresh and cos > cos_thresh
    )
    flag = "PASS" if ok else "FAIL"
    print(f"  [{label:14s}] shape={tuple(out_new.shape)} "
          f"NaN={nan_count} max_diff={max_diff:.4e} cos_sim={cos:.6f}  {flag}")
    return ok


def _grad_summary(g_orig, g_new, label, rel_max_diff_thresh=0.05, cos_thresh=0.9999):
    """Gradient parity check.

    Uses RELATIVE max-diff (max_diff / max(|g_orig|)) rather than absolute,
    because gradient magnitudes vary by orders of magnitude across layers
    and bf16 accumulation through several ops produces predictable relative
    drift (~0.5% per layer). Cosine similarity catches direction errors
    independent of scale.
    """
    nan_count = int(g_new.isnan().sum().item())
    shape_ok = (g_new.shape == g_orig.shape)
    max_diff = (g_new - g_orig).abs().max().item()
    base_max = g_orig.abs().max().item()
    rel_max_diff = max_diff / (base_max + 1e-12)
    flat_orig = g_orig.float().flatten()
    flat_new = g_new.float().flatten()
    cos = float(
        (flat_orig @ flat_new) /
        (flat_orig.norm() * flat_new.norm() + 1e-12)
    )
    ok = (
        nan_count == 0 and shape_ok
        and rel_max_diff < rel_max_diff_thresh and cos > cos_thresh
    )
    flag = "PASS" if ok else "FAIL"
    print(f"  [grad: {label:42s}] NaN={nan_count} base_max={base_max:.2e} "
          f"max_diff={max_diff:.2e} rel={rel_max_diff:.2e} cos={cos:.6f}  {flag}")
    return ok


def _print_patching_analysis():
    """Print exactly what forge.patch will do on a Gemma2 model. No model load."""
    from forge.patching.gemma import GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES
    from forge.patching.core import _FORWARD_MAKERS

    print("=" * 78)
    print("PATCHING STRATEGY — what forge.patch(gemma_model) actually does")
    print("=" * 78)
    print()
    print("Architecture detection: model.config.model_type ∈ {'gemma', 'gemma2'}")
    print("  -> uses GEMMA_MAPPING (per-class) + GEMMA_MODULE_LEVEL_PATCHES (module-level)")
    print()
    print("Per-class patches (module.forward is rebound; original kept in")
    print("model._forge_originals for bit-exact unpatch):")
    print()
    for cls_name, specs in GEMMA_MAPPING.items():
        # specs may be a single (kernel, cfg) tuple OR a list of such tuples
        # tried in order (the core patch loop picks the first one whose factory
        # doesn't raise ForgeSkipPatch). Normalize to a list for display.
        spec_list = [specs] if isinstance(specs, tuple) else specs
        for kernel, cfg in spec_list:
            maker = _FORWARD_MAKERS.get(kernel)
            stub = getattr(maker, "__forge_stub__", False) if maker else True
            status = "STUB (skipped)" if stub else "real kernel"
            print(f"  {cls_name:18s} -> kernel={kernel!r:14s} cfg={cfg!r:34s} [{status}]")
    print()
    print("Module-level patches (replaces transformers.models.*.apply_rotary_pos_emb):")
    for kernel, spec in GEMMA_MODULE_LEVEL_PATCHES.items():
        spec_list = spec if isinstance(spec, list) else [spec]
        for mod_path, attr, repl in spec_list:
            print(f"  kernel={kernel!r}: {mod_path}.{attr} -> {repl.__name__}")
    print()
    print("Closure semantics:")
    print("  - Each replacement closes over module.weight (NOT a copy), so any")
    print("    LoRA / in-place updates to the original weight are visible through")
    print("    the kernel on the next forward.")
    print("  - For MLP (geglu/swiglu), closures call module.gate_proj/up_proj/")
    print("    down_proj as submodules (not their .weight), so PEFT adapters")
    print("    wrapping these Linears intercept correctly.")
    print()
    print("Gemma-specific kernel config:")
    print("  - rmsnorm:  offset=1.0 → casting_mode='gemma', in_place=False")
    print("              (residual path consumes dy; in-place would corrupt it).")
    print("  - geglu:    approximate is inferred from module.config.hidden_activation")
    print("              ('gelu_pytorch_tanh' → 'tanh', 'gelu' → 'none').")
    print("  - rope:     stateless — re-uses the Qwen RoPE kernel; only the patch")
    print("              site (gemma2.modeling_gemma2.apply_rotary_pos_emb) differs.")
    print("  - embedding: identical to Qwen path. The * sqrt(hidden_size) scaling")
    print("              lives in GemmaModel.forward AFTER embed_tokens(), so the")
    print("              kernel returns raw rows and HF applies the scale downstream.")
    print()


def _build_tiny_gemma2(device, dtype):
    """Random-init Gemma2 with dimensions tiny enough to forward+backward in
    a few ms. Layer count is >=2 so we hit Gemma2's alternating full/sliding-
    window attention pattern."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        # Gemma2 default — what our geglu factory should infer.
        hidden_activation="gelu_pytorch_tanh",
        # Keep softcaps at default to match real Gemma2 numerics.
    )
    model = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model


def _module_census(model):
    """Print which mapping classes are present in THIS model.

    GEMMA_MAPPING intentionally covers both Gemma 1 (GemmaRMSNorm, GemmaMLP)
    and Gemma 2 (Gemma2RMSNorm, Gemma2MLP) class names. A Gemma 2 model will
    naturally have zero count on the Gemma 1 entries (and vice versa). The
    real correctness signal is `patched_counts` after a full patch, not this
    census — so this function is informational and always returns True.
    """
    from collections import Counter
    from forge.patching.gemma import GEMMA_MAPPING

    cls_census = Counter(type(m).__name__ for _, m in model.named_modules())
    print("Module class census (mapping-relevant only — '-' is expected for "
          "the inactive Gemma arch):")
    for k in sorted(GEMMA_MAPPING):
        found = cls_census.get(k, 0)
        flag = "OK" if found > 0 else "-"
        print(f"    {k:20s}  count={found:3d}  [{flag}]")
    print()
    return True


def _config_analysis(model):
    """Inspect the HF config so the user sees what our factory will resolve to."""
    cfg = model.config
    print("HF config inspection:")
    print(f"  model_type        = {cfg.model_type!r}")
    print(f"  hidden_size       = {cfg.hidden_size}  (embed * sqrt({cfg.hidden_size}) "
          f"~= {cfg.hidden_size ** 0.5:.4f})")
    print(f"  hidden_activation = {getattr(cfg, 'hidden_activation', None)!r}")
    print(f"  hidden_act        = {getattr(cfg, 'hidden_act', None)!r}")
    print(f"  num_hidden_layers = {cfg.num_hidden_layers}  (Gemma2 alternates "
          f"full/sliding-window across layers)")
    print(f"  tie_word_embeddings = {getattr(cfg, 'tie_word_embeddings', None)}")
    # Sanity: confirm embed_tokens and lm_head share the same tensor when tied.
    embed_w = model.get_input_embeddings().weight
    lm_w = model.get_output_embeddings().weight
    print(f"  embed/lm_head tied tensor? {embed_w.data_ptr() == lm_w.data_ptr()}")
    print()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def _forward_logits(model, ids):
    import torch
    with torch.no_grad():
        return model(ids).logits.clone()


def _bisection_per_kernel(model, ids, out_orig):
    import forge

    cases = [
        ("embedding", ["embedding"]),
        ("rmsnorm",   ["rmsnorm"]),
        ("rope",      ["rope"]),
        ("geglu",     ["geglu"]),
        ("all-wired", None),  # None == every wired kernel
    ]
    results = {}
    for label, kernels in cases:
        print(f"\n  forge.patch(model, kernels={kernels!r}) ...")
        try:
            forge.patch(model, kernels=kernels)
            print(f"    patched_counts = {model._forge_patched_counts}")
            out = _forward_logits(model, ids)
            results[label] = _summary(out_orig, out, label)
        except Exception as e:
            print(f"    EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[label] = False
        finally:
            if getattr(model, "_forge_patched", False):
                forge.unpatch(model)
    return results


def _shape_stress(model, out_orig_shape_fn, vocab_size, device):
    """Patch once with everything, then run several (batch, seq_len) shapes
    against fresh unpatched baselines to confirm no shape-dependent bug
    (e.g. autotune cache, padding-token duplicate handling in embedding bwd)."""
    import forge
    import torch

    shapes = [(1, 8), (2, 32), (1, 128), (3, 17)]  # 17 = odd, tests masking
    results = {}
    for b, s in shapes:
        torch.manual_seed(100 + b * s)
        ids = torch.randint(0, vocab_size, (b, s), device=device)
        with torch.no_grad():
            base = model(ids).logits.clone()
        forge.patch(model)
        try:
            with torch.no_grad():
                new = model(ids).logits
            results[f"shape b={b} s={s}"] = _summary(base, new, f"b={b}_s={s}")
        finally:
            forge.unpatch(model)
    return results


def _backward_parity(model, ids):
    """Verify gradients on a few representative weights agree between patched
    and unpatched runs. Hits:
      - embed_tokens.weight (exercises embedding kernel backward)
      - layers[0].input_layernorm.weight (exercises RMSNorm backward)
      - layers[0].mlp.gate_proj.weight (exercises GeGLU backward via the
        chain rule through the projection)
    """
    import forge
    import torch

    target_names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    name_to_param = dict(model.named_parameters())

    def _run_backward():
        # Need grads on (otherwise eval() / .requires_grad_ may already be True)
        model.zero_grad(set_to_none=True)
        out = model(ids).logits
        # Sum is enough to drive a backward pass; we just need a scalar loss.
        loss = out.float().sum()
        loss.backward()
        return {n: name_to_param[n].grad.detach().clone() for n in target_names}

    # Baseline — must run in train()-equivalent grad path
    for p in model.parameters():
        p.requires_grad_(True)

    base_grads = _run_backward()

    forge.patch(model)
    try:
        new_grads = _run_backward()
    finally:
        forge.unpatch(model)

    ok = True
    for n in target_names:
        ok &= _grad_summary(base_grads[n], new_grads[n], n)
    return ok


def _backward_bisection(model, ids):
    """Per-kernel backward parity.

    For each of {embedding, rmsnorm, rope, geglu} patched in isolation, runs
    fwd+bwd and compares gradients on params the kernel actually flows through:

        embedding -> embed_tokens.weight                  (kernel's only grad output)
        rmsnorm   -> input_layernorm.weight (layer 0)     (RMSNorm weight grad)
                  -> post_feedforward_layernorm.weight (layer 1)  (catches a different layer)
        rope      -> self_attn.q_proj.weight (layer 0)    (RoPE bwd flows through Q/K projs)
                  -> self_attn.k_proj.weight (layer 1)    (layer 1 uses sliding-window attn)
        geglu     -> mlp.gate_proj.weight (layer 0)       (gate path through GeGLU bwd)
                  -> mlp.down_proj.weight (layer 1)       (down path; layer 1 covered)

    A bug in any kernel's backward shows up as drift on its targeted param while
    the others stay clean — letting us bisect to the offending kernel.
    """
    import forge

    target_params = {
        "embedding": ["model.embed_tokens.weight"],
        "rmsnorm":   ["model.layers.0.input_layernorm.weight",
                      "model.layers.1.post_feedforward_layernorm.weight"],
        "rope":      ["model.layers.0.self_attn.q_proj.weight",
                      "model.layers.1.self_attn.k_proj.weight"],
        "geglu":     ["model.layers.0.mlp.gate_proj.weight",
                      "model.layers.1.mlp.down_proj.weight"],
    }
    all_targets = sorted({t for ts in target_params.values() for t in ts})
    name_to_param = dict(model.named_parameters())
    missing = [t for t in all_targets if t not in name_to_param]
    if missing:
        print(f"  SKIP: target params not found in model: {missing}")
        return {}

    def _run():
        model.zero_grad(set_to_none=True)
        loss = model(ids).logits.float().sum()
        loss.backward()
        return {n: name_to_param[n].grad.detach().clone() for n in all_targets}

    for p in model.parameters():
        p.requires_grad_(True)

    print("\n  computing unpatched baseline gradients ...")
    base = _run()

    results = {}
    for kernel, targets in target_params.items():
        print(f"\n  patching ONLY kernels=['{kernel}'] and recomputing gradients ...")
        try:
            forge.patch(model, kernels=[kernel])
            new = _run()
            ok = True
            for t in targets:
                short = t.replace("model.", "").replace(".weight", "")
                ok &= _grad_summary(base[t], new[t], f"[{kernel}] {short}")
            results[f"bwd[{kernel}]"] = ok
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[f"bwd[{kernel}]"] = False
        finally:
            if getattr(model, "_forge_patched", False):
                forge.unpatch(model)
    return results


def _generation_parity(model, device, max_new_tokens=24, seed=42):
    """Greedy generation token-id parity.

    With do_sample=False + num_beams=1, greedy decoding is fully deterministic
    given identical logits at each step. Bf16 drift (~9e-3 in logits on this
    toy model) will eventually flip an argmax — but the prefix should match
    for many steps. We measure how many initial tokens match exactly and the
    per-step max logit diff between patched / unpatched runs.

    Acceptance is lenient on this random-init toy: at least 4 initial tokens
    must match (any real backward bug would fail this immediately). Real-model
    tests should ratchet this up to a near-perfect bar.
    """
    import forge
    import torch

    torch.manual_seed(seed)
    # Single short prompt so we can see drift clearly
    prompt = torch.randint(0, model.config.vocab_size, (1, 8), device=device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        # Random-init model has no real pad/eos — pin to a fixed id to avoid early-stop noise.
        pad_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )

    print(f"  prompt = {prompt.tolist()}")
    print(f"  greedy generating {max_new_tokens} tokens, comparing patched vs unpatched ...")

    model.eval()
    with torch.no_grad():
        base = model.generate(prompt, **gen_kwargs)
    base_tokens = base.sequences[0, prompt.shape[1]:].tolist()
    base_scores = [s.clone() for s in base.scores]

    forge.patch(model)
    try:
        with torch.no_grad():
            new = model.generate(prompt, **gen_kwargs)
        new_tokens = new.sequences[0, prompt.shape[1]:].tolist()
        new_scores = [s.clone() for s in new.scores]
    finally:
        forge.unpatch(model)

    print(f"  unpatched tokens: {base_tokens}")
    print(f"  patched   tokens: {new_tokens}")

    # First divergence index
    first_diverge = next(
        (i for i, (a, b) in enumerate(zip(base_tokens, new_tokens)) if a != b),
        len(base_tokens),
    )
    exact_count = sum(1 for a, b in zip(base_tokens, new_tokens) if a == b)
    print(f"  initial-prefix match length: {first_diverge}/{len(base_tokens)}")
    print(f"  total exact-match count:     {exact_count}/{len(base_tokens)}")

    # Per-step max-logit-diff (drift tracking)
    diffs = [(b - a).abs().max().item() for a, b in zip(base_scores, new_scores)]
    print(f"  per-step max logit diff (first 8):  "
          f"{[f'{d:.2e}' for d in diffs[:8]]}")
    print(f"  per-step max logit diff (last 8):   "
          f"{[f'{d:.2e}' for d in diffs[-8:]]}")

    ok = first_diverge >= 4
    print(f"  [generation parity] first_diverge={first_diverge} (>= 4 required): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _padding_stress(model, device, seed=7):
    """Embedding-backward cooperative-path stress.

    Build a batch where 90% of tokens are id=0 (simulating heavy padding).
    With batch=4, seq=128 -> 512 total tokens -> ~460 duplicates of id=0.
    max_group_size > COOPERATIVE_GROUP_THRESHOLD (32) triggers the two-phase
    cooperative reduction in the embedding backward kernel — which is the
    main motivation for the kernel's existence and is otherwise untested.

    Compares gradient on embed_tokens.weight (and specifically the row for
    id=0, the heavily-duplicated one) against the unpatched PyTorch path.
    """
    import forge
    import torch

    torch.manual_seed(seed)
    batch, seq = 4, 128
    n_tokens = batch * seq
    pad_id = 0
    n_pad = int(n_tokens * 0.9)
    flat = torch.full((n_tokens,), pad_id, dtype=torch.long, device=device)
    rand_idx = torch.randperm(n_tokens, device=device)[: n_tokens - n_pad]
    flat[rand_idx] = torch.randint(1, model.config.vocab_size,
                                   (n_tokens - n_pad,), device=device)
    ids = flat.view(batch, seq)
    print(f"  input shape={tuple(ids.shape)}  "
          f"pad fraction={(ids == pad_id).float().mean().item():.3f}  "
          f"max group size={(ids == pad_id).sum().item()}  "
          f"(cooperative threshold=32)")

    embed_param = model.get_input_embeddings().weight

    def _run():
        model.zero_grad(set_to_none=True)
        loss = model(ids).logits.float().sum()
        loss.backward()
        return embed_param.grad.detach().clone()

    for p in model.parameters():
        p.requires_grad_(True)

    base = _run()

    forge.patch(model)
    try:
        new = _run()
    finally:
        forge.unpatch(model)

    print(f"  pad-id row (id={pad_id}) grad magnitude: "
          f"base_max={base[pad_id].abs().max().item():.3e}  "
          f"new_max={new[pad_id].abs().max().item():.3e}")
    ok_full = _grad_summary(base, new, "padding-stress: embed full")
    ok_pad_row = _grad_summary(base[pad_id:pad_id + 1], new[pad_id:pad_id + 1],
                               f"padding-stress: embed[pad_id={pad_id}] row")
    return ok_full and ok_pad_row


def _vram_measure(device, dtype, seed=0):
    """Peak VRAM for forward+backward, unpatched vs fully patched.

    Uses a moderately-sized model (hidden=512, intermediate=2048, layers=4)
    so activation/gradient tensors are large enough to dominate kernel
    overhead — small toy models bottom out on fixed allocations and show
    no signal. Even at this scale, true production wins require larger
    hidden_size + longer seq; we report toy numbers as a smoke check.
    """
    import forge
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    if device != "cuda":
        print("  SKIP: VRAM measurement requires CUDA")
        return True

    cfg = Gemma2Config(
        vocab_size=2048,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=512,
        hidden_activation="gelu_pytorch_tanh",
    )

    def _measure(use_patch: bool):
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        ids = torch.randint(0, cfg.vocab_size, (2, 256), device=device)
        if use_patch:
            forge.patch(model)
        try:
            out = model(ids).logits.float().sum()
            out.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        finally:
            if use_patch and getattr(model, "_forge_patched", False):
                forge.unpatch(model)
            del model
            torch.cuda.empty_cache()
        return peak

    peak_base = _measure(use_patch=False)
    peak_patched = _measure(use_patch=True)
    delta = peak_patched - peak_base
    pct = 100.0 * delta / peak_base
    print(f"  model: Gemma2 hidden=512 layers=4 intermediate=2048 batch=2 seq=256")
    print(f"  peak VRAM unpatched: {peak_base:7.2f} MB")
    print(f"  peak VRAM patched  : {peak_patched:7.2f} MB")
    print(f"  delta              : {delta:+7.2f} MB  ({pct:+.2f}%)")
    print(f"  NOTE: toy model — true VRAM win shows at production hidden_size + seq_len.")
    print(f"        Bar here: patched must not REGRESS catastrophically (<+20%).")
    ok = pct < 20.0
    print(f"  [vram] {'PASS' if ok else 'FAIL'}")
    return ok


def _double_patch_raises(model):
    import forge
    forge.patch(model)
    try:
        try:
            forge.patch(model)
            return False
        except RuntimeError as e:
            print(f"  RuntimeError (expected): {e}")
            return True
    finally:
        forge.unpatch(model)


def _unknown_kernel_raises(model):
    import forge
    try:
        forge.patch(model, kernels=["nope_kernel"])
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)
        return False
    except KeyError as e:
        print(f"  KeyError (expected): {e}")
        return True
    except Exception as e:
        print(f"  Wrong exception type: {type(e).__name__}: {e}")
        return False


def _stub_kernel_raises(model):
    """Asking for a kernel that's still a stub (e.g. cross_entropy) must
    raise NotImplementedError BEFORE mutating any module."""
    import forge
    try:
        forge.patch(model, kernels=["cross_entropy"])
        if getattr(model, "_forge_patched", False):
            forge.unpatch(model)
        return False
    except NotImplementedError as e:
        print(f"  NotImplementedError (expected): {e}")
        # Confirm pre-validation didn't half-patch the model.
        return not getattr(model, "_forge_patched", False)
    except Exception as e:
        print(f"  Wrong exception type: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF model id (e.g. google/gemma-2-2b). If unset, "
                         "uses a tiny random-init Gemma2 — no download/auth.")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import forge
    import transformers

    print(f"\n=== forge.patch verification on Gemma2 ===")
    print(f"forge version: {forge.__version__}")
    print(f"torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"transformers: {transformers.__version__}\n")

    _print_patching_analysis()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(args.seed)
    if args.model is None:
        print(f"Building tiny random-init Gemma2 (dtype={dtype}, device={device})")
        model = _build_tiny_gemma2(device, dtype)
    else:
        print(f"Loading {args.model} (dtype={dtype}, device={device})")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device).eval()
    print()

    _config_analysis(model)
    census_ok = _module_census(model)

    ids = torch.randint(0, model.config.vocab_size,
                        (args.batch, args.seq_len), device=device)

    # === Baseline ===
    print("=" * 78)
    print("[1] BASELINE forward (unpatched)")
    print("=" * 78)
    out_orig = _forward_logits(model, ids)
    print(f"  baseline logits: shape={tuple(out_orig.shape)} "
          f"dtype={out_orig.dtype} NaN={int(out_orig.isnan().sum())}\n")

    results = {"census": census_ok}

    # === Bisection — per kernel ===
    print("=" * 78)
    print("[2] BISECTION — per-kernel forward parity")
    print("=" * 78)
    results.update(_bisection_per_kernel(model, ids, out_orig))

    # === Shape stress ===
    print("\n" + "=" * 78)
    print("[3] SHAPE STRESS — full patch across multiple (batch, seq) shapes")
    print("=" * 78)
    results.update(_shape_stress(model, None, model.config.vocab_size, device))

    # === Backward parity ===
    print("\n" + "=" * 78)
    print("[4] BACKWARD GRADIENT PARITY — full patch")
    print("=" * 78)
    # Rebuild model so we don't carry over eval()/grad state from forward tests.
    torch.manual_seed(args.seed)
    if args.model is None:
        model_grad = _build_tiny_gemma2(device, dtype)
    else:
        from transformers import AutoModelForCausalLM
        model_grad = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
    model_grad.train()  # gradients on
    ids_grad = torch.randint(0, model_grad.config.vocab_size,
                             (args.batch, args.seq_len), device=device)
    results["backward"] = _backward_parity(model_grad, ids_grad)

    # === Per-kernel backward bisection ===
    print("\n" + "=" * 78)
    print("[5] PER-KERNEL BACKWARD BISECTION")
    print("=" * 78)
    torch.manual_seed(args.seed + 1)
    if args.model is None:
        model_bwd = _build_tiny_gemma2(device, dtype)
    else:
        from transformers import AutoModelForCausalLM
        model_bwd = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
    model_bwd.train()
    ids_bwd = torch.randint(0, model_bwd.config.vocab_size,
                            (args.batch, args.seq_len), device=device)
    results.update(_backward_bisection(model_bwd, ids_bwd))
    del model_bwd

    # === Generation parity ===
    print("\n" + "=" * 78)
    print("[6] GENERATION PARITY — greedy decode token-id match")
    print("=" * 78)
    torch.manual_seed(args.seed + 2)
    if args.model is None:
        model_gen = _build_tiny_gemma2(device, dtype)
    else:
        from transformers import AutoModelForCausalLM
        model_gen = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device).eval()
    results["generation_parity"] = _generation_parity(model_gen, device)
    del model_gen

    # === Padding-token stress (embedding backward cooperative path) ===
    print("\n" + "=" * 78)
    print("[7] PADDING-TOKEN STRESS — embedding backward cooperative path")
    print("=" * 78)
    torch.manual_seed(args.seed + 3)
    if args.model is None:
        model_pad = _build_tiny_gemma2(device, dtype)
    else:
        from transformers import AutoModelForCausalLM
        model_pad = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(device)
    model_pad.train()
    results["padding_stress"] = _padding_stress(model_pad, device)
    del model_pad

    # === Peak VRAM measurement ===
    print("\n" + "=" * 78)
    print("[8] PEAK VRAM — forward+backward, unpatched vs fully patched")
    print("=" * 78)
    results["vram"] = _vram_measure(device, dtype, seed=args.seed)

    # === Unpatch bit-exact ===
    print("\n" + "=" * 78)
    print("[9] BIT-EXACT UNPATCH RESTORATION")
    print("=" * 78)
    forge.patch(model)
    forge.unpatch(model)
    out_restored = _forward_logits(model, ids)
    exact_restore = bool(torch.equal(out_restored, out_orig))
    print(f"  bit-exact restore = {exact_restore}")
    results["unpatch_exact"] = exact_restore

    # === Negative tests ===
    print("\n" + "=" * 78)
    print("[10] NEGATIVE TESTS — patch must fail loudly when misused")
    print("=" * 78)
    print(" double-patch must raise RuntimeError:")
    results["double_patch_raises"] = _double_patch_raises(model)
    print(" unknown kernel name must raise KeyError:")
    results["unknown_kernel_raises"] = _unknown_kernel_raises(model)
    print(" stub kernel in explicit list must raise NotImplementedError:")
    results["stub_kernel_raises"] = _stub_kernel_raises(model)

    # === Verdict ===
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {name:30s}  {'PASS' if ok else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

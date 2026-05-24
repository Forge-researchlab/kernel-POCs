# Phase 3 — FSDP2 + LoRA + Gemma 2 (Resume Doc)

**Audience:** the next Claude session running on a 2-GPU machine.
**Status when this doc was written:** Phases 1 + 2 complete on a 1-GPU box; FSDP2 work intentionally deferred until 2 GPUs were available.
**Date written:** 2026-05-24 (Forge hackathon weekend, Day 2 afternoon)
**Owner:** Xhitij (Meesho, Forge GPU kernel team)

> **How to start when you read this:** the user will say something like *"resume Phase 3"* or *"go ahead with FSDP2"*. Read this whole doc first, then jump to §9 (the action plan). You do not need to re-derive any of the design choices — they are locked.

---

## 1. The one-line summary

Build `forge/tests/verify_fsdp2_lora_gemma.py` and run it under `torchrun --nproc-per-node=2`. It must show that **PEFT-wrapped Gemma 2 + `forge.patch(kernels=["lora_qkv","lora_mlp"])` works correctly when each `Gemma2DecoderLayer` is wrapped with FSDP2's `fully_shard()`**, in both an inference path (forward + greedy generate) and a realistic training step (Adam, bf16 mixed-precision, gradient accumulation, 5 outer steps). Compare against a single-GPU reference baked by rank 0.

The H15 doc — `/workspace/kernel-POCs/context/forge_hackathon_site/HACKATHON_SMOKE_TEST.md` — defines the *minimum* smoke test (RMSNorm + Linear only). Xhitij has explicitly chosen to skip the minimum and run the full Gemma + LoRA scenario instead. Don't re-debate scope.

---

## 2. What's already done (do NOT re-do this work)

### Phase 1 — LoRA kernels parity (single-GPU)

**Wired into `forge.kernels`:**
- `forge/forge/kernels/lora_mlp.py` — additively exports v6 (`LoRAMLPv6`, `lora_mlp_v6`, `LoRAMLPv6Module`, `stack_lora_a`) on top of the existing v3 exports.
- `forge/forge/kernels/lora_qkv.py` — additively exports v4 (`LoRAQKVv4Function`, `lora_qkv_v4`, `pack_weights_backward`, `pack_lora_a`) on top of v3.

**Tests, all GREEN:**
- `forge/tests/test_lora_qkv.py` — 12/12 PASS (forward + 7 gradient cases)
- `forge/tests/test_lora_mlp.py` — 8/8 PASS
- `forge/tests/test_lora_convergence.py` — 2/2 PASS (50-step Adam parity, **subprocess-isolated** to dodge a sys.path collision between the two kernel directories — see §11)

### Phase 2 — PEFT + Gemma 2 patching integration

**Edits:**
- `forge/forge/patching/kernels/lora.py` — two factory upgrades:
  - `make_lora_mlp_forward`: detects activation via a new `_detect_mlp_activation(module)` helper.
    - SiLU (Qwen 2/3 default) → existing `LoRAMLPv3` fused path
    - GeGLU (`gelu_pytorch_tanh` / `gelu` — Gemma 2) → thin layered path: calls `module.gate_proj` / `up_proj` / `down_proj` (PEFT-wrapped, so LoRA-A/B already baked in) and fuses **only** the activation via `forge.kernels.geglu`. This is intentional — there is no fused LoRA-MLP-GeGLU kernel yet, and writing one is out of scope for Day 2.
    - Unrecognized activation → `raise ForgeSkipPatch(...)` so the patch loop falls through.
  - `make_lora_qkv_forward`: three additive Gemma 2 fixes (Qwen path unchanged):
    - Accepts both `past_key_value` (singular, older Qwen) and `past_key_values` (plural, Gemma 2). Uses whichever is non-None.
    - Reads `module.attn_logit_softcapping` (Gemma 2 specific) and forwards as `softcap=` to the attention interface when present.
    - Sliding-window detection was already Gemma-compatible because `getattr(module, "sliding_window", None)` is tried first; the Qwen-specific `config.use_sliding_window` + `max_window_layers` fallback only fires when that returns None.

- `forge/forge/patching/gemma.py` — extended `GEMMA_MAPPING`:
  - `Gemma2MLP`, `GemmaMLP` → `[("lora_mlp", {}), ("geglu", {"activation": "gelu"})]` (list — LoRA tried first, GeGLU fallback). The patch loop in `core.py` already supports list-of-specs and falls through on `ForgeSkipPatch`.
  - `Gemma2Attention` → `("lora_qkv", {})` (single spec — no non-LoRA fused QKV baseline exists).

- `forge/tests/verify_patch_gemma.py` — one-line maintenance fix to `_print_patching_analysis` so it handles both tuple and list-of-tuples mapping values. **No test assertions changed.**

**New file:**
- `forge/tests/verify_patch_lora_gemma.py` — 10-section integration test, **8/8 PASS**:
  - [1] PEFT availability  [2] Patching analysis  [3] Module census
  - [4] Baseline forward  [5] Per-kernel bisection (lora_qkv / lora_mlp / both) — **forward bit-exact, max_diff = 0**
  - [6] Backward gradient parity (q_proj + gate_proj LoRA-A/B) — note: PEFT inits `lora_B=0` so `dA` grads are trivially zero on BOTH sides; the test treats this as "trivial agree" rather than failing on cosine-of-zero
  - [7] `ForgeSkipPatch` on `disable_adapter_layers()` — `patched_counts={}` as expected
  - [8] Bit-exact unpatch  [9] Negative tests (double-patch, unknown kernel)
  - [10] Mini-convergence — 20 SGD steps, rel loss diff max = 6e-5, final = 9e-6

### Repo state at handoff

```
git status (relevant):
  modified:  kernels/lora_mlp/experiments/v6/lora_mlp_kernel_v6.py
  modified:  forge/forge/kernels/lora_mlp.py
  modified:  forge/forge/kernels/lora_qkv.py
  modified:  forge/forge/patching/kernels/lora.py
  modified:  forge/forge/patching/gemma.py
  modified:  forge/tests/verify_patch_gemma.py
  new:       forge/tests/test_lora_qkv.py
  new:       forge/tests/test_lora_mlp.py
  new:       forge/tests/test_lora_convergence.py
  new:       forge/tests/verify_patch_lora_gemma.py
  ?? context/forge_hackathon_site/  (HTML site — read-only reference)
```

**Nothing has been committed yet.** Don't `git commit` unless Xhitij asks.

---

## 3. The architectural bet you're validating

The single biggest unknown in the V1 plan, from `HACKATHON_SMOKE_TEST.md`:

> Does our `torch.autograd.Function` pattern — specifically `ctx.save_for_backward(weight, ...)` — work correctly when FSDP2 has sharded the weight across GPUs?

If yes → every kernel that follows the same pattern is FSDP2-compatible, and Phase 3 of the V1 plan is unblocked.

If no → we need to redesign the pattern (likely: stop calling `save_for_backward` on weights; re-fetch from the module in backward so FSDP2's all-gather hook fires).

For Gemma 2 + LoRA specifically, the pattern question splits into two parts:
- **`LoRAQKVv4Function` / `lora_qkv_v3`** — does the v3/v4 backward survive when `W_q`, `W_k`, `W_v` are sharded params? These weights are *extracted at closure-build time* by `_lora_tensors()` and passed as raw tensors into the kernel. **This is the riskiest path.** If FSDP2 swaps the weight ref after `fully_shard()` is called (it usually does), the closure holds a stale pointer.
- **GeGLU MLP path** — much safer. It calls `module.gate_proj(x)` as a submodule, so PyTorch + FSDP2 handle the weight all-gather. Only the activation goes through Triton, and the activation has no parameters.

So we expect the GeGLU LoRA-MLP path to "just work" and the LoRA-QKV path to be the one that surfaces issues. If C2 or C3 fails, that's where to look first.

---

## 4. Hardware + environment expectations on the 2-GPU box

Run these at the start of the session and paste the output back. The next Claude should verify before proceeding:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"
python -c "from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy; print('FSDP2 API OK')"
python -c "import peft; print('peft', peft.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
torchrun --version 2>&1 || python -c "import torch.distributed.run; print('torchrun via python -m torch.distributed.run is fine')"
```

**Hard requirements:**
- `torch >= 2.4` (FSDP2 stable; `fully_shard` is the new API, NOT the old `FSDP` wrapper)
- `device_count >= 2`
- `peft >= 0.19` (the version Phase 2 was developed against — 0.19.1)
- `transformers` with `Gemma2ForCausalLM` working (4.57+ is fine; Phase 2 used 4.57.6)
- NCCL backend available (it is on any A100/H100 install)

If any of these fail, stop and report. Don't try to install torch from scratch.

---

## 5. The test plan in detail

### 5.1 File to create

`forge/tests/verify_fsdp2_lora_gemma.py`

Designed to be invoked as:
```bash
cd /workspace/kernel-POCs
torchrun --nproc-per-node=2 --standalone forge/tests/verify_fsdp2_lora_gemma.py
```

Exit 0 if all checks pass, exit 1 otherwise. Each rank logs with a `[rank N]` prefix; rank 0 is the authoritative reporter and the verdict printer.

### 5.2 Model setup (must be identical to Phase 2)

Use the same builder as `verify_patch_lora_gemma.py`:

```python
from transformers import Gemma2Config, Gemma2ForCausalLM
from peft import LoraConfig, get_peft_model

cfg = Gemma2Config(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,        # bump to 4 so FSDP2 has multiple decoder layers to shard
    num_attention_heads=4,
    num_key_value_heads=2,      # GQA
    head_dim=32,
    max_position_embeddings=256,
    hidden_activation="gelu_pytorch_tanh",
)
model = Gemma2ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.0,
)
peft_model = get_peft_model(model, peft_cfg)
```

The seed must match across ranks. **Seed BEFORE construction**, both `torch.manual_seed` and `torch.cuda.manual_seed_all`.

### 5.3 The two execution paths

**Path A — single-GPU reference (rank 0 only, runs FIRST)**

Rank 0:
1. Build the PEFT-Gemma model on cuda:0.
2. Save `state_dict()` to a shared file (use a tmpdir like `/tmp/forge_fsdp_init_{pid}.pt` — set `pid` from rank 0's PID and broadcast it).
3. Run one forward + one greedy generate(max_new_tokens=24); save logits and generated token IDs.
4. Run a 5-step Adam training loop on a fixed seeded batch; save the loss curve.
5. Save reference grads after step 1 (LoRA-A and LoRA-B grads on `layers.0.self_attn.q_proj` and `layers.0.mlp.gate_proj`).
6. Write reference to `/tmp/forge_fsdp_ref_{pid}.pt`.

All other ranks: `dist.barrier()` until rank 0 signals done (a second barrier).

**Path B — FSDP2 + forge.patch (all ranks)**

After the barrier:
1. Every rank builds an identical `peft_model`, then `load_state_dict()` from the file rank 0 saved.
2. Wrap each `Gemma2DecoderLayer` with `fully_shard(layer, mp_policy=MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32))`. Also `fully_shard(peft_model.base_model.model)` for the root.
3. Call `forge.patch(peft_model, kernels=["lora_qkv","lora_mlp"])`.
4. Run forward + generate; compare to reference on rank 0.
5. Run the same 5-step Adam loop; compare loss curve.
6. All-gather LoRA-A and LoRA-B grads (they're sharded) on rank 0 and compare to reference grads.
7. Report `torch.cuda.max_memory_allocated()` per rank.

### 5.4 The six checks

| ID | What | How | Pass criterion |
|---|---|---|---|
| **C1** | Doesn't crash | `torchrun` exits 0 | No NCCL errors, no hang, no CUDA OOM |
| **C2** | Forward matches | inference logits FSDP2 vs single-GPU ref on rank 0 | `max_diff < 5e-2`, `cos > 0.999` (bf16 noise floor) |
| **C3** | Gradients match | LoRA-A & LoRA-B grads (all-gathered) on rank 0 | `rel < 5e-2`, `cos > 0.9999`. For LoRA-A on a freshly-initialized model: `dA` is trivially zero (PEFT inits B=0), so the trivial-zero branch applies — both sides must agree on `||grad|| < 1e-6` |
| **C4** | Memory is sharded | `cuda.max_memory_allocated()` per rank | Per-rank peak ≤ 60% of single-GPU peak. **Tiny model caveat:** with hidden=128 the absolute numbers are small; the *ratio* is what matters |
| **C5** | Generation parity | greedy token IDs comparison | exact match for ≥ 22/24 tokens (allow ±2 token drift in case of bf16 tiebreak on logit argmax — exact match preferred) |
| **C6** | Training converges with parity | 5-step loss curve | both runs converge; per-step `rel_diff < 2%`; final losses within `5%` |

### 5.5 Distributed primitives you'll need

```python
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
device = f"cuda:{rank}"

# All-gather a sharded grad onto rank 0 for comparison:
def all_gather_grad(param):
    """Return the full (unsharded) grad of an FSDP2-sharded param on rank 0; None elsewhere."""
    # In FSDP2, param.grad is a DTensor on the local shard. Use .full_tensor() to
    # gather it. full_tensor() returns the full tensor on every rank — that's fine
    # for the comparison; rank 0 just does the assert.
    g = param.grad
    if hasattr(g, "full_tensor"):
        return g.full_tensor()
    return g  # not sharded (shouldn't happen for LoRA params under FSDP2)
```

### 5.6 Mixed precision policy

Match what we'd ship in production:

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,     # reduce-scatter in fp32 to avoid drift
    output_dtype=torch.bfloat16,
)
```

If C2 or C3 fails, **first thing to try** is rerunning with `mp_policy=None` (no mixed precision). If the failure goes away, the issue is mixed-precision casting, not the autograd pattern. The H15 doc names this as Outcome C cause #2.

### 5.7 Gradient accumulation

```python
NUM_MICRO_BATCHES = 4
for step in range(5):
    optimizer.zero_grad(set_to_none=True)
    for mb_idx in range(NUM_MICRO_BATCHES):
        x = micro_batches[mb_idx]
        loss = peft_model(x, labels=x).loss / NUM_MICRO_BATCHES
        loss.backward()
    optimizer.step()
```

FSDP2 handles the reduce-scatter on `.backward()` automatically. **Do NOT** call `model.no_sync()` between micro-batches — the goal is to test the full pipeline including grad-accum + reduce-scatter interaction.

### 5.8 Adam configuration

```python
optimizer = torch.optim.AdamW(
    [p for p in peft_model.parameters() if p.requires_grad],
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

Both runs use the same optimizer; same seed; identical micro-batches; the same loss reduction.

---

## 6. Anticipated failure modes (from the H15 doc, mapped to LoRA+Gemma)

Read these BEFORE running. When something breaks, you'll save time recognizing the symptom.

### Outcome B.1 — crash on `fully_shard()` itself
**Likely cause:** `peft_model.base_model.model.model.layers[i]` doesn't have parameters in the expected place. PEFT inserts `Linear` wrappers (`peft.tuners.lora.layer.Linear`) around `q_proj`, etc.; FSDP2 should still find the underlying weights, but the wrapper layout can confuse `fully_shard`'s policy.
**Fix:** wrap the `Gemma2DecoderLayer` (per-layer), not the whole model. The decoder layer contains the PEFT-wrapped projections as sub-submodules; FSDP2 walks down naturally.

### Outcome B.2 — crash during forward
**Likely cause:** `make_lora_qkv_forward` captures `W_q = module.q_proj.base_layer.weight` (or however `_lora_tensors` extracts it) at closure-build time. FSDP2 may have hooks that depend on going through `module.q_proj.forward()` to trigger the all-gather. Bypassing the submodule call with a raw tensor pointer breaks this.
**Fix path:** in `forge/forge/patching/kernels/lora.py`, the LoRA-QKV factory needs to either (a) re-fetch `module.q_proj.base_layer.weight` inside the inner `forward()` closure (so the all-gather hook fires each forward), or (b) call `module.q_proj(x)` as a submodule and pull the LoRA-A/B contribution from `module.q_proj.lora_A["default"](x)` separately. **Option (b) is the cleanest** — mirrors what we already do in `make_lora_mlp_forward` for Gemma. The trade-off is losing the v4-fused projection (you get 3 independent matmuls instead of 1 packed one), but correctness > speed for the smoke test.

### Outcome C — forward output mismatches
**Triage:**
1. Print `W_q.shape` inside the factory's `forward` closure. If it's `[hidden/2, hidden]` instead of `[hidden, hidden]`, FSDP2 has sharded the param and we're seeing the local shard. That's the smoking gun for "save_for_backward broke."
2. Try `mp_policy=None` to rule out precision casting.
3. Check that `x` is identical on both ranks: `dist.all_reduce(x.float().sum())` should equal `world_size * single_x.sum()`.

### Outcome D — gradients mismatch (forward OK)
**Likely cause:** `LoRAQKVv4Function.backward` is reading `ctx.saved_tensors` and seeing a freed/dead reference because FSDP2 freed the all-gathered weight post-forward.
**Fix:** route through PEFT submodules in the factory (Outcome B.2 fix path above). This sidesteps `save_for_backward` on FSDP-managed weights entirely.

### Outcome E — correctness OK but memory not sharded
**Likely cause:** the closure pins the all-gathered weight as a closed-over Python reference.
**Fix:** same as D — don't capture weights at closure-build time. Re-fetch each forward.

### Outcome F — flaky
**First try:** add `torch.cuda.synchronize(); dist.barrier()` around the comparison. Triton kernels are async by default; FSDP2 collectives are async by default; the comparison may be racing against in-flight work.

### A LoRA-specific failure that's not in the H15 doc
**`scaling` is captured by value, not by ref.** `_lora_tensors` returns `s_q = module.q_proj.scaling["default"]` — this is a Python float. That's fine, it doesn't go through FSDP2. But if the closure ever stored a *tensor* instead of a float, FSDP2 would not know how to handle it. Keep `scaling` as a Python scalar.

---

## 7. The fix-it path if C2 or C3 fails

If forward or gradients mismatch (most likely culprit: the QKV factory), the smallest-blast-radius fix is to **rewrite `make_lora_qkv_forward` to use PEFT submodule calls instead of raw tensor extraction**. Sketch:

```python
def make_lora_qkv_forward(module, config):
    # NEW: do not extract raw tensors at closure-build time.
    # Just verify the PEFT wrapper structure is present (raise ForgeSkipPatch otherwise).
    if not hasattr(module.q_proj, "lora_A"):
        raise ForgeSkipPatch("non-PEFT q_proj; no fused LoRA path")

    modeling = importlib.import_module(module.__class__.__module__)

    def forward(hidden_states, position_embeddings, attention_mask,
                past_key_value=None, past_key_values=None,
                cache_position=None, **kwargs):
        # PEFT submodule call — FSDP2 sees this and all-gathers the weight.
        # The LoRA-A/B contribution is computed inside module.q_proj.forward().
        query_states = module.q_proj(hidden_states)
        key_states   = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)
        # ... rest of the function unchanged (view + transpose + RoPE + attention) ...
```

This loses the v4-fused-QKV win on FSDP2, but is correct. **Only do this if C2/C3 actually fail.** Don't pre-emptively refactor.

Document the failure + fix in the verdict report (Section 9 of this plan).

---

## 8. What to write back to Xhitij in the verdict

Following the H15 doc Section 8 template:

> **Verdict:** Pass / Partial / Fail / Inconclusive
>
> **Numbers:**
> - `max_diff(forward) = ?`
> - `max_diff(input_grad) = ?`
> - `max_diff(weight_grad) = ?` (LoRA-A and LoRA-B separately)
> - `peak_mem_per_rank = ? GB` vs single-GPU `? GB`
> - `generation_token_match = ?/24`
> - `training_loss_rel_diff_max = ?`
>
> **One sentence on the implication for V1:** e.g. "FSDP2 wrap survives the LoRA-fused autograd pattern on Gemma 2 with X tolerance; can proceed to Phase 3 multi-GPU benchmarks." or "QKV factory needs the submodule rewrite (Section 7) — pattern doesn't survive raw-tensor closures."

---

## 9. Action plan — what to do when Xhitij says "go"

1. **Verify environment (§4).** Paste the output. If anything's wrong, stop.
2. **Run the existing Phase 1 + Phase 2 tests** to confirm the repo state on this machine matches what was tested on the 1-GPU box:
   ```bash
   cd /workspace/kernel-POCs
   python forge/tests/test_lora_qkv.py | tail -3        # expect 12/12 PASS
   python forge/tests/test_lora_mlp.py | tail -3        # expect 8/8 PASS
   python forge/tests/test_lora_convergence.py | tail -3 # expect 2/2 PASS
   python forge/tests/verify_patch_lora_gemma.py | tail -3 # expect 8/8 PASS
   ```
   If any regresses, stop and report — the codebase on this box is not in sync.
3. **Write `forge/tests/verify_fsdp2_lora_gemma.py`** per the template in §10.
4. **Smoke-run it single-rank** first: `torchrun --nproc-per-node=1 --standalone forge/tests/verify_fsdp2_lora_gemma.py`. FSDP2 supports world_size=1 (sharding is a no-op); this catches import / wiring bugs before you waste a real 2-GPU run.
5. **Run for real:** `torchrun --nproc-per-node=2 --standalone forge/tests/verify_fsdp2_lora_gemma.py`.
6. **Triage by Outcome (§6).** If green, write the verdict (§8). If red, apply the fix (§7) and re-run. **Time-box: 3 hours total.** If still red at 3 hours, write up what you learned and stop.
7. **Do NOT git commit** unless Xhitij asks.

---

## 10. Test file template

This is the skeleton. Fill in the bodies of the helper functions; the structure is locked.

```python
"""FSDP2 + LoRA + Gemma 2 multi-GPU smoke test.

Runs the full PEFT-wrapped Gemma 2 stack under FSDP2 sharding, with forge.patch
applied to lora_qkv and lora_mlp. Verifies forward parity, gradient parity,
memory sharding, greedy-generate parity, and training-loop parity vs a
single-GPU reference baked by rank 0.

Run:
    torchrun --nproc-per-node=2 --standalone forge/tests/verify_fsdp2_lora_gemma.py
"""
from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

import forge

SEED = 0
HIDDEN = 128
INTERMEDIATE = 256
NUM_LAYERS = 4
VOCAB = 1024
BATCH = 2
SEQ = 32
LR = 1e-3
TRAIN_STEPS = 5
MICRO_BATCHES = 4
GEN_TOKENS = 24


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _build_peft_model(device, dtype, seed=SEED):
    """Identical to the Phase 2 builder. Same seed -> same init across ranks."""
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = Gemma2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256,
        hidden_activation="gelu_pytorch_tanh",
    )
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)
    peft_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.0,
    )
    return get_peft_model(model, peft_cfg)


def _inner_model(peft_model):
    return peft_model.base_model.model.model


def _make_inputs(device, seed):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    return ids


def _cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _all_gather_grad(param):
    """FSDP2 stores grads as DTensors; .full_tensor() gathers."""
    g = param.grad
    if g is None:
        return None
    if hasattr(g, "full_tensor"):
        return g.full_tensor().detach()
    return g.detach()


# ---------------------------------------------------------------
# Reference baking (rank 0 only)
# ---------------------------------------------------------------

def bake_reference(device, dtype, init_path, ref_path):
    """Single-GPU run, no FSDP2, no forge.patch. Save everything needed for comparison."""
    peft_model = _build_peft_model(device, dtype, seed=SEED)
    torch.save(peft_model.state_dict(), init_path)

    # === inference ref ===
    peft_model.eval()
    ids = _make_inputs(device, SEED + 1)
    with torch.no_grad():
        ref_logits = peft_model(ids).logits.clone().cpu()
        ref_gen = peft_model.generate(
            ids[:1], max_new_tokens=GEN_TOKENS, do_sample=False,
            pad_token_id=0,
        ).cpu()

    # === training ref ===
    peft_model.train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.0)
    torch.manual_seed(SEED + 2)
    micro_batches = [torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
                     for _ in range(MICRO_BATCHES)]
    ref_losses = []
    ref_grads_after_step1 = None
    for step in range(TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        for mb in micro_batches:
            out = peft_model(mb, labels=mb)
            (out.loss / MICRO_BATCHES).backward()
        if step == 0:
            ref_grads_after_step1 = {
                "q_lora_A": peft_model.get_parameter(
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
                ).grad.detach().cpu().clone(),
                "q_lora_B": peft_model.get_parameter(
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
                ).grad.detach().cpu().clone(),
                "gate_lora_A": peft_model.get_parameter(
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight"
                ).grad.detach().cpu().clone(),
                "gate_lora_B": peft_model.get_parameter(
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight"
                ).grad.detach().cpu().clone(),
            }
        opt.step()
        ref_losses.append(float(out.loss.detach()))

    torch.save({
        "ref_logits": ref_logits,
        "ref_gen": ref_gen,
        "ref_losses": ref_losses,
        "ref_grads": ref_grads_after_step1,
    }, ref_path)
    print(f"[rank 0] reference baked: logits {tuple(ref_logits.shape)}, "
          f"gen {tuple(ref_gen.shape)}, losses {[f'{l:.4f}' for l in ref_losses]}")
    del peft_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------
# FSDP2 run (all ranks)
# ---------------------------------------------------------------

def run_fsdp2(rank, world, device, dtype, init_path, ref_path):
    peft_model = _build_peft_model(device, dtype, seed=SEED)
    peft_model.load_state_dict(torch.load(init_path, map_location="cpu"))
    peft_model.to(device=device, dtype=dtype)

    # Per-decoder-layer wrap + root wrap
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for layer in _inner_model(peft_model).layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(_inner_model(peft_model), mp_policy=mp_policy)

    # Apply forge.patch AFTER FSDP2 wrap — that's the order Phase 3 will use.
    forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
    if rank == 0:
        print(f"[rank 0] patched_counts = {peft_model._forge_patched_counts}")

    # === inference path ===
    peft_model.eval()
    ids = _make_inputs(device, SEED + 1)
    with torch.no_grad():
        logits = peft_model(ids).logits
        gen = peft_model.generate(
            ids[:1], max_new_tokens=GEN_TOKENS, do_sample=False,
            pad_token_id=0,
        )

    # === training path ===
    peft_model.train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.0)
    torch.manual_seed(SEED + 2)
    micro_batches = [torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
                     for _ in range(MICRO_BATCHES)]
    losses = []
    grads_after_step1 = None
    for step in range(TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        for mb in micro_batches:
            out = peft_model(mb, labels=mb)
            (out.loss / MICRO_BATCHES).backward()
        if step == 0:
            grads_after_step1 = {
                "q_lora_A":   _all_gather_grad(peft_model.get_parameter(
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")),
                "q_lora_B":   _all_gather_grad(peft_model.get_parameter(
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight")),
                "gate_lora_A":_all_gather_grad(peft_model.get_parameter(
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight")),
                "gate_lora_B":_all_gather_grad(peft_model.get_parameter(
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight")),
            }
        opt.step()
        losses.append(float(out.loss.detach()))

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # === rank-0 comparison ===
    if rank == 0:
        ref = torch.load(ref_path, map_location="cpu")
        return _evaluate(logits.cpu(), gen.cpu(), losses, grads_after_step1,
                         ref, peak_mem, rank, world)
    return None


def _evaluate(logits, gen, losses, grads, ref, peak_mem, rank, world):
    """All checks on rank 0."""
    results = {}

    # C2 - forward parity
    diff = (logits.float() - ref["ref_logits"].float()).abs().max().item()
    cos = _cos(logits, ref["ref_logits"])
    results["C2_forward"] = (diff < 5e-2 and cos > 0.999)
    print(f"  C2 forward: max_diff={diff:.3e} cos={cos:.6f}  "
          f"{'PASS' if results['C2_forward'] else 'FAIL'}")

    # C3 - gradient parity (with trivial-zero handling for lora_A)
    for name in ("q_lora_A", "q_lora_B", "gate_lora_A", "gate_lora_B"):
        a = grads[name]; b = ref["ref_grads"][name]
        nm = max(a.abs().max().item(), b.abs().max().item())
        d = (a.float() - b.float()).abs().max().item()
        if nm < 1e-6:
            ok = (d < 1e-6)
            print(f"  C3 grad[{name}]: trivial-zero  d={d:.2e}  "
                  f"{'PASS' if ok else 'FAIL'}")
        else:
            rel = d / (nm + 1e-12); c = _cos(a, b)
            ok = (rel < 5e-2 and c > 0.9999)
            print(f"  C3 grad[{name}]: rel={rel:.2e} cos={c:.6f}  "
                  f"{'PASS' if ok else 'FAIL'}")
        results[f"C3_{name}"] = ok

    # C5 - generation
    n_match = int((gen[0] == ref["ref_gen"][0]).sum())
    results["C5_generation"] = (n_match >= 22)
    print(f"  C5 generation: matched {n_match}/{gen.shape[1]} tokens  "
          f"{'PASS' if results['C5_generation'] else 'FAIL'}")

    # C6 - training convergence
    rel_diffs = [abs(a - b) / (abs(b) + 1e-12)
                 for a, b in zip(losses, ref["ref_losses"])]
    converging_ref = ref["ref_losses"][-1] < ref["ref_losses"][0]
    converging_new = losses[-1] < losses[0]
    results["C6_training"] = (
        converging_ref and converging_new
        and max(rel_diffs) < 2e-2
        and abs(losses[-1] - ref["ref_losses"][-1]) / abs(ref["ref_losses"][-1]) < 5e-2
    )
    print(f"  C6 training: ref={ref['ref_losses']} fsdp={losses} "
          f"max_rel={max(rel_diffs):.2e}  "
          f"{'PASS' if results['C6_training'] else 'FAIL'}")

    # C4 - memory (informational only on this single rank's view; the multi-rank
    # comparison requires a per-rank gather which we do below)
    print(f"  C4 peak mem rank0: {peak_mem:.3f} GB (need to compare across ranks)")
    results["__peak_mem_rank0__"] = peak_mem

    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    if rank == 0:
        print(f"=== FSDP2 + LoRA + Gemma 2 smoke test ===")
        print(f"world_size={world} torch={torch.__version__} "
              f"device={torch.cuda.get_device_name(0)}")

    # tmpdir + paths shared via rank-0 broadcast
    tmpdir = tempfile.mkdtemp(prefix="forge_fsdp_")
    tmpdir_list = [tmpdir]
    dist.broadcast_object_list(tmpdir_list, src=0)
    tmpdir = tmpdir_list[0]
    init_path = os.path.join(tmpdir, "init_state.pt")
    ref_path = os.path.join(tmpdir, "reference.pt")

    # Rank 0 bakes the reference
    if rank == 0:
        print("\n[rank 0] Baking single-GPU reference ...")
        bake_reference(device, dtype, init_path, ref_path)
    dist.barrier()

    # All ranks run FSDP2
    if rank == 0:
        print("\n=== FSDP2 run (all ranks) ===")
    results = run_fsdp2(rank, world, device, dtype, init_path, ref_path)

    # C4 — gather per-rank peak mem
    peak = torch.cuda.max_memory_allocated() / 1e9
    peaks = [None] * world
    dist.all_gather_object(peaks, peak)

    # Verdict on rank 0
    if rank == 0:
        ref_peak = results["__peak_mem_rank0__"]  # NOTE: this is FSDP rank 0 peak,
                                                  # which already includes the reference run's
                                                  # released memory; cuda.max_memory_allocated()
                                                  # is a high-water mark over the process lifetime.
                                                  # For a clean C4 we'd want a reset_peak_memory_stats()
                                                  # between bake_reference and run_fsdp2 — see TODO below.
        max_peak = max(peaks)
        # C4 sanity: with hidden=128 the model is tiny, so absolute numbers aren't
        # informative. We're mostly checking the ratio. With world=2 and a tiny model,
        # the overhead of NCCL buffers may dominate — print and let Xhitij judge.
        print(f"\n  C4 per-rank peaks (GB): {[f'{p:.3f}' for p in peaks]}")
        print(f"        max across ranks:    {max_peak:.3f} GB")

        # Verdict
        del results["__peak_mem_rank0__"]
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k:24s}  {'PASS' if v else 'FAIL'}")
        all_ok = all(results.values())
        print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}\n")
        rc = 0 if all_ok else 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

### Caveats baked into the template

1. **The C4 peak-memory comparison is intentionally approximate.** `torch.cuda.max_memory_allocated()` is a process-lifetime high-water mark, and rank 0 also bakes the reference, so its peak is inflated by the single-GPU reference run. To get a clean C4, add `torch.cuda.reset_peak_memory_stats()` between `bake_reference()` and `run_fsdp2()` and re-measure. **Do this only if C4 is the deciding factor in the verdict** — on a tiny model the numbers are noisy anyway. For the real model story we'd want to run this at hidden=4096+ on a 80GB GPU.

2. **`peft_model.generate` under FSDP2 may misbehave** with KV cache and sharded params. If C5 fails with a generation-specific crash (NCCL hang, CUDA illegal-memory in attention), drop the generate path and rely on raw forward-logits comparison only. Note the gap in the verdict.

3. **The reference is generated by rank 0 with `forge.patch` NOT applied** — that's the right baseline (it's what HF + PEFT alone would do). Phase 2 already proved that `forge.patch(kernels=["lora_qkv","lora_mlp"])` matches PEFT-only forward bit-exact on single GPU, so a single-GPU `forge.patch` run isn't a separate ground truth — the PEFT-only reference IS the ground truth.

4. **The `forge.patch` call is AFTER `fully_shard`.** This is the right order because `fully_shard` mutates the module hierarchy; if you patched first, FSDP2 wouldn't see your patched forward in the wrapped module. Don't swap the order without testing both.

5. **For load_state_dict on a freshly built peft_model on each rank:** PEFT's `get_peft_model` is deterministic given the same seed and config, so each rank's structure matches. `load_state_dict(rank0_dict)` then overwrites all weights to ensure exact agreement (defensive — the seed should make this redundant, but it's cheap insurance against any nondeterminism in PEFT's adapter init).

---

## 11. Known footguns you'll trip over

(These all came up in Phases 1+2. Don't re-discover them.)

### Footgun A — the experiments/reference namespace collision

The two kernel directories `kernels/lora_qkv/` and `kernels/lora_mlp/` both do `sys.path.insert(0, "../..")` at module-import time and expose top-level packages named `experiments` and `reference`. Whichever one loads first wins those names in `sys.modules`, and the other kernel's `from experiments.v5...` imports then resolve to the wrong directory.

**Phase 1 workaround:** `test_lora_convergence.py` runs each kernel's loop in a subprocess. **This is also why Phase 1's `test_lora_qkv.py` and `test_lora_mlp.py` work standalone but break if you try to chain them.**

**For Phase 3:** the FSDP2 test only imports the LoRA kernels through `forge.kernels.lora_*` (which uses `sys.path.insert(0, _POC_ROOT)` and then `from kernels.lora_qkv...`), so as long as you import `forge` ONCE at the top of the test file you should be fine. **Don't import directly from `experiments.vN.*`** — go through the `forge.kernels.*` shims.

### Footgun B — `_cos_sim` on zero vectors

If both sides have zero gradients (the trivial-zero case for `lora_A` at init), cosine similarity is undefined and Python returns NaN or 0. The Phase 2 test has a specific trivial-zero branch that says "both are zero, max_diff is also zero, so PASS." Copy this pattern into the FSDP2 test (already in the template, see `_evaluate`).

### Footgun C — `verify_patch_gemma.py` has 2 stale pre-existing failures (NOT yours)

The `vram` check fails because the test was written for a model bigger than the one it actually builds, and the `stub_kernel_raises` check fails because a refactor renamed `cross_entropy` to `fused_linear_ce`. Both predate Phase 1. **If you see these in passing, ignore them.** Don't be tricked into "fixing" them — Xhitij has flagged both as out-of-scope.

### Footgun D — Gemma 2 alternates full / sliding-window attention by layer

`Gemma2Attention` sets `self.sliding_window` to either `None` or an int (typically 4096) depending on layer index. The LoRA factory already handles this — but if you're debugging a forward mismatch on a specific layer, check that the sliding window matches the reference. `module.sliding_window` is a per-instance attribute, NOT a config field.

### Footgun E — bf16 ground truth tolerances

Across all our tests so far, bf16 reduction-order noise lives at:
- `max_diff(logits) ~ 5e-2` (worst case, large hidden)
- `cos(logits) > 0.999` (almost always > 0.99999 on toy models)
- `dB grads`: `rel < 5e-3`, `cos > 0.99998`
- `dX grads`: `rel ~ 5e-3 to 9e-3`, `cos > 0.99999`

Don't tighten these thresholds for FSDP2. If you do and it fails, you're chasing precision noise, not a real bug.

---

## 12. Pointers to existing files you'll need to read

| File | Why |
|---|---|
| `forge/forge/patching/core.py` | The patch/unpatch loop. Notice it supports list-of-specs and falls through on `ForgeSkipPatch` |
| `forge/forge/patching/kernels/lora.py` | The factories you may need to rewrite (§7) |
| `forge/forge/patching/kernels/common.py` | `_lora_tensors()` extraction logic + `ForgeSkipPatch` |
| `forge/forge/patching/gemma.py` | `GEMMA_MAPPING` (list-of-specs entries for MLP, single spec for Attention) |
| `forge/tests/verify_patch_lora_gemma.py` | The Phase 2 test. The FSDP2 test mirrors its structure (build PEFT model → patch → compare). **Re-read this before writing the new file.** |
| `context/forge_hackathon_site/HACKATHON_SMOKE_TEST.md` | The H15 doc. Outcomes B–F (§6) of this resume doc are condensed from there |

---

## 13. If you have time after the test passes

In priority order (only after green verdict, only if Xhitij asks for more):
1. **Scale up to hidden=4096, num_layers=8** and re-run. This is what the H15 doc envisioned. Use a real C4 with `reset_peak_memory_stats()` between bake + FSDP2.
2. **Add a third check:** wrap a Qwen 3 model too (PEFT-wrapped, with the same kernels=["lora_qkv","lora_mlp"]). Confirms the SiLU path also survives FSDP2.
3. **Sweep `mp_policy`:** test with `(bf16, bf16, bf16)`, `(bf16, fp32, bf16)`, `(fp32, fp32, fp32)`. Document which combinations survive.
4. **Toy-train for longer:** 100 steps instead of 5. Confirm the loss curves stay aligned, not just the first few steps.

None of these are required for the smoke-test verdict. They're follow-up work.

---

## 14. The one-paragraph version (TL;DR if you forgot everything else)

Write `forge/tests/verify_fsdp2_lora_gemma.py` per the template in §10. Build the same PEFT-wrapped Gemma 2 used in Phase 2 (4 layers, hidden=128). Rank 0 bakes a single-GPU reference (forward logits, greedy-generate tokens, 5-step Adam loss curve, post-step-1 LoRA-A/B grads). All ranks load the same state dict, wrap each `Gemma2DecoderLayer` with `fully_shard(layer, mp_policy=MixedPrecisionPolicy(bf16, fp32))`, then call `forge.patch(model, kernels=["lora_qkv","lora_mlp"])`. Run the same inference + training paths, all-gather the LoRA grads on rank 0 with `.full_tensor()`, compare. Six checks: no-crash, forward parity (max_diff < 5e-2, cos > 0.999), grad parity (rel < 5e-2, cos > 0.9999, trivial-zero for lora_A), memory-sharded (informational on tiny model), generation parity (≥22/24 tokens), training convergence (per-step rel_diff < 2%). If C2 or C3 fails → rewrite `make_lora_qkv_forward` to use PEFT submodule calls instead of raw tensor extraction (§7). Time-box 3h. Don't commit.

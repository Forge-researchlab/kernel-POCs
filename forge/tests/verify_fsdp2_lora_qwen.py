"""FSDP2 + LoRA + Qwen 3 multi-GPU smoke test.

Mirror of verify_fsdp2_lora_gemma.py but for Qwen 3 — the SiLU LoRA-MLP
branch of make_lora_mlp_forward (kernels/lora.py) closes over raw weight
tensors at patch time, which is the same DTensor-incompatible pattern the
Phase 3 §7 fix addressed for the LoRA-QKV factory. This test is expected to
surface that bug; an analogous submodule rewrite of the SiLU MLP branch is
the fix.

Run:
    torchrun --nproc-per-node=2 --standalone forge/tests/verify_fsdp2_lora_qwen.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

import forge  # noqa: F401


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


def _build_peft_model(device, dtype, seed=SEED):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,        # GQA
        head_dim=32,
        max_position_embeddings=256,
        # Qwen 3 default activation is "silu" → exercises the SiLU LoRA-MLP branch.
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    peft_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
    )
    return get_peft_model(model, peft_cfg)


def _inner_model(peft_model):
    return peft_model.base_model.model.model


def _make_inputs(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=g).to(device)


def _make_micro_batches(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randint(0, VOCAB, (BATCH, SEQ), generator=g).to(device)
            for _ in range(MICRO_BATCHES)]


def _cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _all_gather_grad(param):
    g = param.grad
    if g is None:
        return None
    if hasattr(g, "full_tensor"):
        return g.full_tensor().detach().cpu()
    return g.detach().cpu()


LORA_GRAD_NAMES = {
    "q_lora_A":    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "q_lora_B":    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
    "gate_lora_A": "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight",
    "gate_lora_B": "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight",
}


def bake_reference(device, dtype, init_path, ref_path):
    print(f"[rank 0] baking single-GPU PEFT-only Qwen 3 reference ...")
    peft_model = _build_peft_model(device, dtype, seed=SEED)
    torch.save(peft_model.state_dict(), init_path)

    peft_model.eval()
    ids = _make_inputs(device, SEED + 1)
    with torch.no_grad():
        ref_logits = peft_model(ids).logits.detach().cpu().clone()
        try:
            ref_gen = peft_model.generate(
                ids[:1], max_new_tokens=GEN_TOKENS,
                do_sample=False, pad_token_id=0,
            ).detach().cpu().clone()
            gen_ok = True
        except Exception as e:
            print(f"[rank 0] reference generate failed: {type(e).__name__}: {e}")
            ref_gen = None
            gen_ok = False

    peft_model.train()
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    micro_batches = _make_micro_batches(device, SEED + 2)
    name_to_param = dict(peft_model.named_parameters())

    ref_losses = []
    ref_grads_after_step1 = None
    for step in range(TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mb in micro_batches:
            out = peft_model(mb, labels=mb)
            loss = out.loss / MICRO_BATCHES
            loss.backward()
            step_loss += float(out.loss.detach())
        if step == 0:
            ref_grads_after_step1 = {
                key: name_to_param[name].grad.detach().cpu().clone()
                for key, name in LORA_GRAD_NAMES.items()
            }
        opt.step()
        ref_losses.append(step_loss / MICRO_BATCHES)

    torch.save({
        "ref_logits": ref_logits,
        "ref_gen": ref_gen,
        "gen_ok": gen_ok,
        "ref_losses": ref_losses,
        "ref_grads": ref_grads_after_step1,
    }, ref_path)
    print(f"[rank 0] reference baked. logits {tuple(ref_logits.shape)}, "
          f"gen_ok={gen_ok}, losses {[f'{l:.4f}' for l in ref_losses]}")
    del peft_model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_fsdp2(rank, world, device, dtype, init_path):
    peft_model = _build_peft_model(device, dtype, seed=SEED)
    state = torch.load(init_path, map_location="cpu", weights_only=True)
    peft_model.load_state_dict(state)
    peft_model.to(device=device, dtype=dtype)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in _inner_model(peft_model).layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(_inner_model(peft_model), mp_policy=mp_policy)

    forge.patch(peft_model, kernels=["lora_qkv", "lora_mlp"])
    if rank == 0:
        print(f"[rank 0] patched_counts = {peft_model._forge_patched_counts}")

    peft_model.eval()
    ids = _make_inputs(device, SEED + 1)
    with torch.no_grad():
        logits = peft_model(ids).logits.detach().cpu().clone()
        try:
            gen = peft_model.generate(
                ids[:1], max_new_tokens=GEN_TOKENS,
                do_sample=False, pad_token_id=0,
            ).detach().cpu().clone()
            gen_ok = True
        except Exception as e:
            if rank == 0:
                print(f"[rank 0] FSDP2 generate failed: "
                      f"{type(e).__name__}: {e}")
            gen = None
            gen_ok = False

    peft_model.train()
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    micro_batches = _make_micro_batches(device, SEED + 2)
    name_to_param = dict(peft_model.named_parameters())

    losses = []
    step_metrics = []
    grads_after_step1 = None
    for step in range(TRAIN_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mb in micro_batches:
            out = peft_model(mb, labels=mb)
            loss = out.loss / MICRO_BATCHES
            loss.backward()
            step_loss += float(out.loss.detach())
        if step == 0:
            grads_after_step1 = {
                key: _all_gather_grad(name_to_param[name])
                for key, name in LORA_GRAD_NAMES.items()
            }
        opt.step()
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000
        avg_loss = step_loss / MICRO_BATCHES
        losses.append(avg_loss)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        step_metrics.append({
            "step": step, "loss": avg_loss,
            "step_time_ms": step_time_ms, "peak_vram_gb": peak_gb,
            "rank": rank,
        })

    return logits, gen, gen_ok, losses, grads_after_step1, step_metrics


def _evaluate(logits, gen, gen_ok, losses, grads, ref):
    results = {}

    diff = (logits.float() - ref["ref_logits"].float()).abs().max().item()
    cos = _cos(logits, ref["ref_logits"])
    results["C2_forward"] = (diff < 5e-2 and cos > 0.999)
    print(f"  C2 forward:      max_diff={diff:.3e} cos={cos:.6f}  "
          f"{'PASS' if results['C2_forward'] else 'FAIL'}")

    for key in ("q_lora_A", "q_lora_B", "gate_lora_A", "gate_lora_B"):
        a = grads[key]
        b = ref["ref_grads"][key]
        if a is None or b is None:
            print(f"  C3 grad[{key}]: MISSING  FAIL")
            results[f"C3_{key}"] = False
            continue
        base_max = b.abs().max().item()
        new_max = a.abs().max().item()
        d = (a.float() - b.float()).abs().max().item()
        trivial = (base_max < 1e-6 and new_max < 1e-6 and d < 1e-6)
        if trivial:
            ok = True
            print(f"  C3 grad[{key:12s}]: trivial-zero d={d:.2e}  PASS")
        else:
            rel = d / (base_max + 1e-12)
            c = _cos(a, b)
            ok = (rel < 5e-2 and c > 0.9999)
            print(f"  C3 grad[{key:12s}]: rel={rel:.2e} cos={c:.6f}  "
                  f"{'PASS' if ok else 'FAIL'}")
        results[f"C3_{key}"] = ok

    if not gen_ok or not ref["gen_ok"] or gen is None or ref["ref_gen"] is None:
        results["C5_generation"] = False
        print(f"  C5 generation:   SKIPPED (gen_ok fsdp={gen_ok} ref={ref['gen_ok']})  FAIL")
    else:
        n_match = int((gen[0] == ref["ref_gen"][0]).sum())
        results["C5_generation"] = (n_match >= 22)
        print(f"  C5 generation:   matched {n_match}/{gen.shape[1]} tokens  "
              f"{'PASS' if results['C5_generation'] else 'FAIL'}")

    rel_diffs = [abs(a - b) / (abs(b) + 1e-12)
                 for a, b in zip(losses, ref["ref_losses"])]
    converging_ref = ref["ref_losses"][-1] < ref["ref_losses"][0]
    converging_new = losses[-1] < losses[0]
    final_rel = abs(losses[-1] - ref["ref_losses"][-1]) / abs(ref["ref_losses"][-1])
    results["C6_training"] = (
        converging_ref and converging_new
        and max(rel_diffs) < 2e-2
        and final_rel < 5e-2
    )
    print(f"  C6 training:     ref={['%.4f'%x for x in ref['ref_losses']]} "
          f"fsdp={['%.4f'%x for x in losses]}")
    print(f"                   max_rel={max(rel_diffs):.2e} final_rel={final_rel:.2e}  "
          f"{'PASS' if results['C6_training'] else 'FAIL'}")

    return results


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    if rank == 0:
        print("=" * 70)
        print("FSDP2 + LoRA + Qwen 3 smoke test")
        print("=" * 70)
        print(f"  world_size={world}  torch={torch.__version__}  "
              f"device0={torch.cuda.get_device_name(0)}")
        print()

    tmpdir_holder = [None]
    if rank == 0:
        tmpdir_holder[0] = tempfile.mkdtemp(prefix="forge_fsdp_qwen_")
    dist.broadcast_object_list(tmpdir_holder, src=0)
    tmpdir = tmpdir_holder[0]
    init_path = os.path.join(tmpdir, "init_state.pt")
    ref_path = os.path.join(tmpdir, "reference.pt")

    if rank == 0:
        bake_reference(device, dtype, init_path, ref_path)
    dist.barrier()

    if rank == 0:
        print()
        print("=" * 70)
        print(f"FSDP2 run (world_size={world})")
        print("=" * 70)

    try:
        logits, gen, gen_ok, losses, grads = run_fsdp2(
            rank, world, device, dtype, init_path
        )
        run_exc = None
    except Exception as e:
        run_exc = e
        if rank == 0:
            print(f"[rank 0] run_fsdp2 raised: {type(e).__name__}: {e}")
            traceback.print_exc()
        logits = gen = losses = grads = None
        gen_ok = False

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    peaks_holder = [None] * world
    dist.all_gather_object(peaks_holder, peak_mem)

    rc = 1
    if rank == 0:
        if run_exc is None:
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            print()
            results = _evaluate(logits, gen, gen_ok, losses, grads, ref)

            print(f"  C4 per-rank peaks (GB): {[f'{p:.3f}' for p in peaks_holder]}")
            ratio = (min(peaks_holder) / max(peaks_holder)) if max(peaks_holder) > 0 else 1.0
            print(f"        min/max ratio: {ratio:.3f} (tiny model; informational)")

            print()
            print("=" * 70)
            print("VERDICT")
            print("=" * 70)
            for k in sorted(results.keys()):
                v = results[k]
                print(f"  {k:24s}  {'PASS' if v else 'FAIL'}")
            all_ok = all(results.values())
            print()
            print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
            rc = 0 if all_ok else 1
        else:
            print()
            print("=" * 70)
            print(f"VERDICT: CRASH ({type(run_exc).__name__}: {run_exc})")
            print("=" * 70)
            rc = 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())

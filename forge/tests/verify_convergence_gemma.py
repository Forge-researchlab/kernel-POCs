"""Multi-step training convergence test for forge.patch on Gemma 2.

Forward parity proves the kernels match at t=0. Convergence parity proves the
kernels don't drift over many optimizer steps — the only test that catches
slow gradient errors that accumulate into divergent training over hours.

Approach (deterministic A/B):
    1. Build TWO identical Gemma2 models with the same seed and same initial weights.
    2. Patch model B; leave model A unpatched.
    3. Same fixed batch + same optimizer + same seed for N steps.
    4. After every step, compare:
         - per-step loss (relative diff)
         - a representative weight tensor (cosine sim against model A)
    5. Pass if relative loss diff stays below ~2% throughout and weight cos sim
       stays above 0.9999. A real backward bug would compound and blow up here.

Bf16 + accumulation across 50 steps means tiny per-step rounding noise grows
slowly. Acceptance thresholds reflect that:
    - Step 1 (no accumulated drift): rel loss diff < 0.5%
    - Step N: rel loss diff < 2%, weight cos sim > 0.9999

Run:    python forge/tests/verify_convergence_gemma.py
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _build_pair(device, dtype, seed):
    """Two identical Gemma2 models — same init via deterministic seed."""
    import torch
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
        hidden_activation="gelu_pytorch_tanh",
    )

    torch.manual_seed(seed)
    model_a = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)

    torch.manual_seed(seed)
    model_b = Gemma2ForCausalLM(cfg).to(device=device, dtype=dtype)

    # Sanity: identical params at init
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(),
                                  model_b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"init mismatch on {na}"
    return model_a, model_b


def _train_one_step(model, ids, labels, optimizer):
    """One SGD step. Returns the scalar loss as a float."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(ids, labels=labels)
    loss = out.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _cos_sim(a, b):
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import forge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\n=== forge convergence test on Gemma2 ===")
    print(f"steps={args.steps} batch={args.batch} seq={args.seq_len} "
          f"lr={args.lr} dtype={dtype} device={device}\n")

    model_a, model_b = _build_pair(device, dtype, args.seed)
    print(f"built two identical Gemma2 models (init verified bit-equal)")

    # Patch B
    forge.patch(model_b)
    print(f"patched model B: {model_b._forge_patched_counts}\n")

    # Deterministic optimizer state: build both SGDs after identical RNG
    opt_a = torch.optim.SGD(model_a.parameters(), lr=args.lr, momentum=0.9)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=args.lr, momentum=0.9)

    # Fixed batch — convergence on the same data isolates kernel-induced drift
    # from data-order noise. Labels = input_ids (causal LM self-supervision).
    torch.manual_seed(args.seed + 999)
    ids = torch.randint(0, model_a.config.vocab_size,
                        (args.batch, args.seq_len), device=device)
    labels = ids.clone()

    # Representative weight to track (cosine sim) across both models
    track_a = dict(model_a.named_parameters())["model.layers.0.mlp.gate_proj.weight"]
    track_b = dict(model_b.named_parameters())["model.layers.0.mlp.gate_proj.weight"]

    print(f"{'step':>5} {'loss_A':>10} {'loss_B':>10} {'rel_diff':>10} "
          f"{'param_cos':>12}")

    losses_a, losses_b, rel_diffs, cos_sims = [], [], [], []
    fail_step = None
    for step in range(1, args.steps + 1):
        la = _train_one_step(model_a, ids, labels, opt_a)
        lb = _train_one_step(model_b, ids, labels, opt_b)
        rel = abs(la - lb) / (abs(la) + 1e-12)
        cos = _cos_sim(track_a.detach(), track_b.detach())

        losses_a.append(la)
        losses_b.append(lb)
        rel_diffs.append(rel)
        cos_sims.append(cos)

        if step <= 5 or step % 5 == 0 or step == args.steps:
            print(f"{step:5d} {la:10.4f} {lb:10.4f} {rel:10.4f} {cos:12.6f}")

        # Acceptance: relative loss diff < 2%, weight cos > 0.9999.
        # Step 1 must be very tight (< 0.5%) since no drift has accumulated yet.
        bar_rel = 0.005 if step == 1 else 0.02
        bar_cos = 0.9999
        if rel > bar_rel or cos < bar_cos:
            if fail_step is None:
                fail_step = step

    forge.unpatch(model_b)

    print("\n--- summary ---")
    print(f"  loss A: first={losses_a[0]:.4f}  last={losses_a[-1]:.4f}  "
          f"drop={losses_a[0] - losses_a[-1]:.4f}")
    print(f"  loss B: first={losses_b[0]:.4f}  last={losses_b[-1]:.4f}  "
          f"drop={losses_b[0] - losses_b[-1]:.4f}")
    print(f"  rel_diff: max={max(rel_diffs):.4f}  mean={sum(rel_diffs)/len(rel_diffs):.4f}")
    print(f"  param cos: min={min(cos_sims):.6f}  last={cos_sims[-1]:.6f}")

    converging_a = losses_a[-1] < losses_a[0]
    converging_b = losses_b[-1] < losses_b[0]
    final_rel = rel_diffs[-1]
    final_cos = cos_sims[-1]

    ok = (
        fail_step is None
        and converging_a
        and converging_b
        and final_rel < 0.02
        and final_cos > 0.9999
    )
    print(f"\n  both models converging?  A={converging_a}  B={converging_b}")
    print(f"  first failure step:       {fail_step if fail_step else 'none'}")
    print(f"  final rel loss diff:      {final_rel:.4e}  (bar < 2e-2)")
    print(f"  final param cos sim:      {final_cos:.6f}  (bar > 0.9999)")
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Multi-step optimization parity for LoRA QKV (v4) and LoRA MLP (v6) kernels.

Forward+backward parity catches kernel bugs at t=0. Convergence parity catches
slow gradient drift that compounds over many optimizer steps — a real-world
correctness signal that pure parity tests miss.

Setup (mirrors verify_convergence_gemma.py for the model-level Gemma test):
  1. Build two independent LoRA-param sets with the same seed → bit-equal init.
  2. Build two `torch.optim.Adam` instances — one over reference params, one
     over Forge-kernel params.
  3. Fixed input X and fixed target — same seed, generated once outside the loop.
  4. 50 steps: forward → MSE(out, target) → backward → optimizer.step().
  5. Per step compare: loss_ref, loss_fge, rel_diff, and a tracked LoRA-A
     weight's cosine sim between the two.

Acceptance bars (same shape as verify_convergence_gemma.py):
  - per-step rel loss diff < 2%
  - tracked LoRA-A weight cos sim > 0.9999
  - both must converge (final loss < initial loss)

Run:    python forge/tests/test_lora_convergence.py
Exit 0 on PASS, 1 on any FAIL.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import forge  # noqa: F401, E402  -- triggers POC-root sys.path injection


def _cos_sim(a, b):
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _convergence_loop(name, ref_apply, fge_apply, ref_params, fge_params,
                      shared_W, X, Y_target, tracked_name, steps=50, lr=1e-3):
    """Run a paired Adam optimization on reference vs Forge kernel.

    ref_apply / fge_apply: callables that take (X, *params, *Ws) and return the
                          (Q,K,V) tuple or Y tensor used to compute MSE vs Y_target.
    ref_params / fge_params: list of trainable tensors (same shapes, bit-equal init).
    shared_W: dict of frozen base weights (shared between sides).
    tracked_name: index into ref_params/fge_params of the LoRA-A weight to track.
    """
    import torch
    opt_ref = torch.optim.Adam(ref_params, lr=lr)
    opt_fge = torch.optim.Adam(fge_params, lr=lr)

    print(f"\n--- {name}: {steps} Adam steps (lr={lr}) ---")
    print(f"{'step':>5} {'loss_ref':>12} {'loss_fge':>12} {'rel_diff':>10} "
          f"{'param_cos':>12}")

    track_ref = ref_params[tracked_name]
    track_fge = fge_params[tracked_name]

    rel_diffs, cos_sims, losses_ref, losses_fge = [], [], [], []
    fail_step = None

    for step in range(1, steps + 1):
        opt_ref.zero_grad(set_to_none=True)
        opt_fge.zero_grad(set_to_none=True)

        out_ref = ref_apply(X, ref_params, shared_W)
        out_fge = fge_apply(X, fge_params, shared_W)

        # MSE against fixed target. For QKV the outputs are tuples — let the
        # caller-side apply functions return either a tensor or a tuple, and
        # compute MSE elementwise across the tuple if needed.
        if isinstance(out_ref, tuple):
            loss_ref = sum(((o - t) ** 2).mean() for o, t in zip(out_ref, Y_target))
            loss_fge = sum(((o - t) ** 2).mean() for o, t in zip(out_fge, Y_target))
        else:
            loss_ref = ((out_ref - Y_target) ** 2).mean()
            loss_fge = ((out_fge - Y_target) ** 2).mean()

        loss_ref.backward()
        loss_fge.backward()
        opt_ref.step()
        opt_fge.step()

        la = float(loss_ref.detach())
        lb = float(loss_fge.detach())
        rel = abs(la - lb) / (abs(la) + 1e-12)
        cos = _cos_sim(track_ref.detach(), track_fge.detach())

        losses_ref.append(la); losses_fge.append(lb)
        rel_diffs.append(rel); cos_sims.append(cos)

        if step <= 5 or step % 10 == 0 or step == steps:
            print(f"{step:5d} {la:12.5f} {lb:12.5f} {rel:10.4f} {cos:12.6f}")

        if rel > 0.02 or cos < 0.9999:
            if fail_step is None:
                fail_step = step

    converging_ref = losses_ref[-1] < losses_ref[0]
    converging_fge = losses_fge[-1] < losses_fge[0]
    print(f"\n  loss ref: {losses_ref[0]:.5f} -> {losses_ref[-1]:.5f}  "
          f"(drop={losses_ref[0] - losses_ref[-1]:.5f})")
    print(f"  loss fge: {losses_fge[0]:.5f} -> {losses_fge[-1]:.5f}  "
          f"(drop={losses_fge[0] - losses_fge[-1]:.5f})")
    print(f"  rel_diff: max={max(rel_diffs):.4e}  mean={sum(rel_diffs)/len(rel_diffs):.4e}")
    print(f"  param cos: min={min(cos_sims):.6f}  last={cos_sims[-1]:.6f}")
    print(f"  first failure step: {fail_step if fail_step else 'none'}")
    print(f"  converging?  ref={converging_ref}  fge={converging_fge}")

    ok = (
        fail_step is None
        and converging_ref and converging_fge
        and rel_diffs[-1] < 0.02
        and cos_sims[-1] > 0.9999
    )
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_lora_qkv_convergence():
    import torch
    from kernels.lora_qkv.reference.lora_qkv_pytorch import LoRAQKV
    from forge.kernels.lora_qkv import LoRAQKVv4Function

    device, dtype = "cuda", torch.bfloat16
    M, K, H_q, H_kv, r = 128, 256, 256, 128, 8
    torch.manual_seed(0)

    # Frozen shared base weights
    W_q = torch.randn(H_q, K, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(H_kv, K, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(H_kv, K, dtype=dtype, device=device) * 0.02

    # Trainable masters
    A_q = torch.randn(r, K, dtype=dtype, device=device) * 0.02
    A_k = torch.randn(r, K, dtype=dtype, device=device) * 0.02
    A_v = torch.randn(r, K, dtype=dtype, device=device) * 0.02
    B_q = torch.randn(H_q, r, dtype=dtype, device=device) * 0.02
    B_k = torch.randn(H_kv, r, dtype=dtype, device=device) * 0.02
    B_v = torch.randn(H_kv, r, dtype=dtype, device=device) * 0.02

    def _pair(t): return (t.detach().clone().requires_grad_(True),
                          t.detach().clone().requires_grad_(True))

    A_q_r, A_q_f = _pair(A_q)
    A_k_r, A_k_f = _pair(A_k)
    A_v_r, A_v_f = _pair(A_v)
    B_q_r, B_q_f = _pair(B_q)
    B_k_r, B_k_f = _pair(B_k)
    B_v_r, B_v_f = _pair(B_v)

    # Order: A_q, B_q, A_k, B_k, A_v, B_v — index 0 = A_q (tracked)
    ref_params = [A_q_r, B_q_r, A_k_r, B_k_r, A_v_r, B_v_r]
    fge_params = [A_q_f, B_q_f, A_k_f, B_k_f, A_v_f, B_v_f]

    # Fixed input + targets
    X = torch.randn(M, K, dtype=dtype, device=device)
    Q_target = torch.randn(M, H_q, dtype=dtype, device=device) * 0.5
    K_target = torch.randn(M, H_kv, dtype=dtype, device=device) * 0.5
    V_target = torch.randn(M, H_kv, dtype=dtype, device=device) * 0.5

    def ref_apply(X, p, W):
        return LoRAQKV.apply(
            X, W["W_q"], W["W_k"], W["W_v"],
            p[0], p[1], 1.0,  # A_q, B_q, s_q
            p[2], p[3], 1.0,  # A_k, B_k, s_k
            p[4], p[5], 1.0,  # A_v, B_v, s_v
        )

    def fge_apply(X, p, W):
        return LoRAQKVv4Function.apply(
            X, W["W_q"], W["W_k"], W["W_v"],
            p[0], p[1], 1.0,
            p[2], p[3], 1.0,
            p[4], p[5], 1.0,
        )

    return _convergence_loop(
        name="QKV v4",
        ref_apply=ref_apply, fge_apply=fge_apply,
        ref_params=ref_params, fge_params=fge_params,
        shared_W=dict(W_q=W_q, W_k=W_k, W_v=W_v),
        X=X, Y_target=(Q_target, K_target, V_target),
        tracked_name=0,   # A_q
    )


def test_lora_mlp_convergence():
    import torch
    from kernels.lora_mlp.reference.lora_mlp_pytorch import LoRAMLP
    from forge.kernels.lora_mlp import LoRAMLPv6

    device, dtype = "cuda", torch.bfloat16
    B, S, H, I, r = 2, 32, 128, 256, 8
    torch.manual_seed(0)

    W_gate = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_up   = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(H, I, dtype=dtype, device=device) * 0.02

    A_gate = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    A_up   = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    A_down = torch.randn(r, I, dtype=dtype, device=device) * 0.02
    B_gate = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    B_up   = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    B_down = torch.randn(H, r, dtype=dtype, device=device) * 0.02

    def _pair(t): return (t.detach().clone().requires_grad_(True),
                          t.detach().clone().requires_grad_(True))

    A_g_r, A_g_f = _pair(A_gate); B_g_r, B_g_f = _pair(B_gate)
    A_u_r, A_u_f = _pair(A_up);   B_u_r, B_u_f = _pair(B_up)
    A_d_r, A_d_f = _pair(A_down); B_d_r, B_d_f = _pair(B_down)

    # Order: A_gate, B_gate, A_up, B_up, A_down, B_down — index 0 tracked
    ref_params = [A_g_r, B_g_r, A_u_r, B_u_r, A_d_r, B_d_r]
    fge_params = [A_g_f, B_g_f, A_u_f, B_u_f, A_d_f, B_d_f]

    X = torch.randn(B, S, H, dtype=dtype, device=device)
    Y_target = torch.randn(B, S, H, dtype=dtype, device=device) * 0.5

    def ref_apply(X, p, W):
        return LoRAMLP.apply(
            X,
            W["W_gate"], p[0], p[1], 1.0,
            W["W_up"],   p[2], p[3], 1.0,
            W["W_down"], p[4], p[5], 1.0,
        )

    def fge_apply(X, p, W):
        return LoRAMLPv6.apply(
            X,
            W["W_gate"], p[0], p[1], 1.0,
            W["W_up"],   p[2], p[3], 1.0,
            W["W_down"], p[4], p[5], 1.0,
            None,   # A_stack
            True,   # enable_streams
            None,   # side_stream
        )

    return _convergence_loop(
        name="MLP v6",
        ref_apply=ref_apply, fge_apply=fge_apply,
        ref_params=ref_params, fge_params=fge_params,
        shared_W=dict(W_gate=W_gate, W_up=W_up, W_down=W_down),
        X=X, Y_target=Y_target,
        tracked_name=0,   # A_gate
    )


def _run_subtest_inproc(which):
    """Used when re-invoked with --subtest. Runs ONE kernel test only so the
    process has a clean sys.path — both lora_qkv and lora_mlp kernels do
    `sys.path.insert(0, "../..")` at import time and expose top-level
    `experiments` / `reference` packages. Running both in one process makes
    whichever loads first win those namespaces and the other kernel breaks
    when its own `experiments.vN` siblings are masked. Separate processes
    sidestep this entirely.
    """
    import torch
    if not torch.cuda.is_available():
        print("CUDA required"); return 1

    if which == "qkv":
        ok = test_lora_qkv_convergence()
    elif which == "mlp":
        ok = test_lora_mlp_convergence()
    else:
        print(f"unknown subtest: {which}"); return 1
    return 0 if ok else 1


def main():
    import subprocess

    if len(sys.argv) > 1 and sys.argv[1] == "--subtest":
        return _run_subtest_inproc(sys.argv[2])

    print(f"\n=== LoRA convergence parity (50 Adam steps each) ===")
    print(f"Running QKV and MLP in separate subprocesses to dodge the\n"
          f"experiments/reference namespace collision between the two kernel dirs.\n")

    results = {}
    for label, which in [("QKV v4", "qkv"), ("MLP v6", "mlp")]:
        print("=" * 72)
        print(f"  subprocess: {label}")
        print("=" * 72)
        proc = subprocess.run(
            [sys.executable, __file__, "--subtest", which],
            cwd=os.environ.get("PWD", os.getcwd()),
        )
        results[label] = (proc.returncode == 0)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:14s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  OVERALL ({sum(results.values())}/{len(results)}): "
          f"{'PASS' if all_ok else 'FAIL'}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

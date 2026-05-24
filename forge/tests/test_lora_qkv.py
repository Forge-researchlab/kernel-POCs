"""Direct kernel tests for LoRAQKVv4Function (forward + backward parity).

These tests bypass forge.patch and the PEFT layer — they call the Triton kernel
directly with fabricated tensors and compare to the plain-PyTorch reference
(`LoRAQKV` in kernels/lora_qkv/reference/lora_qkv_pytorch.py). A failure here
points at the kernel itself, not at any patching wiring.

Coverage matrix (12 cases): rank ∈ {8, 16}, dtype ∈ {bf16, fp16}, shape ∈
{2D, 3D}, GQA on/off, odd seq length for masking, different per-projection
scalings, and a tiny boundary shape.

For each case:
  * Forward Q/K/V compared via max_diff and cos_sim
  * Backward gradients dX, dA_q, dB_q, dA_k, dB_k, dA_v, dB_v compared via
    relative max-diff and cos_sim (rel because grad magnitudes vary by
    orders of magnitude across the seven tensors)
  * NaN/Inf check on every tensor

Run:    python forge/tests/test_lora_qkv.py
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

# Importing forge inserts the POC root into sys.path so `from kernels.*`
# (the reference module) is importable. See forge/forge/__init__.py.
import forge  # noqa: F401, E402


def _cos_sim(a, b):
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _summary_tensor(label, a, b, dtype, kind="forward"):
    """Compare a Forge output against the reference. Returns (ok, line).

    forward bars (bf16): max_diff < 5e-2, cos > 0.999
    forward bars (fp16): max_diff < 1e-2, cos > 0.9995
    grad bars (both):    rel_max_diff < 5e-2, cos > 0.9999
    """
    import torch
    nan = int(a.isnan().sum().item() + b.isnan().sum().item())
    inf = int(a.isinf().sum().item() + b.isinf().sum().item())
    shape_ok = (a.shape == b.shape)
    if not shape_ok:
        return False, f"  [{kind} {label:8s}] SHAPE MISMATCH ref={tuple(a.shape)} fge={tuple(b.shape)}"

    max_diff = (a - b).abs().max().item()
    base_max = a.abs().max().item()
    rel = max_diff / (base_max + 1e-12)
    cos = _cos_sim(a, b)

    import torch
    if kind == "forward":
        if dtype == torch.bfloat16:
            ok = (nan == 0 and inf == 0 and max_diff < 5e-2 and cos > 0.999)
        else:  # fp16
            ok = (nan == 0 and inf == 0 and max_diff < 1e-2 and cos > 0.9995)
        line = (f"  [fwd  {label:6s}] max_diff={max_diff:.3e} cos={cos:.6f} "
                f"NaN={nan} Inf={inf}")
    else:  # grad
        ok = (nan == 0 and inf == 0 and rel < 5e-2 and cos > 0.9999)
        line = (f"  [grad {label:6s}] base_max={base_max:.2e} max_diff={max_diff:.2e} "
                f"rel={rel:.3e} cos={cos:.6f} NaN={nan}")
    line += f"  {'PASS' if ok else 'FAIL'}"
    return ok, line


def _build_inputs(M_or_BS, hidden, H_q, H_kv, r, dtype, device, three_d, seed):
    """Build a paired set of inputs for the reference and the Forge kernel.

    Shared (frozen): W_q, W_k, W_v.
    Cloned per side (requires_grad): X, A_q/k/v, B_q/k/v.
    Returns: (ref_inputs, fge_inputs, scalings) where each *_inputs is a dict.
    """
    import torch

    torch.manual_seed(seed)
    # Frozen base weights — shared between ref and fge
    W_q = torch.randn(H_q, hidden, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(H_kv, hidden, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(H_kv, hidden, dtype=dtype, device=device) * 0.02

    # Master tensors for the trainable params (small init typical of LoRA)
    if three_d:
        # M_or_BS is (B, S)
        B, S = M_or_BS
        X_master = torch.randn(B, S, hidden, dtype=dtype, device=device)
    else:
        X_master = torch.randn(M_or_BS, hidden, dtype=dtype, device=device)
    A_q_master = torch.randn(r, hidden, dtype=dtype, device=device) * 0.02
    A_k_master = torch.randn(r, hidden, dtype=dtype, device=device) * 0.02
    A_v_master = torch.randn(r, hidden, dtype=dtype, device=device) * 0.02
    # B init is non-zero so the LoRA contribution is exercised (PEFT inits B to
    # zero, which would zero the gradient signal we want to check).
    B_q_master = torch.randn(H_q, r, dtype=dtype, device=device) * 0.02
    B_k_master = torch.randn(H_kv, r, dtype=dtype, device=device) * 0.02
    B_v_master = torch.randn(H_kv, r, dtype=dtype, device=device) * 0.02

    def _pair(master):
        ref = master.detach().clone().requires_grad_(True)
        fge = master.detach().clone().requires_grad_(True)
        return ref, fge

    X_ref, X_fge = _pair(X_master)
    A_q_ref, A_q_fge = _pair(A_q_master)
    A_k_ref, A_k_fge = _pair(A_k_master)
    A_v_ref, A_v_fge = _pair(A_v_master)
    B_q_ref, B_q_fge = _pair(B_q_master)
    B_k_ref, B_k_fge = _pair(B_k_master)
    B_v_ref, B_v_fge = _pair(B_v_master)

    return (
        dict(X=X_ref, A_q=A_q_ref, A_k=A_k_ref, A_v=A_v_ref,
             B_q=B_q_ref, B_k=B_k_ref, B_v=B_v_ref),
        dict(X=X_fge, A_q=A_q_fge, A_k=A_k_fge, A_v=A_v_fge,
             B_q=B_q_fge, B_k=B_k_fge, B_v=B_v_fge),
        dict(W_q=W_q, W_k=W_k, W_v=W_v),
    )


def _run_case(label, M_or_BS, hidden, H_q, H_kv, r, dtype, three_d, s_q, s_k, s_v, seed):
    import torch
    from kernels.lora_qkv.reference.lora_qkv_pytorch import LoRAQKV
    from forge.kernels.lora_qkv import LoRAQKVv4Function

    device = "cuda"
    ref_in, fge_in, frozen = _build_inputs(
        M_or_BS, hidden, H_q, H_kv, r, dtype, device, three_d, seed,
    )

    def _fwd_bwd(side):
        Q, K, V = side
        loss = Q.sum() + K.sum() + V.sum()
        loss.backward()
        return Q.detach(), K.detach(), V.detach()

    # Reference
    try:
        Q_ref, K_ref, V_ref = LoRAQKV.apply(
            ref_in["X"], frozen["W_q"], frozen["W_k"], frozen["W_v"],
            ref_in["A_q"], ref_in["B_q"], s_q,
            ref_in["A_k"], ref_in["B_k"], s_k,
            ref_in["A_v"], ref_in["B_v"], s_v,
        )
        Q_ref_d, K_ref_d, V_ref_d = _fwd_bwd((Q_ref, K_ref, V_ref))
    except Exception as e:
        print(f"  REFERENCE EXCEPTION in {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Forge v4
    try:
        Q_fge, K_fge, V_fge = LoRAQKVv4Function.apply(
            fge_in["X"], frozen["W_q"], frozen["W_k"], frozen["W_v"],
            fge_in["A_q"], fge_in["B_q"], s_q,
            fge_in["A_k"], fge_in["B_k"], s_k,
            fge_in["A_v"], fge_in["B_v"], s_v,
        )
        Q_fge_d, K_fge_d, V_fge_d = _fwd_bwd((Q_fge, K_fge, V_fge))
    except Exception as e:
        print(f"  FORGE EXCEPTION in {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print(f"  --- case {label} | M_or_BS={M_or_BS} hidden={hidden} H_q={H_q} "
          f"H_kv={H_kv} r={r} dtype={dtype} 3D={three_d} ---")

    ok = True
    # Forward
    for name, a, b in [("Q", Q_ref_d, Q_fge_d), ("K", K_ref_d, K_fge_d), ("V", V_ref_d, V_fge_d)]:
        case_ok, line = _summary_tensor(name, a, b, dtype, kind="forward")
        print(line)
        ok &= case_ok

    # Backward
    pairs = [
        ("dX",   ref_in["X"].grad,   fge_in["X"].grad),
        ("dA_q", ref_in["A_q"].grad, fge_in["A_q"].grad),
        ("dB_q", ref_in["B_q"].grad, fge_in["B_q"].grad),
        ("dA_k", ref_in["A_k"].grad, fge_in["A_k"].grad),
        ("dB_k", ref_in["B_k"].grad, fge_in["B_k"].grad),
        ("dA_v", ref_in["A_v"].grad, fge_in["A_v"].grad),
        ("dB_v", ref_in["B_v"].grad, fge_in["B_v"].grad),
    ]
    for name, a, b in pairs:
        if a is None or b is None:
            print(f"  [grad {name:6s}] MISSING grad (ref={a is not None} fge={b is not None})  FAIL")
            ok = False
            continue
        case_ok, line = _summary_tensor(name, a, b, dtype, kind="grad")
        print(line)
        ok &= case_ok

    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    import torch
    if not torch.cuda.is_available():
        print("CUDA required"); return 1

    print(f"\n=== LoRAQKVv4Function parity (forward + backward) ===")
    print(f"reference: kernels/lora_qkv/reference/lora_qkv_pytorch.py :: LoRAQKV")
    print(f"under test: kernels/lora_qkv/experiments/v4 :: LoRAQKVv4Function\n")

    bf16, fp16 = torch.bfloat16, torch.float16

    # Coverage matrix — see plan for rationale
    cases = [
        # label,         M_or_BS,       hidden, H_q, H_kv, r,  dtype, 3D,    s_q, s_k, s_v, seed
        ("A1",           64,            256,    256, 256,  8,  bf16,  False, 1.0, 1.0, 1.0, 0),
        ("A2",           64,            256,    256, 256,  16, bf16,  False, 1.0, 1.0, 1.0, 1),
        ("A3",           64,            256,    256, 256,  8,  fp16,  False, 1.0, 1.0, 1.0, 2),
        ("B1",           (2, 32),       256,    256, 256,  8,  bf16,  True,  1.0, 1.0, 1.0, 3),
        ("B2",           (2, 32),       256,    256, 256,  16, bf16,  True,  1.0, 1.0, 1.0, 4),
        ("B3",           (4, 17),       256,    256, 256,  8,  bf16,  True,  1.0, 1.0, 1.0, 5),
        ("C1_GQA",       (2, 32),       256,    256, 128,  8,  bf16,  True,  1.0, 1.0, 1.0, 6),
        ("C2_GQA",       (2, 32),       256,    256, 128,  16, bf16,  True,  1.0, 1.0, 1.0, 7),
        ("C3_GQA_fp16",  (2, 32),       256,    256, 128,  8,  fp16,  True,  1.0, 1.0, 1.0, 8),
        ("D1_8B_GQA",    (2, 64),       512,    512, 128,  8,  bf16,  True,  1.0, 1.0, 1.0, 9),
        ("E1_tiny",      (1, 8),        256,    256, 256,  8,  bf16,  True,  1.0, 1.0, 1.0, 10),
        ("E2_scalings",  (2, 32),       256,    256, 256,  8,  bf16,  True,  2.0, 1.5, 0.5, 11),
    ]

    results = {}
    for case in cases:
        results[case[0]] = _run_case(*case)

    print("=" * 72)
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

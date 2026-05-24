"""Direct kernel tests for LoRAMLPv6 (forward + backward parity).

Mirrors test_lora_qkv.py: call the Triton kernel directly with fabricated
tensors, compare to the plain-PyTorch reference (`LoRAMLP` in
kernels/lora_mlp/reference/lora_mlp_pytorch.py).

Note: both the reference and v6 backward unpack `batch, seq_len, hd = X.shape`,
so X must be 3D. All cases use 3D input.

Coverage matrix (8 cases): rank ∈ {8, 16}, dtype ∈ {bf16, fp16}, shape variety
including odd seq for masking and one LLaMA-8B-ish (hidden=512, intermediate=2048),
different per-projection scalings, and the enable_streams=False path.

For each case:
  * Forward output compared via max_diff and cos_sim
  * Backward gradients dX, dA_gate, dB_gate, dA_up, dB_up, dA_down, dB_down
    compared via relative max-diff and cos_sim
  * NaN/Inf check on every tensor

Run:    python forge/tests/test_lora_mlp.py
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


def _cos_sim(a, b):
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _summary_tensor(label, a, b, dtype, kind="forward"):
    import torch
    nan = int(a.isnan().sum().item() + b.isnan().sum().item())
    inf = int(a.isinf().sum().item() + b.isinf().sum().item())
    if a.shape != b.shape:
        return False, f"  [{kind} {label:10s}] SHAPE MISMATCH ref={tuple(a.shape)} fge={tuple(b.shape)}"

    max_diff = (a - b).abs().max().item()
    base_max = a.abs().max().item()
    rel = max_diff / (base_max + 1e-12)
    cos = _cos_sim(a, b)

    if kind == "forward":
        if dtype == torch.bfloat16:
            ok = (nan == 0 and inf == 0 and max_diff < 5e-2 and cos > 0.999)
        else:
            ok = (nan == 0 and inf == 0 and max_diff < 1e-2 and cos > 0.9995)
        line = (f"  [fwd  {label:8s}] max_diff={max_diff:.3e} cos={cos:.6f} "
                f"NaN={nan} Inf={inf}")
    else:
        ok = (nan == 0 and inf == 0 and rel < 5e-2 and cos > 0.9999)
        line = (f"  [grad {label:8s}] base_max={base_max:.2e} max_diff={max_diff:.2e} "
                f"rel={rel:.3e} cos={cos:.6f} NaN={nan}")
    line += f"  {'PASS' if ok else 'FAIL'}"
    return ok, line


def _build_inputs(B, S, hidden, intermediate, r, dtype, device, seed):
    """Build paired inputs for the reference and Forge kernel.

    Shared (frozen): W_gate, W_up, W_down.
    Cloned per side (requires_grad=True): X, A_*, B_*.
    """
    import torch
    torch.manual_seed(seed)
    H, I = hidden, intermediate

    # Frozen base weights
    W_gate = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_up   = torch.randn(I, H, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(H, I, dtype=dtype, device=device) * 0.02

    # Trainable masters (LoRA-typical small init; B non-zero so the LoRA
    # contribution is exercised — PEFT's default B=0 would mask gradients)
    X_master = torch.randn(B, S, H, dtype=dtype, device=device)
    A_gate_master = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    A_up_master   = torch.randn(r, H, dtype=dtype, device=device) * 0.02
    A_down_master = torch.randn(r, I, dtype=dtype, device=device) * 0.02
    B_gate_master = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    B_up_master   = torch.randn(I, r, dtype=dtype, device=device) * 0.02
    B_down_master = torch.randn(H, r, dtype=dtype, device=device) * 0.02

    def _pair(t):
        return (t.detach().clone().requires_grad_(True),
                t.detach().clone().requires_grad_(True))

    X_ref, X_fge = _pair(X_master)
    A_gate_ref, A_gate_fge = _pair(A_gate_master)
    A_up_ref,   A_up_fge   = _pair(A_up_master)
    A_down_ref, A_down_fge = _pair(A_down_master)
    B_gate_ref, B_gate_fge = _pair(B_gate_master)
    B_up_ref,   B_up_fge   = _pair(B_up_master)
    B_down_ref, B_down_fge = _pair(B_down_master)

    return (
        dict(X=X_ref, A_gate=A_gate_ref, A_up=A_up_ref, A_down=A_down_ref,
             B_gate=B_gate_ref, B_up=B_up_ref, B_down=B_down_ref),
        dict(X=X_fge, A_gate=A_gate_fge, A_up=A_up_fge, A_down=A_down_fge,
             B_gate=B_gate_fge, B_up=B_up_fge, B_down=B_down_fge),
        dict(W_gate=W_gate, W_up=W_up, W_down=W_down),
    )


def _run_case(label, B, S, hidden, intermediate, r, dtype,
              s_gate, s_up, s_down, enable_streams, seed):
    import torch
    from kernels.lora_mlp.reference.lora_mlp_pytorch import LoRAMLP
    from forge.kernels.lora_mlp import LoRAMLPv6

    device = "cuda"
    ref_in, fge_in, W = _build_inputs(B, S, hidden, intermediate, r, dtype, device, seed)

    print(f"  --- case {label} | B={B} S={S} H={hidden} I={intermediate} r={r} "
          f"dtype={dtype} streams={enable_streams} ---")

    # Reference
    try:
        Y_ref = LoRAMLP.apply(
            ref_in["X"],
            W["W_gate"], ref_in["A_gate"], ref_in["B_gate"], s_gate,
            W["W_up"],   ref_in["A_up"],   ref_in["B_up"],   s_up,
            W["W_down"], ref_in["A_down"], ref_in["B_down"], s_down,
        )
        Y_ref.sum().backward()
    except Exception as e:
        print(f"  REFERENCE EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Forge v6
    try:
        Y_fge = LoRAMLPv6.apply(
            fge_in["X"],
            W["W_gate"], fge_in["A_gate"], fge_in["B_gate"], s_gate,
            W["W_up"],   fge_in["A_up"],   fge_in["B_up"],   s_up,
            W["W_down"], fge_in["A_down"], fge_in["B_down"], s_down,
            None,           # A_stack (None → computed internally)
            enable_streams, # enable_streams
            None,           # side_stream
        )
        Y_fge.sum().backward()
    except Exception as e:
        print(f"  FORGE EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    ok = True
    # Forward
    case_ok, line = _summary_tensor("Y", Y_ref.detach(), Y_fge.detach(), dtype, kind="forward")
    print(line)
    ok &= case_ok

    # Backward
    pairs = [
        ("dX",      ref_in["X"].grad,      fge_in["X"].grad),
        ("dA_gate", ref_in["A_gate"].grad, fge_in["A_gate"].grad),
        ("dB_gate", ref_in["B_gate"].grad, fge_in["B_gate"].grad),
        ("dA_up",   ref_in["A_up"].grad,   fge_in["A_up"].grad),
        ("dB_up",   ref_in["B_up"].grad,   fge_in["B_up"].grad),
        ("dA_down", ref_in["A_down"].grad, fge_in["A_down"].grad),
        ("dB_down", ref_in["B_down"].grad, fge_in["B_down"].grad),
    ]
    for name, a, b in pairs:
        if a is None or b is None:
            print(f"  [grad {name:8s}] MISSING grad (ref={a is not None} fge={b is not None})  FAIL")
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

    print(f"\n=== LoRAMLPv6 parity (forward + backward) ===")
    print(f"reference:  kernels/lora_mlp/reference/lora_mlp_pytorch.py :: LoRAMLP")
    print(f"under test: kernels/lora_mlp/experiments/v6 :: LoRAMLPv6\n")

    bf16, fp16 = torch.bfloat16, torch.float16

    # Coverage matrix — all 3D since both ref and v6 backward unpack 3 dims.
    cases = [
        # label,       B, S,  hidden, intermediate, r,  dtype, s_g, s_u, s_d, streams, seed
        ("A1",         1, 64, 128,    256,          8,  bf16,  1.0, 1.0, 1.0, True,    0),
        ("A2",         1, 64, 128,    256,          16, bf16,  1.0, 1.0, 1.0, True,    1),
        ("A3_fp16",    1, 64, 128,    256,          8,  fp16,  1.0, 1.0, 1.0, True,    2),
        ("B1_3D",      2, 32, 128,    256,          8,  bf16,  1.0, 1.0, 1.0, True,    3),
        ("B2_oddseq",  4, 17, 128,    256,          8,  bf16,  1.0, 1.0, 1.0, True,    4),
        ("C1_8B-ish",  2, 64, 512,    2048,         16, bf16,  1.0, 1.0, 1.0, True,    5),
        ("D1_scal",    2, 32, 128,    256,          8,  bf16,  2.0, 1.5, 0.5, True,    6),
        ("D2_nostr",   2, 32, 128,    256,          8,  bf16,  1.0, 1.0, 1.0, False,   7),
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

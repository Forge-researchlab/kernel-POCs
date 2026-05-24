"""Direct kernel tests for ForgeEmbeddingFunction's padding_idx support.

These tests bypass forge.patch and call the kernel directly, so a failure
points at the kernel itself — not at any model wiring. Coverage matrix:

    1. padding_idx=None on small input        -> index_add fallback path
    2. padding_idx=None on large input        -> sort path
    3. padding_idx set, pad NOT in input      -> kernel must NOT crash, result == ref
    4. padding_idx set, few pad tokens        -> index_add fallback + zero
    5. padding_idx set, many pad tokens       -> sort path + zero
    6. padding_idx set, very many pad tokens  -> sort+cooperative path + zero
    7. padding_idx=mid-vocab id (not 0)       -> kernel respects arbitrary indices

Reference for every case: torch.nn.functional.embedding with the same padding_idx.
A real bug would surface as either NaN, wildly wrong magnitude on the pad row,
or non-zero pad row when PyTorch returns zero.

Run:    python forge/tests/test_embedding_padding_idx.py
Exit 0 on PASS, 1 on any FAIL.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _check_case(label, vocab, dim, ids_shape, n_pad, pad_idx, sort_path_expected,
                cooperative_expected, dtype, device):
    """Run one (input distribution, padding_idx) configuration.

    Returns True on pass, False on fail. Compares Forge kernel output and
    gradient against PyTorch's reference, both with the same padding_idx.
    """
    import torch
    from forge.kernels.embedding import ForgeEmbeddingFunction
    from kernels.embedding.experiments.v1.embedding_kernel_v1_upgrade_1 import (
        SORT_BACKWARD_THRESHOLD,
        COOPERATIVE_GROUP_THRESHOLD,
    )

    torch.manual_seed(0)

    # Build ids
    n_tokens = 1
    for s in ids_shape:
        n_tokens *= s
    flat = torch.randint(0, vocab, (n_tokens,), device=device)
    # Ensure the input has no incidental pads first (so n_pad is exact)
    if pad_idx is not None:
        flat[flat == pad_idx] = (pad_idx + 1) % vocab
    if n_pad > 0:
        assert pad_idx is not None, "n_pad>0 requires pad_idx set"
        pad_positions = torch.randperm(n_tokens, device=device)[:n_pad]
        flat[pad_positions] = pad_idx
    ids = flat.view(*ids_shape)

    # Sanity: confirm we're hitting the path we expect
    n_elements = ids.numel()
    will_sort = n_elements >= SORT_BACKWARD_THRESHOLD
    if sort_path_expected is not None:
        if will_sort != sort_path_expected:
            print(f"  [{label}] WARN: expected sort_path={sort_path_expected} "
                  f"but kernel will pick sort_path={will_sort} for "
                  f"n_elements={n_elements} vs threshold={SORT_BACKWARD_THRESHOLD}")

    # Identical weights for both paths
    torch.manual_seed(42)
    w_ref = torch.randn(vocab, dim, dtype=dtype, device=device, requires_grad=True)
    w_fge = w_ref.detach().clone().requires_grad_(True)

    # Forward + backward — PyTorch reference
    out_ref = torch.nn.functional.embedding(ids, w_ref, padding_idx=pad_idx)
    grad_seed = torch.randn_like(out_ref)
    out_ref.backward(grad_seed)

    # Forward + backward — Forge kernel
    out_fge = ForgeEmbeddingFunction.apply(w_fge, ids, pad_idx)
    out_fge.backward(grad_seed)

    # === Checks ===
    # 1. Forward must be bit-exact (kernel forward does no math, just gather)
    fwd_max = (out_ref - out_fge).abs().max().item()

    # 2. Backward overall — same row order, bf16 reduction order difference is allowed
    g_ref = w_ref.grad
    g_fge = w_fge.grad
    nan_count = int(g_fge.isnan().sum().item())
    bwd_max = (g_ref - g_fge).abs().max().item()
    base_max = g_ref.abs().max().item()
    rel = bwd_max / (base_max + 1e-12)
    # cos sim — directional check independent of scale
    cos = float(
        (g_ref.float().flatten() @ g_fge.float().flatten())
        / (g_ref.float().norm() * g_fge.float().norm() + 1e-12)
    )

    # 3. Padding-row check — must be exactly zero if padding_idx set, regardless of dups
    pad_row_max = None
    pad_row_ok = True
    if pad_idx is not None:
        pad_row_max = g_fge[pad_idx].abs().max().item()
        pad_row_ok = (pad_row_max == 0.0)

    ok = (
        nan_count == 0
        and fwd_max == 0.0
        and rel < 0.05
        and cos > 0.999
        and pad_row_ok
    )
    flag = "PASS" if ok else "FAIL"
    print(f"  [{label}]")
    print(f"      n_elements={n_elements} n_pad={n_pad} pad_idx={pad_idx} "
          f"sort_path={will_sort} dtype={dtype}")
    print(f"      forward    max_diff = {fwd_max:.4e}")
    print(f"      backward   max_diff = {bwd_max:.4e}  rel = {rel:.4e}  cos = {cos:.6f}  NaN={nan_count}")
    if pad_idx is not None:
        print(f"      pad row {pad_idx} max abs = {pad_row_max:.4e}  "
              f"(must be 0.0 to honor padding_idx)")
    print(f"      -> {flag}\n")
    return ok


def main():
    import torch
    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n=== ForgeEmbeddingFunction padding_idx unit tests ===\n")

    results = {}

    # --- Path 1: padding_idx=None (back-compat) on small input -> index_add fallback ---
    results["A: None / small / index_add"] = _check_case(
        "A: padding_idx=None, n_tokens=64 (index_add fallback)",
        vocab=128, dim=64, ids_shape=(2, 32),
        n_pad=0, pad_idx=None,
        sort_path_expected=False, cooperative_expected=False,
        dtype=dtype, device=device,
    )

    # --- Path 2: padding_idx=None on large input -> sort path ---
    results["B: None / large / sort"] = _check_case(
        "B: padding_idx=None, n_tokens=512 (sort path)",
        vocab=1024, dim=128, ids_shape=(4, 128),
        n_pad=0, pad_idx=None,
        sort_path_expected=True, cooperative_expected=False,
        dtype=dtype, device=device,
    )

    # --- Path 3: pad_idx set but pad NEVER in input ---
    # Kernel must not crash, must produce identical-to-ref grad
    # (row pad_idx is naturally 0 since it's never accumulated to)
    results["C: pad_idx=0 / not in input"] = _check_case(
        "C: pad_idx=0, but no pad tokens in input",
        vocab=1024, dim=128, ids_shape=(4, 128),
        n_pad=0, pad_idx=0,
        sort_path_expected=True, cooperative_expected=False,
        dtype=dtype, device=device,
    )

    # --- Path 4: padding_idx set, small input (index_add fallback) ---
    results["D: pad_idx=0 / small / few pads"] = _check_case(
        "D: pad_idx=0, n_tokens=64, ~25% pads (fallback + zero)",
        vocab=128, dim=64, ids_shape=(2, 32),
        n_pad=16, pad_idx=0,
        sort_path_expected=False, cooperative_expected=False,
        dtype=dtype, device=device,
    )

    # --- Path 5: padding_idx set, large input, modest duplicates (sort, no cooperative) ---
    # 20 pads of total 512 -> max group ~20 < COOPERATIVE_GROUP_THRESHOLD(32) -> v1 path
    results["E: pad_idx=0 / sort / no coop"] = _check_case(
        "E: pad_idx=0, n_tokens=512, ~20 pad tokens (sort v1 path)",
        vocab=1024, dim=128, ids_shape=(4, 128),
        n_pad=20, pad_idx=0,
        sort_path_expected=True, cooperative_expected=False,
        dtype=dtype, device=device,
    )

    # --- Path 6: padding_idx set, large input, heavy duplicates (cooperative) ---
    # 460 pads -> cooperative path. This is the exact failing case from
    # verify_patch_gemma.py [7] — must now PASS.
    results["F: pad_idx=0 / coop / 90% pad"] = _check_case(
        "F: pad_idx=0, n_tokens=512, ~460 pad tokens (cooperative path)",
        vocab=1024, dim=128, ids_shape=(4, 128),
        n_pad=460, pad_idx=0,
        sort_path_expected=True, cooperative_expected=True,
        dtype=dtype, device=device,
    )

    # --- Path 7: padding_idx is NOT 0 — arbitrary index ---
    # Make sure we're zeroing the right row, not just row 0 by accident
    results["G: pad_idx=137 / coop"] = _check_case(
        "G: pad_idx=137 (arbitrary, not 0), heavy duplicates",
        vocab=1024, dim=128, ids_shape=(4, 128),
        n_pad=400, pad_idx=137,
        sort_path_expected=True, cooperative_expected=True,
        dtype=dtype, device=device,
    )

    # --- Verdict ---
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:38s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

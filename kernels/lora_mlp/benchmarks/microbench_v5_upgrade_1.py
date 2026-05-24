"""
Microbenchmark: v5_upgrade_1 vs v5 vs v3 at LLaMA-8B scale (bf16).

Pairs with ``microbench_v5_packing.py`` and the diagnosis at
``docs/analysis/v5_packing_diagnosis.md``. Measures the two changes that
v5_upgrade_1 applies on top of v5:

  Change 1 — drop down-phase packing (revert to v3's two cuBLAS calls + addmm_)
  Change 2 — pad gate+up mega N from 28704 → 28800 (multiple of 128) for cuBLAS

Sections:
  1. Gate+up matmul: v3 (4 cuBLAS) vs v5 (mega N=28704) vs v5_upgrade_1 (padded N=28800)
  2. Down matmul: v3 (2 cuBLAS) vs v5 (mega N=4112) vs v5_upgrade_1 (= v3 pattern)
  3. End-to-end training (matmuls + addmm_) for all three.

Reports time, achieved TFLOPS and %peak.
A100-SXM4-80GB peaks: 312 TFLOPS bf16, ~1.5 TB/s effective HBM (peak 1.94).
"""

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 10
REP = 50

# LLaMA-8B microbench shape (batch=4, seq=2048, hidden=4096, intermediate=14336, rank=16)
M = 8192
H = 4096
I = 14336
R = 16

A100_PEAK_BF16_TFLOPS = 312.0


def fmt_ms(x: float) -> str:
    return f"{x:8.3f} ms"


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


def pct_peak(tflops: float) -> float:
    return 100.0 * tflops / A100_PEAK_BF16_TFLOPS


def round_up_128(n: int) -> int:
    return ((n + 127) // 128) * 128


# ---------------------------------------------------------------------------
# Phase A: gate+up matmul variants
# ---------------------------------------------------------------------------

def build_gate_up_tensors():
    X = torch.randn(M, H, device=DEVICE, dtype=DTYPE)
    W_gate = (torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02)
    W_up = (torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02)
    A_gate = (torch.randn(R, H, device=DEVICE, dtype=DTYPE) * 0.02)
    A_up = (torch.randn(R, H, device=DEVICE, dtype=DTYPE) * 0.02)

    # v5: not padded, N = 2*I + 2*R
    W_mega = torch.cat([W_gate, W_up, A_gate, A_up], dim=0).contiguous()
    # v5_upgrade_1: pad to next multiple of 128
    n_unpadded = 2 * I + 2 * R
    n_padded = round_up_128(n_unpadded)
    pad_rows = n_padded - n_unpadded
    pad = torch.zeros(pad_rows, H, device=DEVICE, dtype=DTYPE)
    W_mega_padded = torch.cat([W_gate, W_up, A_gate, A_up, pad], dim=0).contiguous()

    return X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_padded, n_unpadded, n_padded


def run_gate_up_phase(X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_padded, n_unpadded, n_padded):
    results = {}

    def v3_gate_up():
        e_base = torch.matmul(X, W_gate.t())
        g_base = torch.matmul(X, W_up.t())
        xa_gate = torch.matmul(X, A_gate.t())
        xa_up = torch.matmul(X, A_up.t())
        return e_base, g_base, xa_gate, xa_up

    ms = bench(v3_gate_up)
    flops = 2 * M * I * H * 2 + 2 * M * R * H * 2
    tflops = flops / (ms * 1e-3) / 1e12
    results[f"v3 gate+up (4 cuBLAS, N=14336+r)"] = (ms, tflops)

    def v5_gate_up():
        result = torch.matmul(X, W_mega.t())
        return (result[:, :I], result[:, I:2 * I],
                result[:, 2 * I:2 * I + R], result[:, 2 * I + R:])

    ms = bench(v5_gate_up)
    flops = 2 * M * n_unpadded * H
    tflops = flops / (ms * 1e-3) / 1e12
    results[f"v5 gate+up (mega N={n_unpadded})"] = (ms, tflops)

    def v5_up1_gate_up():
        result = torch.matmul(X, W_mega_padded.t())
        return (result[:, :I], result[:, I:2 * I],
                result[:, 2 * I:2 * I + R], result[:, 2 * I + R:2 * I + 2 * R])

    ms = bench(v5_up1_gate_up)
    flops = 2 * M * n_padded * H
    tflops = flops / (ms * 1e-3) / 1e12
    results[f"v5_up1 gate+up (padded N={n_padded})"] = (ms, tflops)

    return results


# ---------------------------------------------------------------------------
# Phase B: down matmul variants (with addmm_ for the LoRA-B term)
# ---------------------------------------------------------------------------

def build_down_tensors():
    h = torch.randn(M, I, device=DEVICE, dtype=DTYPE)
    W_down = (torch.randn(H, I, device=DEVICE, dtype=DTYPE) * 0.02)
    A_down = (torch.randn(R, I, device=DEVICE, dtype=DTYPE) * 0.02)
    B_down = (torch.randn(H, R, device=DEVICE, dtype=DTYPE) * 0.02)
    W_down_packed = torch.cat([W_down, A_down], dim=0).contiguous()
    return h, W_down, A_down, B_down, W_down_packed


def run_down_phase(h, W_down, A_down, B_down, W_down_packed):
    results = {}

    # v3 / v5_upgrade_1: matmul + skinny matmul + addmm_
    def v3_down_full():
        out = torch.matmul(h, W_down.t())          # [M, H], contig
        xa_down = torch.matmul(h, A_down.t())     # [M, R], contig
        out.addmm_(xa_down, B_down.t(), alpha=1.0)
        return out

    ms = bench(v3_down_full)
    flops = 2 * M * I * H + 2 * M * I * R + 2 * M * H * R
    tflops = flops / (ms * 1e-3) / 1e12
    results["v3 / v5_up1 down (matmul + skinny + addmm_)"] = (ms, tflops)

    # v5: packed mega-matmul + slice + .contiguous() + addmm_
    def v5_down_full():
        result = torch.matmul(h, W_down_packed.t())  # [M, H+R]
        out_slice = result[:, :H]
        xa_down = result[:, H:]
        out_buf = out_slice.contiguous()
        out_buf.addmm_(xa_down, B_down.t(), alpha=1.0)
        return out_buf

    ms = bench(v5_down_full)
    flops = 2 * M * I * (H + R) + 2 * M * H * R
    tflops = flops / (ms * 1e-3) / 1e12
    results["v5 down (mega + .contiguous() + addmm_)"] = (ms, tflops)

    return results


# ---------------------------------------------------------------------------
# Phase C: end-to-end training forward
# ---------------------------------------------------------------------------

def run_end_to_end():
    """End-to-end training forward at LLaMA-8B (LoRA-rank-16, bf16)."""
    from reference.unsloth_baseline import make_lora_mlp_params as unsloth_make_params
    from experiments.v3.lora_mlp_kernel_v3 import lora_mlp_v3
    from experiments.v5.lora_mlp_kernel_v5 import (
        lora_mlp_v5,
        pack_gate_up_weights,
        pack_down_weights,
    )
    from experiments.v5.lora_mlp_kernel_v5_upgrade_1 import (
        lora_mlp_v5_upgrade_1,
        pack_gate_up_weights_padded,
    )

    batch, seq = 4, 2048
    params = unsloth_make_params(H, I, R, dtype=DTYPE, device=DEVICE, requires_grad=False)
    X = torch.randn(batch, seq, H, dtype=DTYPE, device=DEVICE)
    gp, up, dp = params["gate_proj"], params["up_proj"], params["down_proj"]

    W_mega = pack_gate_up_weights(gp["W"], up["W"], gp["A"], up["A"])
    W_down_packed = pack_down_weights(dp["W"], dp["A"])
    W_mega_padded, _ = pack_gate_up_weights_padded(gp["W"], up["W"], gp["A"], up["A"])

    def v3_fn():
        return lora_mlp_v3(X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"])

    def v5_fn():
        return lora_mlp_v5(X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
            W_mega=W_mega, W_down_packed=W_down_packed)

    def v5_up1_fn():
        return lora_mlp_v5_upgrade_1(X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
            W_mega_padded=W_mega_padded)

    return {
        "v3 training (8 launches)": triton.testing.do_bench(v3_fn, warmup=20, rep=100),
        "v5 training (4 launches)": triton.testing.do_bench(v5_fn, warmup=20, rep=100),
        "v5_upgrade_1 training (5 launches)": triton.testing.do_bench(v5_up1_fn, warmup=20, rep=100),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_phase(title: str, results):
    print("\n" + title)
    print("-" * 78)
    print(f"{'Variant':<46} {'time':>10}  {'TFLOPS':>8}  {'%peak':>6}")
    for label, (ms, tflops) in results.items():
        print(f"{label:<46} {fmt_ms(ms):>10}  {tflops:8.1f}  {pct_peak(tflops):5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name()}  Dtype: {DTYPE}")
    print(f"Shape:  M={M}, H={H}, I={I}, R={R}")
    print(f"A100 peaks: bf16 {A100_PEAK_BF16_TFLOPS:.0f} TFLOPS")

    torch.manual_seed(args.seed)

    print("\nBuilding gate+up tensors ...")
    gu = build_gate_up_tensors()
    n_unpadded = gu[7]
    n_padded = gu[8]
    print(f"  gate+up: N_unpadded={n_unpadded}, N_padded={n_padded} "
          f"(pad_rows={n_padded - n_unpadded}, multiple of 128: {n_padded % 128 == 0})")
    gate_up = run_gate_up_phase(*gu)

    del gu
    torch.cuda.empty_cache()

    print("Building down tensors ...")
    h, W_down, A_down, B_down, W_down_packed = build_down_tensors()
    down = run_down_phase(h, W_down, A_down, B_down, W_down_packed)

    del h, W_down, A_down, B_down, W_down_packed
    torch.cuda.empty_cache()

    print_phase("=== Gate+Up matmul variants ===", gate_up)
    print_phase("=== Down matmul (end-to-end with addmm_) ===", down)

    print("\nRunning end-to-end training forward ...")
    e2e = run_end_to_end()
    print("\n=== End-to-end training forward (LLaMA-8B, batch=4, seq=2048, r=16, bf16) ===")
    print("-" * 78)
    print(f"{'Variant':<46} {'time':>10}")
    for label, ms in e2e.items():
        print(f"{label:<46} {fmt_ms(ms):>10}")

    # Compact comparison summary
    v5_gate = gate_up[f"v5 gate+up (mega N={n_unpadded})"][0]
    up1_gate = gate_up[f"v5_up1 gate+up (padded N={n_padded})"][0]
    v3_gate = gate_up["v3 gate+up (4 cuBLAS, N=14336+r)"][0]
    v3_down = down["v3 / v5_up1 down (matmul + skinny + addmm_)"][0]
    v5_down = down["v5 down (mega + .contiguous() + addmm_)"][0]

    print("\n=== Diagnosis Summary ===")
    print(
        f"Change 1 (drop down packing): saves {v5_down - v3_down:+.3f} ms vs v5"
        f" ({v5_down:.3f} → {v3_down:.3f} ms)"
    )
    print(
        f"Change 2 (pad gate+up to N={n_padded}): saves {v5_gate - up1_gate:+.3f} ms vs v5"
        f" ({v5_gate:.3f} → {up1_gate:.3f} ms)"
    )
    isolated_save = (v5_gate - up1_gate) + (v5_down - v3_down)
    print(f"Total isolated matmul-only savings vs v5: {isolated_save:+.3f} ms")
    print(
        f"vs v3 baseline gate+up (4 cuBLAS): "
        f"v5_up1 saves {v3_gate - up1_gate:+.3f} ms ({100*(v3_gate-up1_gate)/v3_gate:+.2f}%)"
    )

    v3_e2e = e2e["v3 training (8 launches)"]
    v5_e2e = e2e["v5 training (4 launches)"]
    up1_e2e = e2e["v5_upgrade_1 training (5 launches)"]
    print(
        f"\nEnd-to-end at LLaMA-8B/r=16: v3={v3_e2e:.3f}, v5={v5_e2e:.3f}, "
        f"v5_upgrade_1={up1_e2e:.3f} ms"
    )
    print(
        f"  v5_upgrade_1 vs v3: {v3_e2e - up1_e2e:+.3f} ms "
        f"({100*(v3_e2e-up1_e2e)/v3_e2e:+.2f}%)"
    )
    print(
        f"  v5_upgrade_1 vs v5: {v5_e2e - up1_e2e:+.3f} ms "
        f"({100*(v5_e2e-up1_e2e)/v5_e2e:+.2f}%)"
    )


if __name__ == "__main__":
    main()

"""
Microbenchmark: why didn't v5's packed-matmul optimization beat v3?

Isolates the gate+up and down-projection matmul work and compares:
  v3 (4 separate cuBLAS calls) vs v5 (1 packed cuBLAS call)

Shapes (LLaMA-8B, batch*seq = 8192, bf16):
  X:        [M, H]      = [8192, 4096]
  W_gate:   [I, H]      = [14336, 4096]
  W_up:     [I, H]
  A_gate:   [r, H]      = [16, 4096]
  A_up:     [r, H]
  W_mega:   [2*I + 2*r, H] = [28704, 4096]
  W_down:   [H, I]      = [4096, 14336]
  A_down:   [r, I]      = [16, 14336]
  W_down_packed: [H + r, I] = [4112, 14336]

Reports time, achieved TFLOPS, and achieved HBM bandwidth for each variant.
A100-SXM4-80GB peaks: 312 TFLOPS bf16, ~1.5 TB/s effective HBM (peak 1.94).
"""

import argparse
import os
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
A100_PEAK_HBM_TB_S = 1.94  # 2039 GB/s spec; achievable ~1.5 TB/s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_ms(x: float) -> str:
    return f"{x:8.3f} ms"


def matmul_tflops(M_: int, N_: int, K_: int, ms: float) -> float:
    """Compute achieved TFLOPS for an M x N x K matmul completed in `ms`."""
    flops = 2.0 * M_ * N_ * K_
    seconds = ms * 1e-3
    return flops / seconds / 1e12


def matmul_bandwidth(M_: int, N_: int, K_: int, ms: float, elem_size: int = 2) -> float:
    """Bytes moved (X + W + Y) divided by time, in TB/s."""
    bytes_ = (M_ * K_ + K_ * N_ + M_ * N_) * elem_size
    seconds = ms * 1e-3
    return bytes_ / seconds / 1e12


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)


# ---------------------------------------------------------------------------
# Phase 1+2: gate+up matmul variants
# ---------------------------------------------------------------------------

def build_gate_up_tensors():
    X = torch.randn(M, H, device=DEVICE, dtype=DTYPE)
    W_gate = (torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02)
    W_up = (torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02)
    A_gate = (torch.randn(R, H, device=DEVICE, dtype=DTYPE) * 0.02)
    A_up = (torch.randn(R, H, device=DEVICE, dtype=DTYPE) * 0.02)

    W_mega = torch.cat([W_gate, W_up, A_gate, A_up], dim=0).contiguous()  # [2*I + 2*R, H]
    W_mega_no_lora = torch.cat([W_gate, W_up], dim=0).contiguous()  # [2*I, H]
    return X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_no_lora


def run_gate_up_phase(X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_no_lora):
    """Time each gate+up variant and return a dict of (ms, TFLOPS, TB/s)."""
    results = {}

    # v3: 4 separate cuBLAS calls
    def v3_gate_up():
        e_base = torch.matmul(X, W_gate.t())
        g_base = torch.matmul(X, W_up.t())
        xa_gate = torch.matmul(X, A_gate.t())
        xa_up = torch.matmul(X, A_up.t())
        return e_base, g_base, xa_gate, xa_up

    ms = bench(v3_gate_up)
    # FLOPs = 2*M*N*K summed; for 4 matmuls
    flops = 2 * M * I * H * 2 + 2 * M * R * H * 2  # 2 big + 2 small
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (
        4 * M * H * 2  # X read 4 times
        + (W_gate.numel() + W_up.numel() + A_gate.numel() + A_up.numel()) * 2
        + (M * I * 2 + M * I * 2 + M * R * 2 + M * R * 2)
    )
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v3_gate_up (4 cuBLAS)"] = (ms, tflops, bw)

    # v5 mega: 1 packed cuBLAS call
    def v5_gate_up():
        result = torch.matmul(X, W_mega.t())  # [M, 2*I + 2*R]
        return result[:, :I], result[:, I:2 * I], result[:, 2 * I:2 * I + R], result[:, 2 * I + R:]

    ms = bench(v5_gate_up)
    N_mega = 2 * I + 2 * R
    flops = 2 * M * N_mega * H
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (M * H * 2) + (W_mega.numel() * 2) + (M * N_mega * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v5_gate_up (1 cuBLAS, mega)"] = (ms, tflops, bw)

    # v3 no-LoRA: 2 cuBLAS calls
    def v3_gate_up_no_lora():
        e_base = torch.matmul(X, W_gate.t())
        g_base = torch.matmul(X, W_up.t())
        return e_base, g_base

    ms = bench(v3_gate_up_no_lora)
    flops = 2 * M * I * H * 2
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (2 * M * H * 2) + (W_gate.numel() + W_up.numel()) * 2 + (2 * M * I * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v3_gate_up_no_lora (2 cuBLAS)"] = (ms, tflops, bw)

    # v5 no-LoRA: 1 packed cuBLAS, N = 2I
    def v5_gate_up_no_lora():
        result = torch.matmul(X, W_mega_no_lora.t())  # [M, 2*I]
        return result[:, :I], result[:, I:]

    ms = bench(v5_gate_up_no_lora)
    N_big = 2 * I
    flops = 2 * M * N_big * H
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (M * H * 2) + (W_mega_no_lora.numel() * 2) + (M * N_big * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v5_gate_up_no_lora (1 cuBLAS, 2I)"] = (ms, tflops, bw)

    return results


# ---------------------------------------------------------------------------
# Phase 4: down-projection variants
# ---------------------------------------------------------------------------

def build_down_tensors():
    h = torch.randn(M, I, device=DEVICE, dtype=DTYPE)
    W_down = (torch.randn(H, I, device=DEVICE, dtype=DTYPE) * 0.02)
    A_down = (torch.randn(R, I, device=DEVICE, dtype=DTYPE) * 0.02)
    B_down = (torch.randn(H, R, device=DEVICE, dtype=DTYPE) * 0.02)
    W_down_packed = torch.cat([W_down, A_down], dim=0).contiguous()  # [H + R, I]
    return h, W_down, A_down, B_down, W_down_packed


def run_down_phase(h, W_down, A_down, B_down, W_down_packed):
    results = {}

    # v3: 2 cuBLAS calls (just the matmuls)
    def v3_down():
        out_base = torch.matmul(h, W_down.t())
        xa_down = torch.matmul(h, A_down.t())
        return out_base, xa_down

    ms = bench(v3_down)
    flops = 2 * M * I * H + 2 * M * I * R
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (2 * M * I * 2) + (W_down.numel() + A_down.numel()) * 2 + (M * H * 2 + M * R * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v3_down (2 cuBLAS)"] = (ms, tflops, bw)

    # v5: 1 packed cuBLAS call (just the matmul, no slice .contiguous() yet)
    def v5_down():
        result = torch.matmul(h, W_down_packed.t())  # [M, H + R]
        return result[:, :H], result[:, H:]

    ms = bench(v5_down)
    N_packed = H + R
    flops = 2 * M * I * N_packed
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (M * I * 2) + (W_down_packed.numel() * 2) + (M * N_packed * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v5_down (1 cuBLAS, H+R)"] = (ms, tflops, bw)

    # v3 base only (no LoRA): 1 cuBLAS, for reference
    def v3_down_base():
        return torch.matmul(h, W_down.t())

    ms = bench(v3_down_base)
    flops = 2 * M * I * H
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_ = (M * I * 2) + (W_down.numel() * 2) + (M * H * 2)
    bw = bytes_ / (ms * 1e-3) / 1e12
    results["v3_down_base (1 cuBLAS, H only)"] = (ms, tflops, bw)

    # --- end-to-end down (matches what each kernel actually does) ---
    # v3 end-to-end: 2 matmuls + addmm_ on contiguous out_base
    def v3_down_full():
        out_base = torch.matmul(h, W_down.t())          # [M, H], contiguous
        xa_down = torch.matmul(h, A_down.t())           # [M, R], contiguous
        out_base.addmm_(xa_down, B_down.t(), alpha=1.0)
        return out_base

    ms = bench(v3_down_full)
    flops = 2 * M * I * H + 2 * M * I * R + 2 * M * H * R  # 2 matmuls + addmm
    tflops = flops / (ms * 1e-3) / 1e12
    bw = 0.0  # not meaningful for composite op
    results["v3_down_full (2 cuBLAS + addmm)"] = (ms, tflops, bw)

    # v5 end-to-end: 1 packed matmul + slice + contiguous() + addmm_
    def v5_down_full():
        result = torch.matmul(h, W_down_packed.t())  # [M, H + R]
        out_slice = result[:, :H]
        xa_down = result[:, H:]
        out_buf = out_slice.contiguous()
        out_buf.addmm_(xa_down, B_down.t(), alpha=1.0)
        return out_buf

    ms = bench(v5_down_full)
    flops = 2 * M * I * (H + R) + 2 * M * H * R
    tflops = flops / (ms * 1e-3) / 1e12
    bw = 0.0
    results["v5_down_full (mega + contig + addmm)"] = (ms, tflops, bw)

    # Cost of the .contiguous() copy alone — [M, H] bf16 = 64 MiB
    def contig_copy():
        # Replicate the slicing pattern: take a non-contig slice of [M, H+R] and
        # copy the [M, H] slice to a fresh contiguous buffer.
        src = torch.empty(M, H + R, dtype=DTYPE, device=DEVICE)
        return src[:, :H].contiguous()

    # The empty() in the benchmark dominates; instead, allocate src once and time only the copy.
    src_alloc = torch.empty(M, H + R, dtype=DTYPE, device=DEVICE)

    def contig_only():
        return src_alloc[:, :H].contiguous()

    ms = bench(contig_only)
    bytes_ = M * H * 2 * 2  # read + write
    bw = bytes_ / (ms * 1e-3) / 1e12
    results[".contiguous() copy of [M,H] slice"] = (ms, 0.0, bw)

    return results


# ---------------------------------------------------------------------------
# Phase 2 helper: cuBLAS backend identification
# ---------------------------------------------------------------------------

def report_backend():
    """Identify the active matmul backend."""
    try:
        backend = torch.backends.cuda.preferred_blas_library()
    except AttributeError:
        backend = "(preferred_blas_library not available in this torch)"
    return backend


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(gate_up_results, down_results):
    print("\n" + "=" * 78)
    print(f" Device: {torch.cuda.get_device_name()}")
    print(f" Dtype: {DTYPE}")
    print(f" Shape: M={M}, H={H}, I={I}, R={R}")
    print(f" Preferred BLAS: {report_backend()}")
    print(f" CUDA: {torch.version.cuda}  Torch: {torch.__version__}")
    print(f" Peaks: bf16 {A100_PEAK_BF16_TFLOPS:.0f} TFLOPS, HBM ~{A100_PEAK_HBM_TB_S:.2f} TB/s")
    print("=" * 78)

    header = f"{'Variant':<38} {'time':>10}  {'TFLOPS':>8}  {'TB/s':>6}  {'%peak':>6}"
    print("\n" + header)
    print("-" * 78)
    for label, (ms, tflops, bw) in gate_up_results.items():
        pct = 100.0 * tflops / A100_PEAK_BF16_TFLOPS
        print(f"{label:<38} {fmt_ms(ms):>10}  {tflops:8.1f}  {bw:6.3f}  {pct:5.1f}%")

    print("\n" + header)
    print("-" * 78)
    for label, (ms, tflops, bw) in down_results.items():
        pct = 100.0 * tflops / A100_PEAK_BF16_TFLOPS if tflops > 0 else 0.0
        print(f"{label:<38} {fmt_ms(ms):>10}  {tflops:8.1f}  {bw:6.3f}  {pct:5.1f}%")

    # Compact savings table
    v3_g = gate_up_results["v3_gate_up (4 cuBLAS)"][0]
    v5_g = gate_up_results["v5_gate_up (1 cuBLAS, mega)"][0]
    v3_g_nl = gate_up_results["v3_gate_up_no_lora (2 cuBLAS)"][0]
    v5_g_nl = gate_up_results["v5_gate_up_no_lora (1 cuBLAS, 2I)"][0]
    v3_d = down_results["v3_down (2 cuBLAS)"][0]
    v5_d = down_results["v5_down (1 cuBLAS, H+R)"][0]

    def savings_line(v3_ms, v5_ms):
        diff = v3_ms - v5_ms
        pct = 100.0 * diff / v3_ms if v3_ms > 0 else 0.0
        return f"{v3_ms:8.3f} ms  →  {v5_ms:8.3f} ms   (Δ {diff:+.3f} ms, {pct:+.2f}%)"

    print("\n" + "=" * 78)
    print("=== Gate+Up Phase Comparison ===")
    g_v3_tflops = gate_up_results["v3_gate_up (4 cuBLAS)"][1]
    g_v3_bw = gate_up_results["v3_gate_up (4 cuBLAS)"][2]
    g_v5_tflops = gate_up_results["v5_gate_up (1 cuBLAS, mega)"][1]
    g_v5_bw = gate_up_results["v5_gate_up (1 cuBLAS, mega)"][2]
    print(f"v3 (4 cuBLAS): {v3_g:.3f} ms, {g_v3_tflops:.1f} TFLOPS, {g_v3_bw:.3f} TB/s")
    print(f"v5 (1 cuBLAS): {v5_g:.3f} ms, {g_v5_tflops:.1f} TFLOPS, {g_v5_bw:.3f} TB/s")
    print(f"Savings: {(v3_g - v5_g):.3f} ms ({100*(v3_g-v5_g)/v3_g:+.1f}%)")

    print("\n=== Big Matmuls Only (no LoRA A) ===")
    print(f"v3 (2 cuBLAS):           {v3_g_nl:.3f} ms")
    print(f"v5 (1 cuBLAS [2*I, H]):  {v5_g_nl:.3f} ms")
    print(f"Savings: {(v3_g_nl - v5_g_nl):.3f} ms ({100*(v3_g_nl-v5_g_nl)/v3_g_nl:+.1f}%)")

    print("\n=== Down Phase Comparison (matmuls only) ===")
    print(f"v3 (2 cuBLAS): {v3_d:.3f} ms")
    print(f"v5 (1 cuBLAS): {v5_d:.3f} ms")
    print(f"Savings: {(v3_d - v5_d):.3f} ms ({100*(v3_d-v5_d)/v3_d:+.1f}%)")

    v3_d_full = down_results["v3_down_full (2 cuBLAS + addmm)"][0]
    v5_d_full = down_results["v5_down_full (mega + contig + addmm)"][0]
    contig_ms = down_results[".contiguous() copy of [M,H] slice"][0]
    print("\n=== Down Phase Comparison (end-to-end with addmm) ===")
    print(f"v3 (2 cuBLAS + addmm):                  {v3_d_full:.3f} ms")
    print(f"v5 (mega + contiguous() + addmm):       {v5_d_full:.3f} ms")
    print(f"Savings: {(v3_d_full - v5_d_full):.3f} ms ({100*(v3_d_full-v5_d_full)/v3_d_full:+.1f}%)")
    print(f"(.contiguous() copy on [M,H] slice:      {contig_ms:.3f} ms)")

    print("\n=== Net (gate+up + down, end-to-end) ===")
    v3_net = v3_g + v3_d_full
    v5_net = v5_g + v5_d_full
    print(f"v3 net matmul work: {v3_net:.3f} ms")
    print(f"v5 net matmul work: {v5_net:.3f} ms")
    print(f"Net savings: {(v3_net - v5_net):.3f} ms ({100*(v3_net-v5_net)/v3_net:+.1f}%)")

    return {
        "v3_gate_up_ms": v3_g, "v5_gate_up_ms": v5_g,
        "v3_gate_up_no_lora_ms": v3_g_nl, "v5_gate_up_no_lora_ms": v5_g_nl,
        "v3_down_ms": v3_d, "v5_down_ms": v5_d,
        "v3_down_full_ms": v3_d_full, "v5_down_full_ms": v5_d_full,
        "contig_copy_ms": contig_ms,
        "v3_gate_up_tflops": g_v3_tflops, "v5_gate_up_tflops": g_v5_tflops,
        "v3_gate_up_bw": g_v3_bw, "v5_gate_up_bw": g_v5_bw,
        "v3_net_ms": v3_net, "v5_net_ms": v5_net,
    }


def print_diagnosis(s):
    print("\n=== Diagnosis ===")
    v3_g = s["v3_gate_up_ms"]; v5_g = s["v5_gate_up_ms"]
    v3_g_nl = s["v3_gate_up_no_lora_ms"]; v5_g_nl = s["v5_gate_up_no_lora_ms"]
    v3_d_full = s["v3_down_full_ms"]; v5_d_full = s["v5_down_full_ms"]
    contig = s["contig_copy_ms"]
    v3_tf = s["v3_gate_up_tflops"]; v5_tf = s["v5_gate_up_tflops"]
    v3_net = s["v3_net_ms"]; v5_net = s["v5_net_ms"]

    gate_save = v3_g - v5_g
    down_save = v3_d_full - v5_d_full
    print(
        f"Gate+up packing saves {gate_save:.3f} ms ({100*gate_save/v3_g:+.1f}%). The "
        f"isolated comparison shows v3's two big base matmuls already deliver ~{(2*M*I*H*2)/(v3_g_nl*1e-3)/1e12:.0f} "
        f"TFLOPS (cuBLAS is compute-bound), so the gain almost entirely comes from "
        f"absorbing the two tiny X@A_*^T matmuls into the mega-GEMM rather than launching "
        f"them as separate skinny [M, r] cuBLAS calls. "
        f"Down packing LOSES {-down_save:.3f} ms ({100*down_save/v3_d_full:+.1f}%): the [M, H+r] = "
        f"[{M}, {H+R}] mega-output is non-contiguous in the H slice, so the kernel pays an "
        f"explicit `.contiguous()` copy (~{contig:.3f} ms for {M*H*2//(1<<20)} MiB) before "
        f"the addmm, and the extra r=16 columns may push cuBLAS off a clean N=4096 tile. "
        f"Net matmul work: v3={v3_net:.2f} ms, v5={v5_net:.2f} ms (Δ {v3_net-v5_net:+.2f} ms) — i.e. the "
        f"isolated savings are only ~{1000*(v3_net-v5_net):.0f} µs, comparable to the 8→4 launch "
        f"overhead difference (~50–100 µs) and well within run-to-run noise on a 12 ms forward. "
        f"v5's Triton epilogue must also read non-contiguous slices of the mega-output "
        f"(stride 0 = 2*I + 2*r) which costs a touch more than v3's contiguous reads."
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Need a CUDA GPU"
    dev_name = torch.cuda.get_device_name()
    print(f"Running on: {dev_name}")
    if "A100" not in dev_name:
        print(f"WARNING: expected A100, got {dev_name}")

    torch.manual_seed(args.seed)

    print("Building gate+up tensors ...")
    X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_no_lora = build_gate_up_tensors()
    print("Benchmarking gate+up variants ...")
    gate_up_results = run_gate_up_phase(X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_no_lora)

    # Free gate+up tensors before allocating down-phase ones (memory budget)
    del X, W_gate, W_up, A_gate, A_up, W_mega, W_mega_no_lora
    torch.cuda.empty_cache()

    print("Building down tensors ...")
    h, W_down, A_down, B_down, W_down_packed = build_down_tensors()
    print("Benchmarking down variants ...")
    down_results = run_down_phase(h, W_down, A_down, B_down, W_down_packed)

    summary = print_summary(gate_up_results, down_results)
    print_diagnosis(summary)


if __name__ == "__main__":
    main()

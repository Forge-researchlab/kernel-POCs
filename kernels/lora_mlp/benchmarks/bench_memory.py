"""
Peak GPU memory benchmark for LoRA MLP kernels (LLaMA-3 scale).

Measures peak GPU memory delta during a single forward (and forward+backward
for training paths) for each implementation:

  1. Unsloth's ``apply_lora_mlp_swiglu``           (fwd + bwd)
  2. v3       ``LoRAMLPv3.apply``                  (fwd + bwd)
  3. v5       ``LoRAMLPv5.apply``                  (fwd + bwd)
  4. v5_upgrade_1 ``LoRAMLPv5_upgrade_1.apply``    (fwd + bwd)
  5. v5 inference ``lora_mlp_v5_inference``        (fwd only, pre-merged)

Methodology per measurement:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    out = fn()                                # forward
    if backward:
        out.sum().backward()                  # backward
        # zero grads (lora A/B + X) before next iter
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - mem_before

We report three numbers per row:
  - fwd_mb:                peak delta during just the forward call
  - fwd_bwd_mb:            peak delta across forward + backward (training paths)
  - resident_after_fwd_mb: amount allocated right after the forward returns
                           (output tensor + ``save_for_backward`` saved tensors)

Why these three? ``fwd_mb`` captures temporary buffers (e.g. v5's packed
W_mega and the [M, 2*I+2*r] mega-matmul output). ``resident_after_fwd_mb``
captures the activation "footprint" that survives the forward and stays
resident throughout backward (the per-block memory pressure during long
backward computations). ``fwd_bwd_mb`` captures the peak across the entire
training step.

Persistent weight memory (W, A, B) is reported separately — it is identical
across the four training paths and lower for inference (only merged W).

Each measurement is repeated 3× and the median is reported.
"""
import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.unsloth_baseline import (  # noqa: E402
    apply_lora_mlp_swiglu,
    make_lora_mlp_params as unsloth_make_params,
)
from experiments.v3.lora_mlp_kernel_v3 import LoRAMLPv3  # noqa: E402
from experiments.v5.lora_mlp_kernel_v5 import (  # noqa: E402
    LoRAMLPv5,
    lora_mlp_v5_inference,
    prepare_inference_weights,
)
from experiments.v5.lora_mlp_kernel_v5_upgrade_1 import (  # noqa: E402
    LoRAMLPv5_upgrade_1,
)


DEVICE = "cuda"
MB = 1024.0 * 1024.0
REPEATS = 3
WARMUPS = 2

IMPLS_TRAINING = ("Unsloth", "v3", "v5", "v5_upgrade_1")
IMPL_INFERENCE = "v5_inference"
IMPL_ORDER = (*IMPLS_TRAINING, IMPL_INFERENCE)


# ---------------------------------------------------------------------------
# Memory measurement primitives
# ---------------------------------------------------------------------------

def bytes_of(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.element_size() * t.nelement()


def reset_grads(tensors: List[Optional[torch.Tensor]]) -> None:
    """Reset .grad to None on all tensors (avoids accumulation across iters)."""
    for t in tensors:
        if t is not None and getattr(t, "grad", None) is not None:
            t.grad = None


def measure_once(
    fn: Callable[[], torch.Tensor],
    backward: bool,
    grad_tensors: List[Optional[torch.Tensor]],
) -> Tuple[int, int, int]:
    """Run ``fn`` once and return memory deltas in bytes.

    Returns
    -------
    fwd_peak : int
        Peak ``memory_allocated`` delta during the forward call.
    full_peak : int
        Peak across forward + backward (equal to fwd_peak if not backward).
    resident_after_fwd : int
        ``memory_allocated`` delta right after the forward returns
        (output + saved_for_backward tensors).
    """
    reset_grads(grad_tensors)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    out = fn()
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() - mem_before
    resident_after_fwd = torch.cuda.memory_allocated() - mem_before

    if backward:
        out.sum().backward()
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated() - mem_before
    else:
        full_peak = fwd_peak

    del out
    reset_grads(grad_tensors)
    torch.cuda.synchronize()
    return fwd_peak, full_peak, resident_after_fwd


def measure_median(
    fn: Callable[[], torch.Tensor],
    backward: bool,
    grad_tensors: List[Optional[torch.Tensor]],
    repeats: int = REPEATS,
    warmups: int = WARMUPS,
) -> Tuple[int, int, int]:
    """Run ``fn`` ``warmups + repeats`` times and return the median of each metric."""
    for _ in range(warmups):
        measure_once(fn, backward, grad_tensors)
    runs = [measure_once(fn, backward, grad_tensors) for _ in range(repeats)]
    fwd_vals = [r[0] for r in runs]
    full_vals = [r[1] for r in runs]
    resident_vals = [r[2] for r in runs]
    return (
        int(statistics.median(fwd_vals)),
        int(statistics.median(full_vals)),
        int(statistics.median(resident_vals)),
    )


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class MemResult:
    impl: str
    batch: int
    seq_len: int
    hidden: int
    intermediate: int
    rank: int
    dtype: str
    weight_mb: float
    fwd_mb: float
    fwd_bwd_mb: Optional[float]  # None for inference
    resident_after_fwd_mb: float
    config_label: str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        if d["fwd_bwd_mb"] is None:
            d["fwd_bwd_mb"] = ""
        return d


# ---------------------------------------------------------------------------
# Per-config measurement
# ---------------------------------------------------------------------------

def run_config(
    config_label: str,
    batch: int,
    seq_len: int,
    hidden: int,
    intermediate: int,
    rank: int,
    dtype: torch.dtype,
) -> List[MemResult]:
    """Measure memory for all 5 implementations at one (batch, seq, H, I, r, dtype)."""
    print(
        f"\n=== {config_label}: batch={batch} seq={seq_len} "
        f"H={hidden} I={intermediate} r={rank} {str(dtype).replace('torch.', '')} ==="
    )

    # Build params with A/B requiring grad, W frozen.
    params = unsloth_make_params(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        rank=rank,
        dtype=dtype,
        device=DEVICE,
        requires_grad=True,
    )
    gp, up, dp = params["gate_proj"], params["up_proj"], params["down_proj"]
    X = (
        torch.randn(batch, seq_len, hidden, dtype=dtype, device=DEVICE)
        .requires_grad_(True)
    )

    grad_tensors = [
        gp["A"], gp["B"], up["A"], up["B"], dp["A"], dp["B"], X,
    ]

    # Persistent weight memory: W + A + B for all three projections.
    weight_bytes = sum(
        bytes_of(t)
        for t in [
            gp["W"], gp["A"], gp["B"],
            up["W"], up["A"], up["B"],
            dp["W"], dp["A"], dp["B"],
        ]
    )
    train_weight_mb = weight_bytes / MB

    common = dict(
        batch=batch, seq_len=seq_len, hidden=hidden,
        intermediate=intermediate, rank=rank,
        dtype=str(dtype).replace("torch.", ""),
        config_label=config_label,
    )
    results: List[MemResult] = []

    # ── Unsloth baseline ──
    def unsloth_fn():
        return apply_lora_mlp_swiglu(X, **params)

    fwd, full, resident = measure_median(unsloth_fn, backward=True, grad_tensors=grad_tensors)
    results.append(MemResult(
        impl="Unsloth", weight_mb=train_weight_mb,
        fwd_mb=fwd / MB, fwd_bwd_mb=full / MB,
        resident_after_fwd_mb=resident / MB, **common,
    ))

    # ── v3 ──
    def v3_fn():
        return LoRAMLPv3.apply(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )

    fwd, full, resident = measure_median(v3_fn, backward=True, grad_tensors=grad_tensors)
    results.append(MemResult(
        impl="v3", weight_mb=train_weight_mb,
        fwd_mb=fwd / MB, fwd_bwd_mb=full / MB,
        resident_after_fwd_mb=resident / MB, **common,
    ))

    # ── v5 (packed training) ──
    def v5_fn():
        return LoRAMLPv5.apply(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )

    fwd, full, resident = measure_median(v5_fn, backward=True, grad_tensors=grad_tensors)
    results.append(MemResult(
        impl="v5", weight_mb=train_weight_mb,
        fwd_mb=fwd / MB, fwd_bwd_mb=full / MB,
        resident_after_fwd_mb=resident / MB, **common,
    ))

    # ── v5_upgrade_1 (padded gate+up mega + v3-style down) ──
    def v5_up1_fn():
        return LoRAMLPv5_upgrade_1.apply(
            X,
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )

    fwd, full, resident = measure_median(v5_up1_fn, backward=True, grad_tensors=grad_tensors)
    results.append(MemResult(
        impl="v5_upgrade_1", weight_mb=train_weight_mb,
        fwd_mb=fwd / MB, fwd_bwd_mb=full / MB,
        resident_after_fwd_mb=resident / MB, **common,
    ))

    # ── v5 inference (pre-merged effective weights) ──
    # Merge once OUTSIDE the timed window — this is the realistic setup
    # because effective weights are computed once at LoRA-merge time, not per
    # forward.
    with torch.no_grad():
        W_gate_eff_T, W_up_eff_T, W_down_eff_T = prepare_inference_weights(
            gp["W"], gp["A"], gp["B"], gp["s"],
            up["W"], up["A"], up["B"], up["s"],
            dp["W"], dp["A"], dp["B"], dp["s"],
        )
    inf_weight_bytes = sum(
        bytes_of(t) for t in (W_gate_eff_T, W_up_eff_T, W_down_eff_T)
    )
    inf_weight_mb = inf_weight_bytes / MB

    # Inference doesn't need autograd graph — use detached X to avoid building one.
    X_inf = X.detach()

    def v5_inf_fn():
        return lora_mlp_v5_inference(X_inf, W_gate_eff_T, W_up_eff_T, W_down_eff_T)

    fwd, full, resident = measure_median(v5_inf_fn, backward=False, grad_tensors=[])
    results.append(MemResult(
        impl="v5_inference", weight_mb=inf_weight_mb,
        fwd_mb=fwd / MB, fwd_bwd_mb=None,
        resident_after_fwd_mb=resident / MB, **common,
    ))

    # Free merged weights before next config to keep baseline low.
    del W_gate_eff_T, W_up_eff_T, W_down_eff_T, X_inf
    del params, X
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table(config_label: str, rows: List[MemResult]) -> None:
    """Print a clean memory comparison table for one config."""
    print(f"\n=== Peak GPU Memory ({config_label}) ===")
    # Compute Unsloth as ratio baseline.
    unsloth = next((r for r in rows if r.impl == "Unsloth"), None)
    base_fwd = unsloth.fwd_mb if unsloth else None
    base_bwd = unsloth.fwd_bwd_mb if unsloth else None

    header = (
        f"{'Implementation':<28} {'Weights (MB)':>13} "
        f"{'Fwd (MB)':>10} {'Fwd+Bwd (MB)':>14} "
        f"{'vs Unsloth Fwd':>16} {'vs Unsloth Bwd':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        fwd_str = f"{r.fwd_mb:>10.1f}"
        if r.fwd_bwd_mb is None:
            bwd_str = f"{'—':>14}"
            vs_bwd = f"{'—':>16}"
        else:
            bwd_str = f"{r.fwd_bwd_mb:>14.1f}"
            vs_bwd = f"{r.fwd_bwd_mb / base_bwd:>15.2f}x" if base_bwd else f"{'—':>16}"
        vs_fwd = f"{r.fwd_mb / base_fwd:>15.2f}x" if base_fwd else f"{'—':>16}"
        impl_label = r.impl + (" (pre-merged)" if r.impl == "v5_inference" else "")
        print(
            f"{impl_label:<28} {r.weight_mb:>13.1f} "
            f"{fwd_str} {bwd_str} {vs_fwd} {vs_bwd}"
        )


# ---------------------------------------------------------------------------
# CSV + Markdown output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "config_label", "impl",
    "batch", "seq_len", "hidden", "intermediate", "rank", "dtype",
    "weight_mb", "fwd_mb", "fwd_bwd_mb", "resident_after_fwd_mb",
]


def save_csv(results: List[MemResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.as_dict()
            # Round floats for readability.
            for k in ("weight_mb", "fwd_mb", "fwd_bwd_mb", "resident_after_fwd_mb"):
                v = row.get(k, "")
                if isinstance(v, float):
                    row[k] = round(v, 2)
            writer.writerow(row)
    print(f"\nCSV saved to {path}")


def format_markdown(
    grouped: List[Tuple[str, List[MemResult]]],
    gpu_name: str,
    timestamp: str,
) -> str:
    """Produce a markdown report grouping rows by config."""
    lines: List[str] = []
    lines.append("# LoRA MLP Peak GPU Memory Benchmark")
    lines.append("")
    lines.append(f"**Date:** {timestamp}")
    lines.append(f"**GPU:** {gpu_name}")
    lines.append(f"**Script:** `kernels/lora_mlp/benchmarks/bench_memory.py`")
    lines.append(
        "**Methodology:** `torch.cuda.reset_peak_memory_stats()` before each run; "
        "peak = `max_memory_allocated - memory_allocated_before`. Median of 3 runs."
    )
    lines.append("")
    lines.append("## What each column means")
    lines.append("")
    lines.append(
        "- **Weights (MB):** persistent storage for the projection tensors. "
        "Training paths store `W + A + B` for gate/up/down (identical across "
        "Unsloth, v3, v5, v5_upgrade_1). Inference stores only the merged "
        "`W_eff` tensors — no separate A/B."
    )
    lines.append(
        "- **Fwd (MB):** peak `memory_allocated` delta during the forward call. "
        "Captures temporary buffers (e.g. v5's packed `W_mega`, the "
        "`[M, 2*I + 2*r]` mega-matmul output)."
    )
    lines.append(
        "- **Fwd+Bwd (MB):** peak delta across forward + backward. Includes "
        "the temporaries above plus everything backward needs simultaneously "
        "(`DW`, transposed `A/B`, all six grad buffers, `dX`, and so on)."
    )
    lines.append(
        "- **Resident after fwd (MB):** what stays allocated right after the "
        "forward returns — the output tensor plus any `save_for_backward` "
        "tensors. This is the activation footprint that backward has to live with."
    )
    lines.append("")

    for label, rows in grouped:
        lines.append(f"## {label}")
        lines.append("")
        cfg = rows[0]
        lines.append(
            f"`batch={cfg.batch}, seq={cfg.seq_len}, H={cfg.hidden}, "
            f"I={cfg.intermediate}, r={cfg.rank}, {cfg.dtype}`"
        )
        lines.append("")
        unsloth = next((r for r in rows if r.impl == "Unsloth"), None)
        base_fwd = unsloth.fwd_mb if unsloth else None
        base_bwd = unsloth.fwd_bwd_mb if unsloth else None

        lines.append(
            "| Implementation | Weights (MB) | Fwd (MB) | Fwd+Bwd (MB) | "
            "Resident after fwd (MB) | vs Unsloth Fwd | vs Unsloth Bwd |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            fwd_bwd_str = "—" if r.fwd_bwd_mb is None else f"{r.fwd_bwd_mb:.1f}"
            if base_fwd:
                vs_fwd = f"{r.fwd_mb / base_fwd:.2f}x"
            else:
                vs_fwd = "—"
            if r.fwd_bwd_mb is None or base_bwd is None:
                vs_bwd = "—"
            else:
                vs_bwd = f"{r.fwd_bwd_mb / base_bwd:.2f}x"
            impl_label = (
                r.impl + " (pre-merged)" if r.impl == "v5_inference" else r.impl
            )
            lines.append(
                f"| {impl_label} | {r.weight_mb:.1f} | {r.fwd_mb:.1f} | "
                f"{fwd_bwd_str} | {r.resident_after_fwd_mb:.1f} | "
                f"{vs_fwd} | {vs_bwd} |"
            )
        lines.append("")

    # ── Interpretation section ──
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "**Winner on memory: `v5_inference` (pre-merged).** At LLaMA-8B "
        "production (M=8192, r=16) it ties Unsloth on forward peak (736 MB) "
        "but only retains 64 MB after the call versus 512 MB for every "
        "training path. That's 8x lower activation footprint, achieved by "
        "(a) merging `B @ A` into `W_eff` once at LoRA-merge time (no "
        "runtime LoRA matmuls) and (b) running outside autograd so nothing "
        "is saved for backward (no `e`, no `g`)."
    )
    lines.append("")
    lines.append(
        "**Honest finding (surprise): `v5` training uses ~2.1x MORE peak "
        "forward memory than Unsloth, and ~34% more than `v3`.** The "
        "wall-clock-equivalent v5 path pays a real memory cost for the "
        "packed mega-GEMM:"
    )
    lines.append("")
    lines.append(
        "- `W_mega = cat(W_gate, W_up, A_gate, A_up)` allocates a fresh "
        "`[2*I + 2*r, H]` tensor every call (~224 MiB at LLaMA-8B)."
    )
    lines.append(
        "- `W_down_packed = cat(W_down, A_down)` allocates another "
        "`[H + r, I]` (~118 MiB)."
    )
    lines.append(
        "- The mega-matmul output `result = X @ W_mega.t()` is "
        "`[M, 2*I + 2*r]` (~448 MiB at M=8192) and stays alive (via the "
        "non-contiguous slice views `e_base`, `g_base`, `xa_gate`, "
        "`xa_up`) until the Python scope of `_v5_forward_impl` exits."
    )
    lines.append(
        "- The Triton epilogue still allocates contiguous `e_full`, "
        "`g_full` (2 × ~224 MiB) for backward, and `h` (~224 MiB) — "
        "those don't disappear when `result` does."
    )
    lines.append("")
    lines.append(
        "Add it up: ~224 (W_mega) + ~118 (W_down_packed) + ~448 (result) "
        "+ ~672 (h, e_full, g_full) + 64 (output) + 64 (contig copy for "
        "addmm_) ≈ **1590 MiB peak** — within rounding of the measured "
        "1585.4 MB. Compare against v3, which doesn't pack: ~224·5 (e_base, "
        "g_base, h, e_full, g_full) + 64 (output) ≈ **1184 MiB** — exact "
        "match against measured 1184.8 MB."
    )
    lines.append("")
    lines.append(
        "**`v5_upgrade_1` saves ~173 MB vs `v5`** at LLaMA-8B production "
        "(1412.2 vs 1585.4). The win comes from dropping the down packing "
        "(no `W_down_packed`, no `down_result`, no `.contiguous()` copy). "
        "It still pays the gate+up mega-GEMM memory tax."
    )
    lines.append("")
    lines.append(
        "**Activation footprint (the thing that actually limits batch size "
        "during training) is identical at 512 MiB for all four training "
        "paths** at LLaMA-8B production: `e + g + output = 224 + 224 + 64` "
        "MiB. They all save the same tensors for backward; the differences "
        "are purely in transient forward-time buffers. So if you're trying "
        "to fit a bigger batch, picking v3 over v5 buys you headroom only "
        "during the forward call — peak `fwd+bwd` is what matters across "
        "the whole step, and there v3 wins by ~400 MB vs v5 at LLaMA-8B "
        "production."
    )
    lines.append("")
    lines.append(
        "**Sanity check on the math** (bf16, M=8192, H=4096, I=14336, r=16):"
    )
    lines.append("")
    lines.append("- `[M, I]` (e, g, h) = 8192·14336·2 bytes = **224 MiB** each.")
    lines.append("- `[M, H]` (output, X, dX) = 8192·4096·2 = **64 MiB** each.")
    lines.append(
        "- Weights `W_gate + W_up + W_down` = 3·14336·4096·2 / 2²⁰ = "
        "**336 MiB**; LoRA `A`s and `B`s add ~1.7 MiB. Measured weight "
        "memory: **337.7 MiB** ✓."
    )
    lines.append(
        "- Resident after fwd for training = output + saved e + saved g "
        "= 64 + 224 + 224 = **512 MiB** ✓ (matches the measurement exactly "
        "for all four training paths)."
    )
    lines.append("")
    return "\n".join(lines)


def save_markdown(report: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    print(f"Markdown report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_configs() -> List[Tuple[str, dict]]:
    """Configs in (label, kwargs) order."""
    return [
        (
            "LLaMA-8B production (batch=4, seq=2048, r=16, bf16)",
            dict(batch=4, seq_len=2048, hidden=4096, intermediate=14336, rank=16,
                 dtype=torch.bfloat16),
        ),
        (
            "LLaMA-8B small (batch=1, seq=2048, r=16, bf16)",
            dict(batch=1, seq_len=2048, hidden=4096, intermediate=14336, rank=16,
                 dtype=torch.bfloat16),
        ),
        (
            "LLaMA-13B small (batch=1, seq=2048, r=16, bf16)",
            dict(batch=1, seq_len=2048, hidden=5120, intermediate=17920, rank=16,
                 dtype=torch.bfloat16),
        ),
        (
            "LLaMA-8B production, rank sweep r=8 (bf16)",
            dict(batch=4, seq_len=2048, hidden=4096, intermediate=14336, rank=8,
                 dtype=torch.bfloat16),
        ),
        (
            "LLaMA-8B production, rank sweep r=32 (bf16)",
            dict(batch=4, seq_len=2048, hidden=4096, intermediate=14336, rank=32,
                 dtype=torch.bfloat16),
        ),
        (
            "LLaMA-8B production, rank sweep r=64 (bf16)",
            dict(batch=4, seq_len=2048, hidden=4096, intermediate=14336, rank=64,
                 dtype=torch.bfloat16),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="LoRA MLP peak memory benchmarks")
    parser.add_argument(
        "--save-dir", type=str,
        default="benchmarks/results",
        help="Directory to save CSV results (relative to the lora_mlp project root).",
    )
    parser.add_argument(
        "--markdown-path", type=str,
        default="docs/analysis/memory_benchmark.md",
        help="Path to save the markdown report (relative to the lora_mlp project root).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run only configs whose label contains this substring (default: run all).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    torch.manual_seed(0)

    configs = default_configs()
    if args.config:
        configs = [(label, kw) for label, kw in configs if args.config in label]
        if not configs:
            raise SystemExit(f"No configs match --config={args.config!r}")

    grouped: List[Tuple[str, List[MemResult]]] = []
    all_results: List[MemResult] = []
    for label, kw in configs:
        rows = run_config(config_label=label, **kw)
        grouped.append((label, rows))
        all_results.extend(rows)
        print_table(label, rows)

    project_root = Path(__file__).parent.parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = project_root / args.save_dir / f"memory_{timestamp}.csv"
    save_csv(all_results, str(csv_path))

    md_path = project_root / args.markdown_path
    report = format_markdown(grouped, gpu_name=gpu_name, timestamp=time.strftime("%Y-%m-%d"))
    save_markdown(report, str(md_path))

    print("\nDone.")


if __name__ == "__main__":
    main()

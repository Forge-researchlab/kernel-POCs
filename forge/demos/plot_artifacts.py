"""Generate the 4 shareable artifacts from the training JSONL log.

Reads:
    artifacts/lora_demo/train_log.jsonl    (one JSON entry per training step)
    artifacts/lora_demo/run_meta.json      (final run metadata)

Produces (all under artifacts/lora_demo/):
    loss_curve.png     — training loss vs step + 20-step moving average
    vram.png           — peak VRAM per rank over steps + peak bar chart
    step_time.png      — per-step wall time + cumulative + average
    summary.png        — one-figure dashboard combining all three for sharing
    artifacts.md       — markdown that pulls everything together
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARTIFACTS_DIR = Path("/workspace/kernel-POCs/artifacts/lora_demo")
LOG_PATH = ARTIFACTS_DIR / "train_log.jsonl"
META_PATH = ARTIFACTS_DIR / "run_meta.json"


def load_log():
    rows = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta():
    with META_PATH.open() as f:
        return json.load(f)


def _moving_avg(x, w):
    if len(x) < w:
        return np.array(x, dtype=float)
    c = np.convolve(np.array(x, dtype=float), np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, c[0])
    return np.concatenate([pad, c])


def plot_loss_curve(rows, meta, out_path):
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    ma = _moving_avg(loss, 20)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(steps, loss, color="#9ca3af", linewidth=1.0, label="step loss", zorder=2)
    ax.plot(steps, ma, color="#2563eb", linewidth=2.2, label="20-step moving avg", zorder=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(f"LoRA fine-tune loss — {meta['model']}\n"
                 f"r={meta['lora_r']} α={meta['lora_alpha']} bs={meta['effective_batch']} "
                 f"lr={meta['lr']} · forge.patch{meta['forge_kernels']} · FSDP2 world={meta['world_size']}",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    # Drop-arrow annotation showing the descent
    ax.annotate(f"start: {loss[0]:.3f}",
                xy=(steps[0], loss[0]), xytext=(steps[0] + 20, loss[0] + 0.4),
                arrowprops=dict(arrowstyle="->", color="#475569"),
                fontsize=9, color="#475569")
    ax.annotate(f"final: {loss[-1]:.3f}  (-{(loss[0]-loss[-1])/loss[0]*100:.0f}%)",
                xy=(steps[-1], loss[-1]),
                xytext=(steps[-1] - 60, loss[-1] - 0.6),
                arrowprops=dict(arrowstyle="->", color="#16a34a"),
                fontsize=9, color="#16a34a")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_vram(rows, meta, out_path):
    steps = np.array([r["step"] for r in rows])
    n_ranks = len(rows[0]["peak_vram_gb_per_rank"])
    per_rank = np.array([r["peak_vram_gb_per_rank"] for r in rows])  # [steps, ranks]

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 4.5), dpi=140,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for r in range(n_ranks):
        ax_line.plot(steps, per_rank[:, r], color=colors[r % len(colors)],
                     linewidth=1.6, label=f"rank {r}")
    ax_line.set_xlabel("Step")
    ax_line.set_ylabel("Peak VRAM (GB, high-water mark)")
    ax_line.set_title(f"Peak VRAM per rank over training\n"
                      f"{meta['device_name']} · FSDP2 world={meta['world_size']} · "
                      f"forge.patch enabled", fontsize=11)
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc="lower right")

    final_peaks = meta["final_peak_vram_gb_per_rank"]
    bar_colors = [colors[r % len(colors)] for r in range(n_ranks)]
    bars = ax_bar.bar([f"rank {r}" for r in range(n_ranks)], final_peaks,
                      color=bar_colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, final_peaks):
        ax_bar.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} GB",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Final peak (GB)")
    ax_bar.set_title("Final per-rank peak", fontsize=11)
    ax_bar.set_ylim(0, max(final_peaks) * 1.18)
    ax_bar.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_step_time(rows, meta, out_path):
    steps = np.array([r["step"] for r in rows])
    times = np.array([r["step_time_s"] for r in rows]) * 1000  # ms
    cumulative = np.cumsum(times) / 1000  # seconds
    avg_after_warmup = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))

    fig, (ax_step, ax_cum) = plt.subplots(2, 1, figsize=(8, 6), dpi=140, sharex=True)

    ax_step.plot(steps, times, color="#2563eb", linewidth=1.2)
    ax_step.axhline(avg_after_warmup, color="#dc2626", linestyle="--",
                    linewidth=1.4, label=f"avg (post-warmup): {avg_after_warmup:.0f} ms")
    ax_step.set_ylabel("Step time (ms)")
    ax_step.set_title(f"Per-step training time\n"
                      f"world={meta['world_size']} · effective batch {meta['effective_batch']} · "
                      f"forge.patch enabled", fontsize=11)
    ax_step.grid(True, alpha=0.25)
    ax_step.legend(loc="upper right")

    ax_cum.plot(steps, cumulative, color="#16a34a", linewidth=1.8)
    ax_cum.fill_between(steps, 0, cumulative, color="#16a34a", alpha=0.12)
    ax_cum.set_xlabel("Step")
    ax_cum.set_ylabel("Cumulative training time (s)")
    ax_cum.set_title(f"Cumulative wall time (total: {cumulative[-1]:.1f}s "
                     f"for {len(steps)} steps)", fontsize=10)
    ax_cum.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(rows, meta, out_path):
    """Single 2x2 figure combining all three charts — best for one-screenshot
    sharing with the team."""
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    times_ms = np.array([r["step_time_s"] for r in rows]) * 1000
    per_rank = np.array([r["peak_vram_gb_per_rank"] for r in rows])
    n_ranks = per_rank.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    fig.suptitle(
        f"Forge LoRA fine-tune — {meta['model']} · "
        f"forge.patch{meta['forge_kernels']} · FSDP2 world={meta['world_size']}\n"
        f"{meta['steps']} steps · LoRA r={meta['lora_r']} α={meta['lora_alpha']} · "
        f"effective batch {meta['effective_batch']} · lr={meta['lr']}",
        fontsize=12, y=1.0,
    )

    # (0,0) Loss curve
    ax = axes[0, 0]
    ax.plot(steps, loss, color="#9ca3af", linewidth=1.0, alpha=0.7, zorder=2)
    ax.plot(steps, _moving_avg(loss, 20), color="#2563eb", linewidth=2.2, zorder=3,
            label="20-step MA")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"Training loss — first {loss[0]:.3f} → final {loss[-1]:.3f}  "
                 f"(-{(loss[0]-loss[-1])/loss[0]*100:.0f}%)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # (0,1) Per-rank VRAM over time
    ax = axes[0, 1]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for r in range(n_ranks):
        ax.plot(steps, per_rank[:, r], color=colors[r % len(colors)],
                linewidth=1.6, label=f"rank {r}")
    ax.set_xlabel("Step"); ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title(f"Peak VRAM per rank (high-water mark)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    # (1,0) Step time
    ax = axes[1, 0]
    ax.plot(steps, times_ms, color="#2563eb", linewidth=1.2)
    avg_post = float(np.mean(times_ms[5:])) if len(times_ms) > 5 else float(np.mean(times_ms))
    ax.axhline(avg_post, color="#dc2626", linestyle="--", linewidth=1.4,
               label=f"avg (post-warmup): {avg_post:.0f} ms")
    ax.set_xlabel("Step"); ax.set_ylabel("Step time (ms)")
    ax.set_title("Per-step wall time", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # (1,1) Per-rank peak final
    ax = axes[1, 1]
    final_peaks = meta["final_peak_vram_gb_per_rank"]
    bar_colors = [colors[r % len(colors)] for r in range(n_ranks)]
    bars = ax.bar([f"rank {r}" for r in range(n_ranks)], final_peaks,
                  color=bar_colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, final_peaks):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Peak VRAM (GB)")
    cum_total = float(np.sum(times_ms) / 1000)
    ax.set_title(f"Final per-rank peak  ·  total wall time: {cum_total:.1f}s",
                 fontsize=11)
    ax.set_ylim(0, max(final_peaks) * 1.18)
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_artifacts_md(meta, out_path):
    final_peaks = meta["final_peak_vram_gb_per_rank"]
    body = f"""# Forge LoRA fine-tune — artifacts

**Model:** `{meta['model']}`
**Setup:** FSDP2 world={meta['world_size']} · {meta['device_name']} · torch {meta['torch']}
**Kernels patched by `forge.patch`:** `{meta['forge_kernels']}` → counts `{meta['patched_counts']}`
**LoRA:** r={meta['lora_r']}, α={meta['lora_alpha']}, targets q/k/v/gate/up/down
**Training:** {meta['steps']} steps, lr={meta['lr']}, effective batch {meta['effective_batch']} (= {meta['batch_per_micro']} × {meta['micro_batches']} micro-batches), max_seq={meta['max_seq_len']}

## Numbers

| Metric | Value |
|---|---|
| First-step loss | {meta['first_loss']:.4f} |
| Final-step loss | {meta['final_loss']:.4f} |
| Minimum loss | {meta['min_loss']:.4f} |
| Loss reduction | {(meta['first_loss']-meta['final_loss'])/meta['first_loss']*100:.1f}% |
| Final peak VRAM per rank | {', '.join(f'{p:.2f} GB' for p in final_peaks)} |

## Charts

| | |
|---|---|
| ![loss](loss_curve.png) | ![vram](vram.png) |
| ![step time](step_time.png) | (one-pager: `summary.png`) |

See `summary.png` for a single-figure dashboard combining all of the above.
See `inference_samples.md` for held-out generations before vs after fine-tune.
"""
    out_path.write_text(body)


def main():
    rows = load_log()
    meta = load_meta()
    print(f"Loaded {len(rows)} step rows and meta with keys: {list(meta.keys())}")

    plot_loss_curve(rows, meta, ARTIFACTS_DIR / "loss_curve.png")
    plot_vram(rows, meta, ARTIFACTS_DIR / "vram.png")
    plot_step_time(rows, meta, ARTIFACTS_DIR / "step_time.png")
    plot_summary_dashboard(rows, meta, ARTIFACTS_DIR / "summary.png")
    write_artifacts_md(meta, ARTIFACTS_DIR / "artifacts.md")

    print(f"Wrote 4 PNGs + artifacts.md to {ARTIFACTS_DIR}")
    for p in ARTIFACTS_DIR.glob("*.png"):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

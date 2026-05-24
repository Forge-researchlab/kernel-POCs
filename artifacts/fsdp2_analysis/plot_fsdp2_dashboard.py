"""Generate FSDP2 smoke-test analysis dashboards for Qwen 3 and Gemma 2.

Reads the JSONL metrics + JSON summary files produced by the instrumented
verify_fsdp2_lora_{qwen,gemma}.py tests and produces:
  - Per-model 2x2 dashboards (loss curve, VRAM, step time, verdict table)
  - Combined comparison dashboard
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(model_name):
    jsonl_path = os.path.join(HERE, f"{model_name}_fsdp2_metrics.jsonl")
    summary_path = os.path.join(HERE, f"{model_name}_fsdp2_summary.json")

    with open(summary_path) as f:
        summary = json.load(f)

    ref_metrics = []
    fsdp_rank0 = []
    fsdp_rank1 = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if row["run"] == "ref_single_gpu":
                ref_metrics.append(row)
            elif row.get("rank", 0) == 0:
                fsdp_rank0.append(row)
            else:
                fsdp_rank1.append(row)

    return summary, ref_metrics, fsdp_rank0, fsdp_rank1


def make_model_dashboard(model_name, display_name, summary, ref_metrics, fsdp_rank0, fsdp_rank1):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    title = (
        f"Forge FSDP2 smoke test — {display_name} · "
        f"forge.patch{summary['kernels_patched']} · "
        f"FSDP2 world={summary['world_size']}\n"
        f"{summary['train_steps']} steps · LoRA r={summary['lora_r']} "
        f"α={summary['lora_alpha']} · "
        f"effective batch {summary['micro_batches']} · lr={summary['lr']}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                           left=0.08, right=0.95, top=0.90, bottom=0.06)

    # --- 1) Training loss: ref vs FSDP2 ---
    ax1 = fig.add_subplot(gs[0, 0])
    steps = [m["step"] for m in ref_metrics]
    ref_loss = [m["loss"] for m in ref_metrics]
    fsdp_loss = [m["loss"] for m in fsdp_rank0]

    ax1.plot(steps, ref_loss, "o-", color="#2196F3", linewidth=2,
             markersize=7, label="Single-GPU ref (no FSDP2)")
    ax1.plot(steps, fsdp_loss, "s--", color="#FF5722", linewidth=2,
             markersize=7, label="FSDP2 + forge.patch")

    first_loss = ref_loss[0]
    last_ref = ref_loss[-1]
    last_fsdp = fsdp_loss[-1]
    pct_drop = (1 - last_ref / first_loss) * 100
    ax1.set_title(
        f"Training loss — {first_loss:.3f} → {last_ref:.3f} "
        f"(-{pct_drop:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    rel_diffs = [abs(a - b) / (abs(b) + 1e-12) for a, b in zip(fsdp_loss, ref_loss)]
    max_rel = max(rel_diffs)
    ax1.annotate(
        f"max rel diff: {max_rel:.2e}",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="#666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
    )

    # --- 2) Peak VRAM per rank ---
    ax2 = fig.add_subplot(gs[0, 1])
    fsdp_vram_r0 = [m["peak_vram_gb"] * 1024 for m in fsdp_rank0]  # MB
    fsdp_vram_r1 = [m["peak_vram_gb"] * 1024 for m in fsdp_rank1]  # MB
    ref_vram = [m["peak_vram_gb"] * 1024 for m in ref_metrics]  # MB

    ax2.plot(steps, ref_vram, "o-", color="#9E9E9E", linewidth=2,
             markersize=6, label="Single-GPU ref")
    ax2.plot(steps, fsdp_vram_r0, "s-", color="#2196F3", linewidth=2,
             markersize=6, label="FSDP2 rank 0")
    ax2.plot(steps, fsdp_vram_r1, "^-", color="#FF5722", linewidth=2,
             markersize=6, label="FSDP2 rank 1")

    ax2.set_title("Peak VRAM per rank (high-water mark)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Peak VRAM (MB)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    final_r0 = summary["peaks_gb"][0] * 1024
    final_r1 = summary["peaks_gb"][1] * 1024
    ax2.annotate(
        f"rank0: {final_r0:.1f} MB  rank1: {final_r1:.1f} MB",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="#666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
    )

    # --- 3) Per-step wall time ---
    ax3 = fig.add_subplot(gs[1, 0])
    ref_times = [m["step_time_ms"] for m in ref_metrics]
    fsdp_times = [m["step_time_ms"] for m in fsdp_rank0]

    ax3.plot(steps, ref_times, "o-", color="#9E9E9E", linewidth=2,
             markersize=6, label="Single-GPU ref")
    ax3.plot(steps, fsdp_times, "s-", color="#2196F3", linewidth=2,
             markersize=6, label="FSDP2 rank 0")

    avg_ref = np.mean(ref_times[1:]) if len(ref_times) > 1 else ref_times[0]
    avg_fsdp = np.mean(fsdp_times[1:]) if len(fsdp_times) > 1 else fsdp_times[0]
    ax3.axhline(avg_ref, color="#9E9E9E", linestyle="--", alpha=0.6)
    ax3.axhline(avg_fsdp, color="#2196F3", linestyle="--", alpha=0.6)

    ax3.set_title("Per-step wall time", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Step time (ms)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax3.annotate(
        f"avg (post-warmup): ref={avg_ref:.0f}ms  fsdp={avg_fsdp:.0f}ms",
        xy=(0.02, 0.92), xycoords="axes fraction",
        fontsize=9, color="#666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
    )

    # --- 4) Verdict table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    checks = summary["checks"]
    check_names = {
        "C2_forward": "C2  Forward parity",
        "C3_q_lora_A": "C3  Grad q_lora_A",
        "C3_q_lora_B": "C3  Grad q_lora_B",
        "C3_gate_lora_A": "C3  Grad gate_lora_A",
        "C3_gate_lora_B": "C3  Grad gate_lora_B",
        "C5_generation": "C5  Generation parity",
        "C6_training": "C6  Training convergence",
    }

    table_data = []
    cell_colors = []
    for key in sorted(checks.keys()):
        label = check_names.get(key, key)
        passed = checks[key]
        status = "PASS" if passed else "FAIL"
        color = "#C8E6C9" if passed else "#FFCDD2"
        table_data.append([label, status])
        cell_colors.append([color, color])

    overall = summary["overall"]
    table_data.append(["OVERALL", "PASS" if overall else "FAIL"])
    cell_colors.append(
        ["#4CAF50" if overall else "#F44336"] * 2
    )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Check", "Result"],
        cellColours=cell_colors,
        colColours=["#E3F2FD", "#E3F2FD"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if row == len(table_data):
            cell.set_text_props(fontweight="bold", color="white")

    ax4.set_title(
        f"FSDP2 Verification Checks — {display_name}",
        fontsize=11, fontweight="bold", pad=15,
    )

    out_path = os.path.join(HERE, f"{model_name}_fsdp2_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def make_comparison_dashboard(qwen_data, gemma_data):
    q_summary, q_ref, q_fsdp0, q_fsdp1 = qwen_data
    g_summary, g_ref, g_fsdp0, g_fsdp1 = gemma_data

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Forge FSDP2 Verification — Qwen 3 vs Gemma 2 Comparison\n"
        f"2× A100-80GB · torch {q_summary['torch_version']} · "
        f"forge.patch['lora_qkv','lora_mlp'] · FSDP2 world=2",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35,
                           left=0.06, right=0.96, top=0.88, bottom=0.08)

    # --- Loss comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    q_steps = [m["step"] for m in q_ref]
    ax1.plot(q_steps, [m["loss"] for m in q_ref], "o-", color="#2196F3",
             linewidth=2, markersize=6, label="Qwen ref")
    ax1.plot(q_steps, [m["loss"] for m in q_fsdp0], "o--", color="#2196F3",
             linewidth=1.5, markersize=5, alpha=0.7, label="Qwen FSDP2")
    ax1.plot(q_steps, [m["loss"] for m in g_ref], "s-", color="#FF5722",
             linewidth=2, markersize=6, label="Gemma ref")
    ax1.plot(q_steps, [m["loss"] for m in g_fsdp0], "s--", color="#FF5722",
             linewidth=1.5, markersize=5, alpha=0.7, label="Gemma FSDP2")
    ax1.set_title("Training Loss: Ref vs FSDP2", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Loss diff (parity) ---
    ax2 = fig.add_subplot(gs[0, 1])
    q_diffs = [abs(a["loss"] - b["loss"]) for a, b in zip(q_fsdp0, q_ref)]
    g_diffs = [abs(a["loss"] - b["loss"]) for a, b in zip(g_fsdp0, g_ref)]
    ax2.plot(q_steps, q_diffs, "o-", color="#2196F3", linewidth=2,
             markersize=6, label="Qwen |ref - fsdp|")
    ax2.plot(q_steps, g_diffs, "s-", color="#FF5722", linewidth=2,
             markersize=6, label="Gemma |ref - fsdp|")
    ax2.set_title("Loss Parity (absolute diff)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("|Δloss|")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    # --- Step time comparison ---
    ax3 = fig.add_subplot(gs[0, 2])
    q_times_ref = [m["step_time_ms"] for m in q_ref]
    q_times_fsdp = [m["step_time_ms"] for m in q_fsdp0]
    g_times_ref = [m["step_time_ms"] for m in g_ref]
    g_times_fsdp = [m["step_time_ms"] for m in g_fsdp0]

    ax3.plot(q_steps, q_times_ref, "o-", color="#2196F3", linewidth=2,
             markersize=6, label="Qwen ref")
    ax3.plot(q_steps, q_times_fsdp, "o--", color="#2196F3", linewidth=1.5,
             markersize=5, alpha=0.7, label="Qwen FSDP2")
    ax3.plot(q_steps, g_times_ref, "s-", color="#FF5722", linewidth=2,
             markersize=6, label="Gemma ref")
    ax3.plot(q_steps, g_times_fsdp, "s--", color="#FF5722", linewidth=1.5,
             markersize=5, alpha=0.7, label="Gemma FSDP2")
    ax3.set_title("Per-step Wall Time", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Time (ms)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- VRAM bar chart ---
    ax4 = fig.add_subplot(gs[1, 0])
    labels = ["Qwen\nref", "Qwen\nrank0", "Qwen\nrank1",
              "Gemma\nref", "Gemma\nrank0", "Gemma\nrank1"]
    q_ref_vram = max(m["peak_vram_gb"] for m in q_ref) * 1024
    q_r0_vram = q_summary["peaks_gb"][0] * 1024
    q_r1_vram = q_summary["peaks_gb"][1] * 1024
    g_ref_vram = max(m["peak_vram_gb"] for m in g_ref) * 1024
    g_r0_vram = g_summary["peaks_gb"][0] * 1024
    g_r1_vram = g_summary["peaks_gb"][1] * 1024
    vrams = [q_ref_vram, q_r0_vram, q_r1_vram, g_ref_vram, g_r0_vram, g_r1_vram]
    colors = ["#90CAF9", "#2196F3", "#1565C0", "#FFAB91", "#FF5722", "#BF360C"]

    bars = ax4.bar(labels, vrams, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vrams):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_title("Peak VRAM (MB)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("MB")
    ax4.grid(True, alpha=0.3, axis="y")

    # --- Combined verdict table ---
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")

    check_labels = {
        "C2_forward": "C2 Forward parity",
        "C3_q_lora_A": "C3 Grad q_lora_A",
        "C3_q_lora_B": "C3 Grad q_lora_B",
        "C3_gate_lora_A": "C3 Grad gate_lora_A",
        "C3_gate_lora_B": "C3 Grad gate_lora_B",
        "C5_generation": "C5 Generation match",
        "C6_training": "C6 Training convergence",
    }

    table_data = []
    cell_colors = []
    for key in sorted(check_labels.keys()):
        q_pass = q_summary["checks"].get(key, False)
        g_pass = g_summary["checks"].get(key, False)
        q_str = "PASS" if q_pass else "FAIL"
        g_str = "PASS" if g_pass else "FAIL"
        q_col = "#C8E6C9" if q_pass else "#FFCDD2"
        g_col = "#C8E6C9" if g_pass else "#FFCDD2"
        table_data.append([check_labels[key], q_str, g_str])
        cell_colors.append(["#FAFAFA", q_col, g_col])

    q_ov = q_summary["overall"]
    g_ov = g_summary["overall"]
    table_data.append([
        "OVERALL",
        "PASS" if q_ov else "FAIL",
        "PASS" if g_ov else "FAIL",
    ])
    cell_colors.append([
        "#E0E0E0",
        "#4CAF50" if q_ov else "#F44336",
        "#4CAF50" if g_ov else "#F44336",
    ])

    table = ax5.table(
        cellText=table_data,
        colLabels=["Check", "Qwen 3", "Gemma 2"],
        cellColours=cell_colors,
        colColours=["#E3F2FD", "#BBDEFB", "#FFCCBC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if row == len(table_data) and col > 0:
            cell.set_text_props(fontweight="bold", color="white")

    ax5.set_title(
        "FSDP2 Verification Checks — Side by Side",
        fontsize=11, fontweight="bold", pad=15,
    )

    out_path = os.path.join(HERE, "comparison_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Loading metrics...")
    qwen_data = load_data("qwen")
    gemma_data = load_data("gemma")

    print("Generating per-model dashboards...")
    make_model_dashboard("qwen", "Qwen 3 (SiLU)", *qwen_data)
    make_model_dashboard("gemma", "Gemma 2 (GeGLU)", *gemma_data)

    print("Generating comparison dashboard...")
    make_comparison_dashboard(qwen_data, gemma_data)

    print("\nAll dashboards generated in:", HERE)

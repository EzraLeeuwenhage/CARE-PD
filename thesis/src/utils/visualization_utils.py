from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sequence_length_distributions(dataset, out_dir: Union[str, Path]):
    """
    Extracts sequence/chunk lengths from a Dataset instance and generates:
      1. Overview plot: Global distribution + Overlaid per-class density curves.
      2. 2x2 breakdown: Independent histogram & KDE for each severity class.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lengths = []
    severities = []

    if hasattr(dataset, "window_indices") and hasattr(dataset, "pose_data"):
        for idx, (key, start_idx) in enumerate(dataset.window_indices):
            raw_len = dataset.pose_data[key].shape[0]
            max_len = getattr(dataset, "max_len", getattr(dataset, "window_size", raw_len))
            eff_len = min(max_len, raw_len - start_idx)
            sev = dataset.get_severity(idx)
            lengths.append(eff_len)
            severities.append(sev)
    else:
        for idx in range(len(dataset)):
            item = dataset[idx]
            lengths.append(int(item["seq_len"]))
            severities.append(int(item["severity"]))

    df = pd.DataFrame({"length": lengths, "severity": severities})
    total_seqs = len(df)

    palette = {0: "#5b8def", 1: "#f28e2b", 2: "#59a14f", 3: "#e15759"}
    classes = sorted(df["severity"].unique())

    # Overview of sequence length distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    mean_len = df["length"].mean()
    median_len = df["length"].median()

    # Left: Global histogram & KDE
    sns.histplot(
        df["length"],
        kde=True,
        ax=axes[0],
        color="#7ea4e0",
        edgecolor="black",
        line_kws={"linewidth": 2, "color": "#2c64b5"},
    )
    axes[0].axvline(mean_len, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_len:.1f}")
    axes[0].axvline(median_len, color="green", linestyle=":", linewidth=2, label=f"Median: {median_len:.1f}")
    axes[0].set_title(f"Distribution of Sequence Lengths (N={total_seqs} seqs)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sequence Length (Frames)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Right: Class KDE Curves
    for sev in classes:
        sub = df[df["severity"] == sev]
        sns.kdeplot(
            sub["length"],
            ax=axes[1],
            label=f"Class {sev} (N={len(sub)})",
            color=palette.get(sev, "gray"),
            fill=True,
            alpha=0.3,
            linewidth=2,
        )

    axes[1].set_title("Sequence Lengths by Severity Class", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Sequence Length (Frames)", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].legend(title="Class_Label", loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig1_path = out_path / "01_sequence_length_overview.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved sequence length overview plot to: {fig1_path}")

    # Breakdown per severity class
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, sev in enumerate(classes):
        if idx >= 4:
            break
        ax = axes[idx]
        sub = df[df["severity"] == sev]
        n_samples = len(sub)
        c_mean = sub["length"].mean()
        c_median = sub["length"].median()
        color = palette.get(sev, "gray")

        sns.histplot(
            sub["length"],
            kde=True,
            ax=ax,
            color=color,
            alpha=0.4,
            edgecolor="black",
            line_kws={"linewidth": 2, "color": color},
        )
        ax.axvline(c_mean, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {c_mean:.1f}")
        ax.axvline(c_median, color="green", linestyle=":", linewidth=1.8, label=f"Median: {c_median:.1f}")

        ax.set_title(f"Class {sev} (N={n_samples})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sequence Length (Frames)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Hide unused subplots if fewer than 4 classes exist
    for idx in range(len(classes), 4):
        axes[idx].set_visible(False)

    plt.suptitle("Sequence Length Distributions by Severity Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig2_path = out_path / "01_sequence_lengths_by_class.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2x2 per-class breakdown plot to: {fig2_path}")
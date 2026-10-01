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

def plot_stationary_sequence_eval_metrics(stationary_records, distances_df, output_dir, show_outliers=False):
    """Renders a single 2x2 comparison dashboard of the 4 stationary sequence metrics (GT vs. Gen).
    
    If Generated data has 0 sequences, Ground Truth boxes are rendered with empty
    adjacent slots reserved for synthetic data.
    """
    df = pd.DataFrame(stationary_records)
    if df.empty:
        print("[Stationary Plot] No stationary records available to plot.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_color(score):
        if score < 0.10: return '#85e085'  # Green
        if score < 0.20: return '#ffe680'  # Yellow
        if score < 0.40: return '#ffb366'  # Orange
        return '#ff6666'                  # Red

    palette = {"Ground Truth": "cornflowerblue", "Generated": "salmon"}
    cls_order = ["Overall"] + sorted([c for c in df["Severity Class"].unique() if c != "Overall"])

    metrics_config = [
        {
            "key": "mean_global_walking_speed",
            "name": "Global Walking Speed",
            "title": "Global Walking Speed (H36M)",
            "ylabel": "Speed (m/s)",
            "clinical_note": "Lower = Slower Progression"
        },
        {
            "key": "mean_cadence",
            "name": "Mean Cadence",
            "title": "Mean Cadence (H36M)",
            "ylabel": "Cadence (steps/min)",
            "clinical_note": "Stepping Rhythm / Shuffling"
        },
        {
            "key": "ankle_bradykinesia",
            "name": "Ankle Bradykinesia",
            "title": "Worse Ankle Bradykinesia (SMPL)",
            "ylabel": "Physical Amplitude AUC (deg/s·Hz)",
            "clinical_note": "Lower = Reduced Drive"
        },
        {
            "key": "wrist_smoothness_auc",
            "name": "Wrist Smoothness AUC",
            "title": "Worse Wrist Jitter Energy (SMPL)",
            "ylabel": "Relative Jitter Energy (%)",
            "clinical_note": "Higher = Pathological Tremor"
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes_flat = axes.flatten()

    for idx, cfg in enumerate(metrics_config):
        ax = axes_flat[idx]
        sub_df = df[df["Metric"] == cfg["key"]]

        # hue_order ensures Ground Truth sits on the left even if Generated has 0 samples
        sns.boxplot(
            data=sub_df, 
            x="Severity Class", 
            y="Value", 
            hue="Source",
            order=cls_order, 
            hue_order=["Ground Truth", "Generated"],
            palette=palette, 
            showfliers=show_outliers,
            width=0.5, 
            ax=ax
        )

        ax.set_title(f"{cfg['title']}\n[{cfg['clinical_note']}]", fontsize=12, fontweight='bold', pad=25)
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.set_xlabel("Clinical Severity Class", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Plot distance badges only when distances_df is available
        if distances_df is not None and not distances_df.empty and not sub_df[sub_df["Source"] == "Generated"].empty:
            y_min_auto, y_max_auto = ax.get_ylim()
            y_range = max(y_max_auto - y_min_auto, 1e-5)
            ax.set_ylim(y_min_auto - (y_range * 0.05), y_max_auto + (y_range * 0.32))

            x_ticks = [l.get_text() for l in ax.get_xticklabels()]
            for x_idx, label_text in enumerate(x_ticks):
                match = distances_df[(distances_df['Severity'] == label_text) & (distances_df['Metric'] == cfg["name"])]
                if not match.empty:
                    ks = match.iloc[0]['KS_Stat']
                    w = match.iloc[0]['Wasserstein']
                    ax.text(
                        x_idx, y_max_auto + (y_range * 0.10), f"K: {ks:.2f}\nW: {w:.2f}",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                        bbox=dict(facecolor=get_color(ks), edgecolor='black', boxstyle='round,pad=0.25', alpha=0.9)
                    )

        ax.legend(title="Data Source", loc="upper right")

    fig.suptitle(
        "Stationary Gait Pathology Profile (Displacement < 0.5m)\n"
        "[Ground Truth vs. Generated Kinematic Comparison]",
        fontsize=15, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    save_path = out_dir / "stationary_pathology_overview_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Stationary Overview Dashboard to: {save_path}")

def plot_physical_realism_tracking(val_epochs, floating_gt, floating_gen, foot_disp_gt, foot_disp_gen, out_dir):
    """Generates a Matplotlib tracked history plot over all epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(val_epochs, floating_gt, color='cornflowerblue', linestyle='--', linewidth=2.5, label='Ground Truth Baseline')
    axes[0].plot(val_epochs, floating_gen, color='salmon', linestyle='-', linewidth=2.5, label='Generated Model')
    axes[0].set_title("Floating over Epochs", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontweight='bold')
    axes[0].set_ylabel("Mean Floating (m)", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(val_epochs, foot_disp_gt, color='cornflowerblue', linestyle='--', linewidth=2.5, label='Ground Truth Baseline')
    axes[1].plot(val_epochs, foot_disp_gen, color='salmon', linestyle='-', linewidth=2.5, label='Generated Model')
    axes[1].set_title("Foot Displacement (Skating) over Epochs", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontweight='bold')
    axes[1].set_ylabel("Mean Displacement (m)", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    tracking_path = Path(out_dir) / "physical_realism_tracking.png"
    plt.savefig(tracking_path, dpi=300)
    plt.close()
    
    return tracking_path
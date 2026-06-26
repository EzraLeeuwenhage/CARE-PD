import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataframe(data):
    """
    Converts the nested dictionary of arrays from H36MEvaluator into a 
    long-form Pandas DataFrame, making Seaborn plotting extremely easy.
    """
    records = []
    # Ensure 'overall' is plotted first, followed by sorted class IDs (0, 1, 2, 3)
    keys = ["overall"] + sorted([k for k in data.keys() if k != "overall"])

    for cls_key in keys:
        metrics = data[cls_key]
        n_seq = len(metrics["sequence_length"])
        cls_name = "Overall" if cls_key == "overall" else f"Class {cls_key}"
        label = f"{cls_name}\n(N={n_seq})"

        # Convert parallel arrays into row-based records
        for i in range(n_seq):
            rec = {"Class_Label": label, "Class_ID": cls_key}
            for k, v in metrics.items():
                rec[k] = v[i]
            records.append(rec)

    return pd.DataFrame(records)

# ---------------------------------------------------------
# PHYSICAL REALISM BOX PLOTS
# ---------------------------------------------------------
def plot_physical_realism_grouped(df, output_dir):
    """Creates grouped boxplots comparing Overall and Per-Class distributions."""
    sns.set_theme(style="whitegrid")

    # Smoothness (Jerk)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Class_Label", y="mean_jerk", color="salmon", width=0.5)
    plt.title("Jerk (Mean Rate of Acceleration Change)", fontsize=14, pad=10, fontweight='bold')
    plt.ylabel("Jerk (m/s³)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "01a_phys_jerk.png", dpi=300)
    plt.close()

    # Floating & Skating
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=df, x="Class_Label", y="floating", ax=axes[0], color="mediumaquamarine", width=0.5)
    axes[0].set_title("Floating (Lowest Foot Y-Coord at Strike)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Vertical Position (meters)")
    axes[0].set_xlabel("")

    sns.boxplot(data=df, x="Class_Label", y="foot_skating", ax=axes[1], color="turquoise", width=0.5)
    axes[1].set_title("Foot Skating (Stance Foot Velocity at Strike)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "01b_phys_environment.png", dpi=300)
    plt.close()

    # Structural Constancy
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Class_Label", y="mean_bone_length_variance", color="plum", width=0.5)
    plt.title("Structural Constancy (Mean Bone Length Variance)", fontsize=14, pad=10, fontweight='bold')
    plt.ylabel("Variance (m²)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "01c_phys_bones.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# PD FEATURES BAR CHARTS
# ---------------------------------------------------------
def plot_pd_feature_bars(df, output_dir):
    """
    Plots summary stats (bars with error whiskers) for specified clinical features.
    Safely ignores NaNs using np.nanmean and np.nanstd.
    """
    features = [
        {"key": "mean_step_length", "title": "Mean Step Length", "ylabel": "Length (m)"},
        {"key": "variance_step_length", "title": "Variance in Step Length", "ylabel": "Variance (m²)"},
        {"key": "mean_walking_speed", "title": "Walking Speed", "ylabel": "Speed (m/s)"},
        {"key": "mean_vertical_foot_lifting", "title": "Vertical Foot Lifting", "ylabel": "Height (m)"},
        {"key": "mean_emos", "title": "Estimated Margin of Stability (eMoS)", "ylabel": "eMoS (m)"},
        {"key": "variance_emos", "title": "Variance in eMoS", "ylabel": "Variance (m²)"}
    ]

    grid_shape = (2, 3)
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(6 * grid_shape[1], 5 * grid_shape[0]))
    axes_flat = axes.flatten()

    labels = df["Class_Label"].unique() 

    for idx, feat_info in enumerate(features):
        ax = axes_flat[idx]
        key = feat_info["key"]

        means = []
        stds = []
        
        # calculate stats per class
        for label in labels:
            # Extract just this class and metric, dropping all NaNs immediately
            subset = df[df["Class_Label"] == label][key].dropna()
            
            # If the entire subset was NaNs, avoid crashing script
            if subset.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(subset))
                stds.append(np.std(subset))

        x_pos = np.arange(len(labels))
        colors = sns.color_palette("muted", n_colors=len(labels))
        
        ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(feat_info["title"], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(feat_info["ylabel"])
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Clinical PD Features by Severity Class", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "02_pd_features_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# SEQUENCE LENGTH DISTRIBUTION
# ---------------------------------------------------------
def plot_sequence_length_distribution(df, output_dir):
    """Plots a histogram of the raw sequence lengths across the dataset."""
    
    # We only need the "Overall" subset to avoid double-counting sequences
    raw_lengths = df[df["Class_ID"] == "overall"]["sequence_length"].dropna()
    
    if len(raw_lengths) == 0:
        print("No sequence length data found to plot.")
        return

    plt.figure(figsize=(8, 5))
    
    # Plot histogram with a density curve
    sns.histplot(raw_lengths, bins=20, kde=True, color="cornflowerblue", edgecolor="black")
    
    mean_len = np.mean(raw_lengths)
    median_len = np.median(raw_lengths)
    N = len(raw_lengths)
    
    plt.title(f"Distribution of Sequence Lengths (N={N} seqs)", fontsize=14, fontweight='bold', pad=10)
    plt.xlabel("Sequence Length (Frames)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # Add vertical lines for Mean and Median context
    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='green', linestyle='dotted', linewidth=2, label=f'Median: {median_len:.1f}')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    out_path = output_dir / "00_sequence_length_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    pkl_path = Path("thesis\data\processed\evaluation\gen_h36m_distributions.pkl")
    output_dir = Path("thesis/visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pkl_path.exists():
        print(f"Could not find ground truth file at {pkl_path}")
        print("Please run evaluate.py first to generate this file.")
    else:
        print("Loading cached distributions...")
        data = load_data(pkl_path)
        df = prepare_dataframe(data)

        plot_sequence_length_distribution(df, output_dir)
        plot_physical_realism_grouped(df, output_dir)
        plot_pd_feature_bars(df, output_dir)

        print(f"\nSuccessfully generated all visuals in: {output_dir}")
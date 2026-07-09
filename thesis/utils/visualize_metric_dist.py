import pickle
import argparse
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
    df_clean = df.dropna(subset=["sequence_length"])
    
    if df_clean.empty:
        print("No sequence length data found to plot.")
        return

    # overall distribution
    overall_df = df_clean[df_clean["Class_ID"] == "overall"]
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=overall_df, x="sequence_length", bins=20, kde=True, color="cornflowerblue", edgecolor="black")
    
    mean_len = overall_df["sequence_length"].mean()
    median_len = overall_df["sequence_length"].median()
    N = len(overall_df)
    
    plt.title(f"Overall Sequence Lengths (N={N})", fontsize=14, fontweight='bold', pad=10)
    plt.xlabel("Sequence Length (Frames)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='green', linestyle='dotted', linewidth=2, label=f'Median: {median_len:.1f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "00a_sequence_length_overall.png", dpi=300)
    plt.close()

    # separate distributions per class
    class_df = df_clean[df_clean["Class_ID"] != "overall"]
    classes = sorted(class_df["Class_ID"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    
    colors = sns.color_palette("muted", n_colors=len(classes))

    for idx, cls in enumerate(classes):
        ax = axes_flat[idx]
        subset = class_df[class_df["Class_ID"] == cls]
        
        sns.histplot(data=subset, x="sequence_length", bins=15, kde=True, ax=ax, color=colors[idx], edgecolor="black")
        
        c_mean = subset["sequence_length"].mean()
        c_median = subset["sequence_length"].median()
        c_n = len(subset)
        
        ax.set_title(f"Class {cls} (N={c_n})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sequence Length (Frames)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        
        ax.axvline(c_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {c_mean:.1f}')
        ax.axvline(c_median, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {c_median:.1f}')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("Sequence Length Distributions by Severity Class", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "00b_sequence_length_per_class.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="thesis/data/processed/evaluation/gt_h36m_distributions.pkl")
    parser.add_argument("-o", "--output", type=str, default="thesis/visualizations")
    args = parser.parse_args()

    pkl_path = Path(args.input)
    output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pkl_path.exists():
        print(f"Could not find file at {pkl_path}")
        print("Please run evaluate.py first to generate this file.")
    else:
        print("Loading cached distributions...")
        data = load_data(pkl_path)
        df = prepare_dataframe(data)

        plot_sequence_length_distribution(df, output_dir)
        plot_physical_realism_grouped(df, output_dir)
        plot_pd_feature_bars(df, output_dir)

        print(f"\nSuccessfully generated all visuals in: {output_dir}")
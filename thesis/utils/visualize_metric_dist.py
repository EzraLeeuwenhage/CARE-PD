import pickle
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from thesis.src.evaluate_distributions import DistributionComparator

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataframe(data):
    """Converts the nested dictionary of arrays into dataframe for seaborn"""
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
# PHYSICAL REALISM
# ---------------------------------------------------------
def plot_physical_realism_grouped(df, output_dir):
    """Creates grouped violin plots comparing Overall and Per-Class distributions."""
    sns.set_theme(style="whitegrid")
    
    # Get correct ordering for x-axis
    order = df["Class_Label"].unique()

    # Floating & Stance Anchoring
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    sns.violinplot(data=df, x="Class_Label", y="floating", ax=axes[0], color="mediumaquamarine", inner="quartile", order=order)
    axes[0].set_title("Floating (Lowest Foot Y-Coord at Strike)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Vertical Position (meters)")
    axes[0].set_xlabel("")

    sns.violinplot(data=df, x="Class_Label", y="mean_stance_displacement", ax=axes[1], color="turquoise", inner="quartile", order=order)
    axes[1].set_title("Stance Anchoring (Mean Displacement)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Displacement (m)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "01b_phys_environment.png", dpi=300)
    plt.close()

    # Structural Constancy
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="Class_Label", y="mean_bone_length_variance", color="plum", inner="quartile", order=order)
    plt.title("Structural Constancy (Mean Bone Length Variance)", fontsize=14, pad=10, fontweight='bold')
    plt.ylabel("Variance (m²)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "01c_phys_bones.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# PD FEATURES
# ---------------------------------------------------------
def plot_pd_feature_violins(df, output_dir):
    """
    Plots violin distributions for specified clinical features.
    Seaborn handles NaNs natively during plotting.
    """
    features = [
        {"key": "mean_step_length", "title": "Mean Step Length", "ylabel": "Length (m)"},
        {"key": "mean_step_asymmetry", "title": "Mean Step Asymmetry", "ylabel": "Difference (m)"},
        {"key": "mean_walking_speed", "title": "Walking Speed", "ylabel": "Speed (m/s)"},
        {"key": "max_ankle_clearance", "title": "Max Ankle Clearance", "ylabel": "Clearance (m)"},
        {"key": "mean_emos", "title": "Estimated Margin of Stability (eMoS)", "ylabel": "eMoS (m)"},
        {"key": "mean_jerk", "title": "Mean Joint Jerk (Smoothness)", "ylabel": "Jerk (m/s³)"}
    ]

    grid_shape = (2, 3)
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(5.5 * grid_shape[1], 5 * grid_shape[0]))
    axes_flat = axes.flatten()

    labels = df["Class_Label"].unique() 

    for idx, feat_info in enumerate(features):
        ax = axes_flat[idx]
        key = feat_info["key"]
        
        if key in ["mean_emos"]:
            # Use box plot for eMoS features
            sns.boxplot(
                data=df, 
                x="Class_Label", 
                y=key, 
                ax=ax, 
                order=labels,
                hue="Class_Label", 
                palette="muted", 
                legend=False,
                width=0.5
            )
        else:
            # Use violin plot for other features
            sns.violinplot(
                data=df, 
                x="Class_Label", 
                y=key, 
                ax=ax, 
                order=labels,
                hue="Class_Label", 
                palette="muted", 
                legend=False, 
                inner="quartile"
            )

        ax.set_title(feat_info["title"], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(feat_info["ylabel"])
        ax.set_xlabel("Severity Class")
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

# ---------------------------------------------------------
# COMBINED (SPLIT) VIOLIN PLOTS
# ---------------------------------------------------------
def prepare_combined_dataframe(gt_data, gen_data):
    """
    Merges ground-truth and generated data dicts into one dataframe.
    Adds 'Source' column for Seaborn split-violins plotting.
    """
    records = []
    keys = ["overall"] + sorted([k for k in gt_data.keys() if k != "overall"])

    for cls_key in keys:
        n_seq_gt = len(gt_data[cls_key]["sequence_length"])
        n_seq_gen = len(gen_data[cls_key]["sequence_length"]) if cls_key in gen_data else 0
        
        cls_name = "Overall" if cls_key == "overall" else f"Class {cls_key}"
        
        # Compact label to prevent x-axis text overlapping
        label = f"{cls_name}\n({n_seq_gt} / {n_seq_gen})"

        # ground truth data
        for i in range(n_seq_gt):
            rec = {"Class_Label": label, "Class_ID": cls_key, "Source": "Ground Truth"}
            for k, v in gt_data[cls_key].items():
                rec[k] = v[i]
            records.append(rec)
            
        # Generated data
        if cls_key in gen_data:
            for i in range(n_seq_gen):
                rec = {"Class_Label": label, "Class_ID": cls_key, "Source": "Generated"}
                for k, v in gen_data[cls_key].items():
                    rec[k] = v[i]
                records.append(rec)

    return pd.DataFrame(records)

def plot_pd_feature_split_violins(df, distances_df, output_dir):
    """
    Plots individual split-violin distributions for GT vs Gen data.
    Adds text with the KS and Hellinger distances.
    """
    features = [
        {"key": "mean_step_length", "title": "Mean Step Length", "ylabel": "Length (m)"},
        {"key": "mean_step_asymmetry", "title": "Mean Step Asymmetry", "ylabel": "Difference (m)"},
        {"key": "mean_walking_speed", "title": "Walking Speed", "ylabel": "Speed (m/s)"},
        {"key": "max_ankle_clearance", "title": "Max Ankle Clearance", "ylabel": "Clearance (m)"},
        {"key": "mean_emos", "title": "Estimated Margin of Stability (eMoS)", "ylabel": "eMoS (m)"},
        {"key": "mean_jerk", "title": "Mean Joint Jerk (Smoothness)", "ylabel": "Jerk (m/s³)"}
    ]

    labels = df["Class_Label"].unique() 

    def get_color(score):
        if score < 0.10: return '#85e085' # Green
        if score < 0.20: return '#ffe680' # Yellow
        if score < 0.40: return '#ffb366' # Orange
        return '#ff6666' # Red

    for feat_info in features:
        # Create a fresh, standalone figure for each feature
        fig, ax = plt.subplots(figsize=(8, 6))
        key = feat_info["key"]
        
        # Split Violin Plot
        sns.violinplot(
            data=df, x="Class_Label", y=key, hue="Source", split=True, 
            ax=ax, order=labels, inner="quartile",
            palette={"Ground Truth": "cornflowerblue", "Generated": "salmon"}
        )

        ax.set_title(f"{feat_info['title']} (GT vs. Generated)", fontsize=14, fontweight='bold', pad=30)
        ax.set_ylabel(feat_info["ylabel"])
        ax.set_xlabel("Severity Class\n(Sample sizes: GT / Generated)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Calculate text placement bounds
        y_max = df[key].max()
        y_range = y_max - df[key].min()
        ax.set_ylim(df[key].min() - (y_range * 0.05), y_max + (y_range * 0.25))

        # Plot the scoring text
        x_ticks = [l.get_text() for l in ax.get_xticklabels()]
        for x_idx, label_text in enumerate(x_ticks):
            sev_name = label_text.split('\n')[0] # Extract "Overall" or "Class 0"
            
            match = distances_df[(distances_df['Severity'] == sev_name) & (distances_df['Metric'] == key)]
            if not match.empty:
                ks = match.iloc[0]['KS_Stat']
                h = match.iloc[0]['Hellinger']
                worst_score = max(ks, h)
                
                ax.text(x_idx, y_max + (y_range * 0.05), f"K: {ks:.2f}\nH: {h:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=get_color(worst_score), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

        ax.legend(
            title="Data Source", 
            bbox_to_anchor=(1.02, 1.0), 
            loc='upper left', 
            borderaxespad=0
        )

        plt.tight_layout()
        
        out_filename = output_dir / f"02b_{key}_split_violin.png"
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_smpl_mpjae(json_path, output_dir):
    """Plots the sequence-level MPJAE distributions from the JSON cache."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_dist = data.get("raw_distributions", {})
    if not raw_dist:
        print("No SMPL distributions found in JSON.")
        return

    records = []
    # Sort keys, keeping 'Overall' first
    keys = ["Overall"] + sorted([k for k in raw_dist.keys() if k != "Overall"])
    for k in keys:
        for val in raw_dist[k]:
            records.append({"Severity Class": k, "MPJAE (rad)": val})

    df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df, 
        x="Severity Class", 
        y="MPJAE (rad)", 
        order=keys, 
        inner="quartile", 
        color="lightcoral"
    )
    
    plt.title("6D Pose Reconstruction Error (Sequence MPJAE)", fontsize=14, fontweight='bold', pad=10)
    plt.ylabel("Angular Error (radians)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    out_path = Path(output_dir) / "03_smpl_mpjae_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SMPL MPJAE plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, 
                        default="thesis/data/processed/baseline_model_v2/evaluation/gt_h36m_distributions.pkl")
    parser.add_argument("--gen", type=str, 
                        default="thesis/data/processed/baseline_model_v2/evaluation/gen_h36m_distributions.pkl")
    parser.add_argument("--smpl", type=str, 
                            default="thesis/data/processed/baseline_model_v2/evaluation/smpl_mpjae_evaluation.json")
    parser.add_argument("-o", "--output", type=str, 
                        default="thesis/visualizations/baseline_model_v2")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    gen_path = Path(args.gen)
    smpl_path = Path(args.smpl)
    output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if smpl_path.exists():
        plot_smpl_mpjae(args.smpl, output_dir)

    if not gt_path.exists() or not gen_path.exists():
        print("Could not find required .pkl files.")
    else:
        print("Loading cached distributions...")
        gt_data = load_data(gt_path)

        # GT visuals
        gt_df = prepare_dataframe(gt_data)
        plot_sequence_length_distribution(gt_df, output_dir)
        plot_physical_realism_grouped(gt_df, output_dir)
        plot_pd_feature_violins(gt_df, output_dir)

        # Combined split violin plots for GT and generated data
        gen_data = load_data(gen_path)

        print("Computing distribution distances for split violin plot...")
        comparator = DistributionComparator()
        results = comparator.compare(gt_data, gen_data)
        results_df = comparator._format_results_to_dataframe(results)
        combined_df = prepare_combined_dataframe(gt_data, gen_data)
        plot_pd_feature_split_violins(combined_df, results_df, output_dir)

        print(f"\nSuccessfully generated all visuals in: {output_dir}")
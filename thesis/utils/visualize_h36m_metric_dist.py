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
# DATASET SUMMARY & PHYSICAL REALISM
# ---------------------------------------------------------
def plot_dataset_summary_stats(df, output_dir, prefix="", dataset_label=""):
    """Generates a single overview plot containing dataset counts and physical realism stats."""
    overall_df = df[df["Class_ID"] == "overall"]
    class_df = df[df["Class_ID"] != "overall"]
    
    total_seqs = len(overall_df)
    if total_seqs == 0:
        print("No data found to plot summary.")
        return

    mean_len = overall_df["sequence_length"].mean()
    std_len = overall_df["sequence_length"].std()
    median_len = overall_df["sequence_length"].median()
    
    mean_floating = overall_df["floating"].mean()
    mean_foot_disp = overall_df["mean_stance_displacement"].mean()
    mean_bone_var = overall_df["mean_bone_length_variance"].mean()
    bone_constancy = mean_bone_var < 1e-4

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    title_suffix = f" - {dataset_label}" if dataset_label else ""
    fig.suptitle(f"Dataset Overview & Physical Realism{title_suffix}", fontsize=16, fontweight='bold')

    # Left Panel: Bar chart of sequence counts
    classes = sorted(class_df["Class_ID"].unique())
    if len(classes) > 0:
        sns.countplot(data=class_df, x="Class_ID", order=classes, ax=axes[0], palette="muted")
        axes[0].set_title("Sequences per Severity Class", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Severity Class", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height())}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, "No Class Data", ha='center', va='center')
        axes[0].axis('off')

    # Right Panel: Text / Table for global stats
    axes[1].axis('off')
    
    stats_text = (
        f"Global Dataset Statistics (N = {total_seqs})\n"
        f"──────────────────────────────────────────────────\n"
        f"Sequence Length (Mean ± Std): {mean_len:.1f} ± {std_len:.1f} frames\n"
        f"Sequence Length (Median):     {median_len:.1f} frames\n\n"
        f"Physical Realism Metrics\n"
        f"──────────────────────────────────────────────────\n"
        f"Overall Mean Floating:        {mean_floating:.4f} m\n"
        f"Overall Mean Foot Displace.:  {mean_foot_disp:.4f} m\n"
        f"Overall Mean Bone Variance:   {mean_bone_var:.2e} m²\n"
        f"Bone Constancy Achieved:      {'Yes' if bone_constancy else 'No'} (< 1e-4 m²)\n"
    )
    
    # Add a bounding box for the text to make it look like a nice card
    axes[1].text(0.05, 0.5, stats_text, fontsize=13, va='center', ha='left', 
                 family='monospace', bbox=dict(facecolor='whitesmoke', alpha=0.8, edgecolor='silver', boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_filename = output_dir / f"{prefix}00_dataset_summary.png"
    plt.savefig(out_filename, dpi=300)
    plt.close()
    print(f"Saved dataset summary plot to: {out_filename}")

# ---------------------------------------------------------
# PD FEATURES
# ---------------------------------------------------------
def plot_pd_feature_violins(df, output_dir, prefix="", dataset_label=""):
    """
    Plots violin distributions for specified clinical features.
    Seaborn handles NaNs natively during plotting.
    """
    features = [
        {"key": "mean_step_length", "title": "Mean Step Length", "ylabel": "Length (m)"},
        {"key": "mean_walking_speed", "title": "Walking Speed", "ylabel": "Speed (m/s)"},
        {"key": "max_ankle_clearance", "title": "Max Ankle Clearance", "ylabel": "Clearance (m)"},
        {"key": "mean_emos", "title": "Mean Margin of Stability (eMoS)", "ylabel": "eMoS (m)"}
    ]

    grid_shape = (2, 2)
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(11, 10))
    axes_flat = axes.flatten()

    labels = df["Class_Label"].unique() 
    title_suffix = f" ({dataset_label})" if dataset_label else ""

    for idx, feat_info in enumerate(features):
        ax = axes_flat[idx]
        key = feat_info["key"]
        
        if key in ["mean_emos"]:
            # Use box plot for eMoS features, hiding extreme outliers to prevent y-axis squashing
            sns.boxplot(
                data=df, 
                x="Class_Label", 
                y=key, 
                ax=ax, 
                order=labels,
                palette="muted", 
                showfliers=False,
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
                palette="muted", 
                inner="quartile"
            )

        ax.set_title(feat_info["title"], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(feat_info["ylabel"])
        ax.set_xlabel("Severity Class")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f"Clinical PD Features by Severity Class{title_suffix}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}02_pd_features_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# COMBINED COMPARISON PLOTS
# ---------------------------------------------------------
def prepare_combined_dataframe(gt_data, gen_data):
    """
    Merges ground-truth and generated data dicts into one dataframe.
    Adds 'Source' column for Seaborn violins/boxplots plotting.
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

def plot_pd_feature_comparison_plots(df, distances_df, output_dir):
    """
    Plots individual paired distributions for GT vs Gen data.
    Adds text with the KS and Hellinger distances.
    """
    features = [
        {"key": "mean_step_length", "title": "Mean Step Length", "ylabel": "Length (m)"},
        {"key": "mean_walking_speed", "title": "Walking Speed", "ylabel": "Speed (m/s)"},
        {"key": "max_ankle_clearance", "title": "Max Ankle Clearance", "ylabel": "Clearance (m)"},
        {"key": "mean_emos", "title": "Mean Margin of Stability (eMoS)", "ylabel": "eMoS (m)"}
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
        
        # Side-by-side boxplots
        sns.boxplot(
            data=df, x="Class_Label", y=key, hue="Source", 
            ax=ax, order=labels,
            palette={"Ground Truth": "cornflowerblue", "Generated": "salmon"}
        )

        ax.set_title(f"{feat_info['title']} (GT vs. Generated)", fontsize=14, fontweight='bold', pad=30)
        ax.set_ylabel(feat_info["ylabel"])
        ax.set_xlabel("Severity Class\n(Sample sizes: GT / Generated)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Calculate text placement bounds based on actual data min/max
        y_max = df[key].max()
        y_min = df[key].min()

        y_range = y_max - y_min
        # Increase top padding to make room for text balloons higher up
        ax.set_ylim(y_min - (y_range * 0.05), y_max + (y_range * 0.35))

        # Plot the scoring text
        x_ticks = [l.get_text() for l in ax.get_xticklabels()]
        for x_idx, label_text in enumerate(x_ticks):
            sev_name = label_text.split('\n')[0] # Extract "Overall" or "Class 0"
            
            match = distances_df[(distances_df['Severity'] == sev_name) & (distances_df['Metric'] == key)]
            if not match.empty:
                ks = match.iloc[0]['KS_Stat']
                h = match.iloc[0]['Hellinger']
                worst_score = max(ks, h)
                
                # Move text higher up: y_max + (y_range * 0.15)
                ax.text(x_idx, y_max + (y_range * 0.15), f"K: {ks:.2f}\nH: {h:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=get_color(worst_score), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

        ax.legend(
            title="Data Source", 
            bbox_to_anchor=(1.02, 1.0), 
            loc='upper left', 
            borderaxespad=0
        )

        plt.tight_layout()
        
        out_filename = output_dir / f"02b_{key}_comparison_plot.png"
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    model_folder = "JointModel-MLP-Baseline"
    base_dir = f"thesis/data/processed/{model_folder}/evaluation"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, 
                        default=f"{base_dir}/gt_h36m_distributions.pkl")
    parser.add_argument("--gen", type=str, 
                        default=f"{base_dir}/gen_h36m_distributions.pkl")
    parser.add_argument("-o", "--output", type=str, 
                        default=f"thesis/visualizations/{model_folder}")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    gen_path = Path(args.gen)
    output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists() or not gen_path.exists():
        print("Could not find required .pkl files.")
    else:
        print("Loading cached distributions...")
        gt_data = load_data(gt_path)
        gen_data = load_data(gen_path)

        # GT visuals
        print("Generating Ground Truth standalone visualizations...")
        gt_df = prepare_dataframe(gt_data)
        plot_dataset_summary_stats(gt_df, output_dir, prefix="gt_", dataset_label="Ground Truth")
        plot_pd_feature_violins(gt_df, output_dir, prefix="gt_", dataset_label="Ground Truth")

        # Synthetic data visuals
        print("Generating Generated standalone visualizations...")
        gen_df = prepare_dataframe(gen_data)
        plot_dataset_summary_stats(gen_df, output_dir, prefix="gen_", dataset_label="Generated")
        plot_pd_feature_violins(gen_df, output_dir, prefix="gen_", dataset_label="Generated")

        # Combined comparison plots for GT vs Generated data
        print("Computing distribution distances for comparison plots...")
        comparator = DistributionComparator()
        results = comparator.compare(gt_data, gen_data)
        results_df = comparator._format_results_to_dataframe(results)
        combined_df = prepare_combined_dataframe(gt_data, gen_data)
        plot_pd_feature_comparison_plots(combined_df, results_df, output_dir)

        print(f"\nSuccessfully generated all visuals in: {output_dir}")
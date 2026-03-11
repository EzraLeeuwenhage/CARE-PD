import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------
# 1. PHYSICAL REALISM BOX PLOTS (Raw Data)
# ---------------------------------------------------------
def plot_physical_realism_grouped(data, output_dir):
    """Creates grouped boxplots for physical realism to avoid clutter."""
    raw_phys = data.get("global_physical_realism_raw", {})
    if not raw_phys:
        return

    # Extract N for annotation
    N = len(raw_phys.get("jitters", []))
    x_label = f"Number of Patients (N={N} seqs)"

    # Group 1: Smoothness (Jitter & Jerk)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(y=raw_phys["jitters"], ax=axes[0], color="lightcoral", width=0.3)
    axes[0].set_title("Jitter (Mean Joint Acceleration)", fontsize=12, pad=10)
    axes[0].set_ylabel("Acceleration (Leg-lengths / frames^2)")
    axes[0].set_xlabel(x_label)

    sns.boxplot(y=raw_phys["mean_joint_jerks"], ax=axes[1], color="salmon", width=0.3)
    axes[1].set_title("Jerk (Mean Rate of Accel. Change)", fontsize=12, pad=10)
    axes[1].set_ylabel("Jerk (Leg-lengths / frames^3)")
    axes[1].set_xlabel(x_label)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01a_phys_smoothness.png", dpi=300)
    plt.close()

    # Group 2: Environment Interaction
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(y=raw_phys["heel_strike_y_ranges"], ax=axes[0], color="mediumaquamarine", width=0.3)
    axes[0].set_title("Floor Consistency (Heel Strike Y-Range)", fontsize=12, pad=10)
    axes[0].set_ylabel("Vertical Distance (Leg-lengths)")
    axes[0].set_xlabel(x_label)

    sns.boxplot(y=raw_phys["foot_skating_velocities"], ax=axes[1], color="turquoise", width=0.3)
    axes[1].set_title("Foot Skating (Stance Ankle Velocity)", fontsize=12, pad=10)
    axes[1].set_ylabel("Velocity (Leg-lengths / s)")
    axes[1].set_xlabel(x_label)

    plt.tight_layout()
    plt.savefig(output_dir / "01b_phys_environment.png", dpi=300)
    plt.close()

    # Group 3: Bone Variance (Structural Constancy)
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(y=raw_phys["bone_length_variances"], color="plum", width=0.4)
    ax.set_title("Structural Constancy (Bone Length Variance)", fontsize=12, pad=10)
    ax.set_ylabel("Variance (Leg-lengths^2)")
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.savefig(output_dir / "01c_phys_bones.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 2. PD FEATURES BAR CHARTS (Summary Stats)
# ---------------------------------------------------------
def plot_pd_feature_bars(feature_dicts, data, title, filename, output_dir, grid_shape):
    """
    Plots summary stats (bars with error whiskers) for specified features.
    feature_dicts is a list of dicts: {"key": "step_length", "title": "...", "ylabel": "..."}
    """
    overall_stats = data.get("overall_pd_features", {})
    class_stats = data.get("per_class_pd_features", {})
    
    classes = sorted(list(class_stats.keys()))
    
    # Create descriptive X-labels with N counts dynamically appended!
    n_overall = overall_stats.get('count', 0)
    x_labels = [f"Overall\n(N={n_overall})"]
    for c in classes:
        n_class = class_stats[c].get('count', 0)
        x_labels.append(f"Class {c}\n(N={n_class})")
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(6 * grid_shape[1], 5 * grid_shape[0]))
    # Handle single subplots vs arrays cleanly
    axes_flat = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, feat_info in enumerate(feature_dicts):
        ax = axes_flat[idx]
        base_key = feat_info["key"]
        
        means, stds = [], []
        
        # Extract Overall
        means.append(overall_stats.get(f"{base_key}_mean", 0))
        stds.append(overall_stats.get(f"{base_key}_std", 0))
        
        # Extract Per Class
        for cls in classes:
            means.append(class_stats[cls].get(f"{base_key}_mean", 0))
            stds.append(class_stats[cls].get(f"{base_key}_std", 0))

        # Plot Bars and Whiskers
        x_pos = np.arange(len(x_labels))
        ax.bar(x_pos, means, yerr=stds, capsize=6, color=sns.color_palette("muted")[idx % 10], alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(feat_info["title"], fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel(feat_info["ylabel"])
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Clean up empty subplots
    for j in range(len(feature_dicts), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.resolve()
    data_file = SCRIPT_DIR / "patient_007_distribution.pkl"
    output_directory = SCRIPT_DIR / "visualizations"
    output_directory.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_file}")
    dataset = load_data(data_file)
    
    # 1. Physical Realism (Raw Boxplots)
    plot_physical_realism_grouped(dataset, output_directory)
    
    # 2. Step Spatial Features (2x2 Grid)
    # Grouping Mean and Std together to explicitly show intra-sequence vs inter-sequence distributions
    spatial_features = [
        {"key": "step_length_mean", "title": "Dist. of Sequence Mean Step Lengths", "ylabel": "Distance (Leg-lengths)"},
        {"key": "step_length_std", "title": "Intra-Sequence Variability (Std) of Step Length", "ylabel": "Distance (Leg-lengths)"},
        {"key": "step_width_mean", "title": "Dist. of Sequence Mean Step Widths", "ylabel": "Distance (Leg-lengths)"},
        {"key": "step_width_std", "title": "Intra-Sequence Variability (Std) of Step Width", "ylabel": "Distance (Leg-lengths)"}
    ]
    plot_pd_feature_bars(spatial_features, dataset, "Spatial Kinematics (Leg-Length Normalized)", "02_spatial_features.png", output_directory, (2, 2))

    # 3. Temporal & Pace Features (2x2 Grid)
    pace_features = [
        {"key": "cadence", "title": "Dist. of Sequence Cadence", "ylabel": "Pace (Steps / Minute)"},
        {"key": "walking_speed", "title": "Dist. of Sequence Walking Speed", "ylabel": "Speed (Leg-lengths / s)"},
        {"key": "step_time_mean", "title": "Dist. of Sequence Mean Step Times", "ylabel": "Time (Seconds)"},
        {"key": "step_time_std", "title": "Intra-Sequence Arrhythmia (Std of Step Time)", "ylabel": "Time (Seconds)"}
    ]
    plot_pd_feature_bars(pace_features, dataset, "Pace and Temporal Arrhythmia", "03_pace_features.png", output_directory, (2, 2))

    # 4. Posture, Arm Swing, and Foot Lifting (1x3 Grid)
    posture_features = [
        {"key": "gaitgen_stoop_posture", "title": "Stooped Posture (Neck-to-Pelvis Drop)", "ylabel": "Distance (Leg-lengths)"},
        {"key": "gaitgen_arm_swing", "title": "Upper-Body Rigidity (Min. Arm Swing)", "ylabel": "Distance (Leg-lengths)"},
        {"key": "foot_lifting", "title": "Vertical Foot Clearance (Foot Lifting)", "ylabel": "Distance (Leg-lengths)"}
    ]
    plot_pd_feature_bars(posture_features, dataset, "Posture and Limb Clearances", "04_posture_features.png", output_directory, (1, 3))

    # 5. Stability / Balance (1x2 Grid)
    stability_features = [
        {"key": "emos_min", "title": "Worst-Case Balance (Min eMoS)", "ylabel": "Margin (Leg-lengths)"},
        {"key": "emos_std", "title": "Balance Variability (Std of eMoS)", "ylabel": "Margin (Leg-lengths)"}
    ]
    plot_pd_feature_bars(stability_features, dataset, "Dynamic Lateral Stability (eMoS)", "05_stability_features.png", output_directory, (1, 2))

    print("\nSmartly grouped visualizations complete! Check the 'visualizations' folder.")
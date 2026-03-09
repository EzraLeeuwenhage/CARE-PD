import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def plot_physical_realism_boxplots(data, output_dir):
    """Creates a grid of boxplots for the raw physical realism features."""
    raw_phys = data.get("global_physical_realism_raw", {})
    if not raw_phys:
        print("No raw physical realism data found.")
        return

    features = list(raw_phys.keys())
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5))
    
    # Use seaborn for pretty boxplots
    for ax, feat in zip(axes, features):
        sns.boxplot(y=raw_phys[feat], ax=ax, color="skyblue", width=0.4)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=12, pad=10)
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_path = output_dir / "01_physical_realism_boxplots.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

def plot_pd_feature_bars(feature_keys, data, title, filename, output_dir, grid_shape):
    """Plots a grid of bar charts comparing Overall vs. Severity Classes."""
    overall_stats = data.get("overall_pd_features", {})
    class_stats = data.get("per_class_pd_features", {})
    
    # Identify X-axis labels (Overall + sorted severity classes)
    classes = sorted(list(class_stats.keys()))
    x_labels = ['Overall'] + [f"Class {c}" for c in classes]
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(5 * grid_shape[1], 5 * grid_shape[0]))
    axes = np.array(axes).flatten() # Flatten in case of 2D grid

    for idx, base_feat in enumerate(feature_keys):
        ax = axes[idx]
        
        means = []
        stds = []
        
        # Note: Your process_dataset script appends '_mean' and '_std' to the base feature name
        mean_key = f"{base_feat}_mean"
        std_key = f"{base_feat}_std"
        
        # 1. Get Overall stat
        means.append(overall_stats.get(mean_key, 0))
        stds.append(overall_stats.get(std_key, 0))
        
        # 2. Get Class stats
        for cls in classes:
            means.append(class_stats[cls].get(mean_key, 0))
            stds.append(class_stats[cls].get(std_key, 0))

        # Plot Bar with Error Bars
        x_pos = np.arange(len(x_labels))
        ax.bar(x_pos, means, yerr=stds, capsize=8, color=sns.color_palette("pastel")[0], edgecolor='black', alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(base_feat.replace('_', ' ').title(), fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide any unused subplots
    for j in range(len(feature_keys), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_joint_variances(data, output_dir):
    """Plots the expected joint variances across the 17 joints as a line plot."""
    overall_stats = data.get("overall_pd_features", {})
    class_stats = data.get("per_class_pd_features", {})
    
    overall_vars = overall_stats.get("gaitgen_joint_variances")
    
    if overall_vars is None:
        print("No joint variance data found.")
        return

    plt.figure(figsize=(12, 6))
    
    # X-axis for 17 joints
    joints = np.arange(17)
    
    # Plot Overall
    plt.plot(joints, overall_vars, marker='o', linewidth=3, color='black', label="Overall Patient", zorder=5)
    
    # Plot Each Class
    colors = sns.color_palette("Set2", len(class_stats))
    for idx, (cls, stats) in enumerate(sorted(class_stats.items())):
        cls_vars = stats.get("gaitgen_joint_variances")
        if cls_vars is not None:
            plt.plot(joints, cls_vars, marker='s', linestyle='--', linewidth=2, color=colors[idx], label=f"Class {cls}")

    plt.title("GaitGen Joint Variances (Motion Energy by Joint)", fontsize=14)
    plt.xlabel("Joint Index (0-16)", fontsize=12)
    plt.ylabel("Variance (\u03c3\u00b2)", fontsize=12)
    plt.xticks(joints)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_path = output_dir / "04_joint_variances_profile.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    data_file = SCRIPT_DIR / "patient_007_distribution.pkl"
    output_directory = SCRIPT_DIR / "visualizations"
    output_directory.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_file}")
    dataset = load_data(data_file)
    
    # 1. Physical Realism Boxplots
    plot_physical_realism_boxplots(dataset, output_directory)
    
    # 2. Step Spatial Features (Grouped logically as requested)
    step_spatial_keys = ["step_length_mean", "step_length_std", "step_width_mean", "step_width_std"]
    plot_pd_feature_bars(step_spatial_keys, dataset, "Step Spatial Kinematics", "02_step_spatial_features.png", output_directory, grid_shape=(2, 2))
    
    # 3. General Gait & Stability Features
    general_gait_keys = [
        "cadence", "walking_speed", "step_time_mean", "step_time_std", 
        "emos_min", "emos_std", "foot_lifting", "gaitgen_stoop_posture", "gaitgen_arm_swing"
    ]
    # 3x3 grid fits exactly 9 features
    plot_pd_feature_bars(general_gait_keys, dataset, "Clinical Gait & Stability Features", "03_general_gait_features.png", output_directory, grid_shape=(3, 3))
    
    # 4. Joint Variances
    plot_joint_variances(dataset, output_directory)
    
    print("\nVisualization complete! Check the 'visualizations' folder.")
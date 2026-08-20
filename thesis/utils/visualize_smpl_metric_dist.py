import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def plot_smpl_mpjae(json_path, output_dir):
    """Plots SMPL MPJAE by category and by individual joint in degrees."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_dist = data.get("raw_distributions", {})
    if not raw_dist or "Overall" not in raw_dist:
        print("No valid SMPL distributions found in JSON.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define standard categories vs. 24 individual joints
    categories = [
        'Overall', 'Lower Body', 'Upper Body', 'Hips', 
        'Knees', 'Ankles', 'Shoulders', 
        'Left Body', 'Right Body'
    ]
    all_joints = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
        'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
        'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    # --------------------------------------------
    # BROAD CATEGORIES & SEVERITY CLASSES
    # --------------------------------------------
    cat_records = []
    for cls_key, metrics_dict in raw_dist.items():
        for cat_name in categories:
            if cat_name in metrics_dict:
                for val in metrics_dict[cat_name]:
                    cat_records.append({
                        "Severity Class": cls_key,
                        "Category": cat_name,
                        "MPJAE (deg)": np.degrees(val)
                    })
    
    if cat_records:
        df_cat = pd.DataFrame(cat_records)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left Panel: Overall distributions across broad categories
        df_cat_overall = df_cat[df_cat["Severity Class"] == "Overall"]
        sns.violinplot(
            data=df_cat_overall, 
            x="Category", 
            y="MPJAE (deg)", 
            ax=axes[0], 
            order=categories,
            inner="quartile", 
            color="lightcoral"
        )
        axes[0].set_title("6D Pose Reconstruction Error by Body Region (Overall Dataset)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Angular Error (degrees)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis='x', rotation=30)

        # Right Panel: Category trends across clinical severity classes
        df_cat_classes = df_cat[df_cat["Severity Class"] != "Overall"]
        cls_order = sorted(df_cat_classes["Severity Class"].unique())
        sns.barplot(
            data=df_cat_classes, 
            x="Category", 
            y="MPJAE (deg)", 
            hue="Severity Class", 
            ax=axes[1],
            order=categories, 
            hue_order=cls_order, 
            palette="muted", 
            errorbar="se"
        )
        axes[1].set_title("Mean Angular Error by Region across Severity Classes", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Mean MPJAE (degrees)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].legend(title="Severity Class", loc="upper right")

        plt.tight_layout()
        cat_plot_path = out_dir / "03a_smpl_mpjae_categories.png"
        plt.savefig(cat_plot_path, dpi=300)
        plt.close()
        print(f"Saved SMPL Category breakdown plot to: {cat_plot_path}")

    # -----------------------------
    # 24 INDIVIDUAL JOINTS
    # -----------------------------
    joint_records = []
    overall_metrics = raw_dist.get("Overall", {})
    for joint_name in all_joints:
        if joint_name in overall_metrics:
            for val in overall_metrics[joint_name]:
                joint_records.append({
                    "Joint": joint_name,
                    "MPJAE (deg)": np.degrees(val)
                })
    
    if joint_records:
        df_joints = pd.DataFrame(joint_records)

        # Sort joints by median error so the plot naturally ranks hardest vs. easiest joints
        joint_order = df_joints.groupby("Joint")["MPJAE (deg)"].median().sort_values(ascending=False).index

        plt.figure(figsize=(10, 8))
        sns.boxplot(
            data=df_joints, 
            y="Joint", 
            x="MPJAE (deg)", 
            order=joint_order,
            palette="vlag_r", 
            showfliers=False
        )

        # Vertical dashed red line for Overall Mean MPJAE across all joints
        overall_mean = np.degrees(np.mean(overall_metrics.get("Overall", [0])))
        plt.axvline(
            overall_mean, 
            color="red", 
            linestyle="--", 
            linewidth=1.8, 
            label=f"Overall Mean: {overall_mean:.2f}°"
        )

        plt.title("Error Breakdown across all 24 SMPL Joints (Ranked by Median Error)", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Angular Error (degrees)")
        plt.ylabel("")
        plt.legend(loc="upper right")
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        joint_plot_path = out_dir / "03b_smpl_mpjae_all_24_joints.png"
        plt.savefig(joint_plot_path, dpi=300)
        plt.close()
        print(f"Saved 24-Joint ranked breakdown plot to: {joint_plot_path}")


def plot_arm_swing_metrics(json_path, output_dir):
    """Plots full side-by-side violin distributions for Arm Swing ROM and Asymmetry."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_dist = data.get("raw_distributions", {})
    if not raw_dist or "Overall" not in raw_dist or "GT_ROM_L" not in raw_dist["Overall"]:
        print("No Arm Swing metrics found in JSON. Skipping arm swing plot.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract all arm swing metrics into a single DataFrame
    records = []
    for cls_key, metrics_dict in raw_dist.items():
        
        # Left ROM (Convert rad to deg)
        for val in metrics_dict.get("GT_ROM_L", []):
            records.append({"Severity Class": cls_key, "Metric": "Left Arm ROM", "Source": "Ground Truth", "Value": np.degrees(val)})
        for val in metrics_dict.get("Gen_ROM_L", []):
            records.append({"Severity Class": cls_key, "Metric": "Left Arm ROM", "Source": "Generated", "Value": np.degrees(val)})
        
        # Right ROM (Convert rad to deg)
        for val in metrics_dict.get("GT_ROM_R", []):
            records.append({"Severity Class": cls_key, "Metric": "Right Arm ROM", "Source": "Ground Truth", "Value": np.degrees(val)})
        for val in metrics_dict.get("Gen_ROM_R", []):
            records.append({"Severity Class": cls_key, "Metric": "Right Arm ROM", "Source": "Generated", "Value": np.degrees(val)})
            
        # Asymmetry (Already in %, no conversion needed)
        for val in metrics_dict.get("GT_Symmetry_Index", []):
            records.append({"Severity Class": cls_key, "Metric": "Swing Asymmetry (SI)", "Source": "Ground Truth", "Value": val})
        for val in metrics_dict.get("Gen_Symmetry_Index", []):
            records.append({"Severity Class": cls_key, "Metric": "Swing Asymmetry (SI)", "Source": "Generated", "Value": val})

    df = pd.DataFrame(records)

    # Determine X-axis order (Overall first, then Class 1, 2, 3...)
    cls_order = ["Overall"] + sorted([c for c in df["Severity Class"].unique() if c != "Overall"])
    
    # 3-Panel Figure Layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Map metrics to their specific Y-axis labels
    metrics_to_plot = [
        ("Left Arm ROM", "Degrees (°)"), 
        ("Right Arm ROM", "Degrees (°)"), 
        ("Swing Asymmetry (SI)", "Symmetry Index (%)")
    ]
    
    # Standard color palette for easy GT vs. Gen visual distinction
    palette = {"Ground Truth": "lightsteelblue", "Generated": "lightcoral"}

    for i, (metric, y_label) in enumerate(metrics_to_plot):
        df_metric = df[df["Metric"] == metric]
        
        # Full side-by-side violins (split=False) grouped by Severity Class and separated by Hue
        sns.violinplot(
            data=df_metric, 
            x="Severity Class", 
            y="Value", 
            hue="Source", 
            split=False, 
            inner="quartile",
            order=cls_order,
            ax=axes[i],
            palette=palette
        )
        
        axes[i].set_title(f"{metric} Distribution: Ground Truth vs Generated", fontsize=15, fontweight='bold')
        axes[i].set_ylabel(y_label, fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', labelsize=11)
        
        # Keep legend only on the final subplot to avoid clutter
        if i < 2:
            axes[i].legend().set_visible(False)
        else:
            axes[i].legend(title="Data Source", fontsize=11, title_fontsize=12, loc="upper right")

    plt.tight_layout()
    arm_plot_path = out_dir / "03c_smpl_arm_swing_distributions.png"
    plt.savefig(arm_plot_path, dpi=300)
    plt.close()
    print(f"Saved Arm Swing distributions plot to: {arm_plot_path}")


if __name__ == "__main__":
    model_folder = "ConditionalModel-MLP-Baseline"
    base_dir = f"thesis/data/processed/{model_folder}/evaluation"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smpl", type=str, 
        default=f"{base_dir}/smpl_evaluation.json",
    )
    parser.add_argument(
        "-o", "--output", type=str, 
        default=f"thesis/visualizations/{model_folder}",
    )
    args = parser.parse_args()

    smpl_path = Path(args.smpl)
    output_dir = Path(args.output)

    if not smpl_path.exists():
        print(f"Error: Could not find SMPL evaluation JSON at: {smpl_path}")
    else:
        print(f"Loading cached SMPL evaluation data from: {smpl_path}")
        plot_smpl_mpjae(smpl_path, output_dir)
        plot_arm_swing_metrics(smpl_path, output_dir)
        print(f"\nSuccessfully generated SMPL plots in: {output_dir}")
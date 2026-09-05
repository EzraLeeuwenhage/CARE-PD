import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from thesis.src.evaluate_distributions import DistributionComparator


def plot_smpl_mpjae(data, output_dir):
    """Plots SMPL MPJAE by category and by individual joint in degrees."""
    if isinstance(data, (str, Path)):
        with open(data, 'r') as f:
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

        plt.title("MPJAE across all 24 SMPL Joints (Ordered by Median Error)", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Angular Error (degrees)")
        plt.ylabel("")
        plt.legend(loc="upper right")
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        joint_plot_path = out_dir / "03b_smpl_mpjae_all_24_joints.png"
        plt.savefig(joint_plot_path, dpi=300)
        plt.close()
        print(f"Saved 24-Joint ordered breakdown plot to: {joint_plot_path}")


def plot_arm_swing_metrics(data, output_dir, distances_df=None):
    """Plots violin distributions for Arm Swing Asymmetry with optional distance balloons."""
    if isinstance(data, (str, Path)):
        with open(data, 'r') as f:
            data = json.load(f)
            
    raw_dist = data.get("raw_distributions", {})
    if not raw_dist or "Overall" not in raw_dist or "GT_Symmetry_Index" not in raw_dist["Overall"]:
        print("No Arm Swing metrics found in JSON. Skipping arm swing plot.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_color(score):
        if score < 0.10: return '#85e085' # Green
        if score < 0.20: return '#ffe680' # Yellow
        if score < 0.40: return '#ffb366' # Orange
        return '#ff6666' # Red

    # Extract only the asymmetry metrics into a DataFrame
    records = []
    for cls_key, metrics_dict in raw_dist.items():
        for val in metrics_dict.get("GT_Symmetry_Index", []):
            records.append({"Severity Class": cls_key, "Source": "Ground Truth", "Value": val})
        for val in metrics_dict.get("Gen_Symmetry_Index", []):
            records.append({"Severity Class": cls_key, "Source": "Generated", "Value": val})

    df = pd.DataFrame(records)

    # Determine X-axis order (Overall first, then Class 1, 2, 3...)
    cls_order = ["Overall"] + sorted([c for c in df["Severity Class"].unique() if c != "Overall"])
    
    # Single Panel Figure Layout
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    metric_name = "Swing Asymmetry (SI)"
    palette = {"Ground Truth": "lightsteelblue", "Generated": "lightcoral"}

    sns.violinplot(
        data=df, 
        x="Severity Class", 
        y="Value", 
        hue="Source", 
        split=False, 
        inner="quartile",
        order=cls_order,
        ax=ax,
        palette=palette,
        bw_adjust=0.2
    )
    
    ax.set_title(f"{metric_name} Distribution: Ground Truth vs Generated", fontsize=15, fontweight='bold')
    ax.set_ylabel("Symmetry Index (%)", fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=11)

    # Plot textual distance balloons if distances_df is provided
    if distances_df is not None:
        # Retrieve the auto-scaled Y limits that already include the KDE tails
        y_min_auto, y_max_auto = ax.get_ylim()
        y_range = y_max_auto - y_min_auto
        
        # Add padding to the top for the balloons, keeping the bottom KDE tail intact
        ax.set_ylim(y_min_auto, y_max_auto + (y_range * 0.20))

        x_ticks = [l.get_text() for l in ax.get_xticklabels()]
        for x_idx, label_text in enumerate(x_ticks):
            match = distances_df[(distances_df['Severity'] == label_text) & (distances_df['Metric'] == metric_name)]
            if not match.empty:
                ks = match.iloc[0]['KS_Stat']
                h = match.iloc[0]['Hellinger']
                worst_score = max(ks, h)
                
                # Anchor the balloon slightly above the original KDE tail
                ax.text(x_idx, y_max_auto + (y_range * 0.05), f"K: {ks:.2f}\nH: {h:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=get_color(worst_score), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

    ax.legend(title="Data Source", fontsize=11, title_fontsize=12, loc="upper right")

    plt.tight_layout()
    arm_plot_path = out_dir / "03c_smpl_arm_swing_distributions.png"
    plt.savefig(arm_plot_path, dpi=300)
    plt.close()
    print(f"Saved Arm Swing distributions plot to: {arm_plot_path}")


def plot_sparc_metrics(data, output_dir, distances_df=None):
    """Plots full side-by-side distributions for SPARC smoothness metrics with optional distance balloons."""
    if isinstance(data, (str, Path)):
        with open(data, 'r') as f:
            data = json.load(f)
            
    raw_dist = data.get("raw_distributions", {})
    if not raw_dist or "Overall" not in raw_dist or "GT_SPARC_Overall" not in raw_dist["Overall"]:
        print("No SPARC metrics found in JSON. Skipping SPARC plots.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_color(score):
        if score < 0.10: return '#85e085' # Green
        if score < 0.20: return '#ffe680' # Yellow
        if score < 0.40: return '#ffb366' # Orange
        return '#ff6666' # Red
    
    palette = {"Ground Truth": "lightsteelblue", "Generated": "lightcoral"}

    # ---------------------------------------------------------
    # KNEE JOINTS FOR SEVERITY CLASS PROGRESSION
    # ---------------------------------------------------------
    knee_joints = ['L_Knee', 'R_Knee']
    knee_records = []
    
    for cls_key, metrics_dict in raw_dist.items():
        if cls_key == "Overall":
            continue
        for j_name in knee_joints:
            for val in metrics_dict.get(f"GT_SPARC_{j_name}", []):
                knee_records.append({"Severity Class": cls_key, "Joint": j_name, "Source": "Ground Truth", "SPARC": val})
            for val in metrics_dict.get(f"Gen_SPARC_{j_name}", []):
                knee_records.append({"Severity Class": cls_key, "Joint": j_name, "Source": "Generated", "SPARC": val})
                
    if knee_records:
        df_knees = pd.DataFrame(knee_records)
        cls_order = sorted(df_knees["Severity Class"].unique())
        
        # Create two fully independent subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, j_name in enumerate(knee_joints):
            ax = axes[idx]
            # Isolate data for this specific joint
            df_joint = df_knees[df_knees["Joint"] == j_name]
            
            sns.boxplot(
                data=df_joint, 
                x="Severity Class", 
                y="SPARC", 
                hue="Source", 
                palette=palette,
                order=cls_order,
                ax=ax,
                showfliers=True
            )
            
            ax.set_title(f"{j_name} Joint", fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel("SPARC Value (Higher = Smoother)", fontsize=12)
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelsize=11)
            
            # Add distance balloons dynamically scaled to plot
            if distances_df is not None:
                y_max = df_joint["SPARC"].max()
                y_min = df_joint["SPARC"].min()
                y_range = y_max - y_min
                
                # Pad top by 25% and bottom by 5% to ensure balloons and fliers fit
                ax.set_ylim(y_min - (y_range * 0.05), y_max + (y_range * 0.25))
                
                x_ticks = [l.get_text() for l in ax.get_xticklabels()]
                for x_idx, label_text in enumerate(x_ticks):
                    match = distances_df[(distances_df['Severity'] == label_text) & (distances_df['Metric'] == 'SPARC_Knees')]
                    if not match.empty:
                        ks = match.iloc[0]['KS_Stat']
                        h = match.iloc[0]['Hellinger']
                        worst_score = max(ks, h)
                        ax.text(x_idx, y_max + (y_range * 0.05), f"K: {ks:.2f}\nH: {h:.2f}",
                                ha='center', va='bottom', fontsize=10, fontweight='bold',
                                bbox=dict(facecolor=get_color(worst_score), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))
            
            if idx == 0:
                ax.get_legend().remove()
            else:
                ax.legend(title="Data Source", fontsize=11, title_fontsize=12, 
                          bbox_to_anchor=(1.03, 0.5), loc='center left', borderaxespad=0.)

        fig.suptitle("SPARC Smoothness in Knees per Severity Class", 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        knee_plot_path = out_dir / "03f_sparc_knee_class_discriminators.png"
        plt.savefig(knee_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved independent Knee Discriminators box plots to: {knee_plot_path}")

    # --------------------------------------------
    # BROAD CATEGORIES & SEVERITY CLASSES FOR SPARC
    # --------------------------------------------
    categories = [
        'Overall', 'Lower Body', 'Upper Body', 'Hips', 
        'Knees', 'Ankles', 'Shoulders', 
        'Left Body', 'Right Body'
    ]
    
    cat_records = []
    for cls_key, metrics_dict in raw_dist.items():
        for cat_name in categories:
            for val in metrics_dict.get(f"GT_SPARC_{cat_name}", []):
                cat_records.append({"Severity Class": cls_key, "Category": cat_name, "Source": "Ground Truth", "SPARC": val})
            for val in metrics_dict.get(f"Gen_SPARC_{cat_name}", []):
                cat_records.append({"Severity Class": cls_key, "Category": cat_name, "Source": "Generated", "SPARC": val})

    if cat_records:
        df_cat = pd.DataFrame(cat_records)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left Panel
        df_cat_overall = df_cat[df_cat["Severity Class"] == "Overall"]
        sns.violinplot(
            data=df_cat_overall, 
            x="Category", 
            y="SPARC", 
            hue="Source",
            split=False,
            ax=axes[0], 
            order=categories,
            inner="quartile", 
            palette=palette,
            cut=0
        )
        axes[0].set_title("SPARC Smoothness by Body Region (Overall Dataset)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("SPARC Value (Higher = Smoother)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].legend(title="Data Source", loc="lower right")

        # Right Panel: Primary Walking Joints across clinical severity classes
        target_joints = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
        leg_records = []
        
        for cls_key, metrics_dict in raw_dist.items():
            if cls_key == "Overall":
                continue
            for j_name in target_joints:
                for val in metrics_dict.get(f"GT_SPARC_{j_name}", []):
                    leg_records.append({"Severity Class": cls_key, "Source": "Ground Truth", "SPARC": val})
                for val in metrics_dict.get(f"Gen_SPARC_{j_name}", []):
                    leg_records.append({"Severity Class": cls_key, "Source": "Generated", "SPARC": val})

        df_legs = pd.DataFrame(leg_records)
        cls_order = sorted(df_legs["Severity Class"].unique())
        
        sns.violinplot(
            data=df_legs, 
            x="Severity Class", 
            y="SPARC", 
            hue="Source", 
            split=False,
            inner="quartile",
            ax=axes[1],
            order=cls_order, 
            palette=palette,
            cut=0
        )
        axes[1].set_title("Primary Walking Joints (Hips/Knees/Ankles) Across Severity", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("SPARC Value (Higher = Smoother)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis='x', rotation=30)
        
        # Add Text Balloons for SPARC Distances
        if distances_df is not None:
            y_max = df_legs["SPARC"].max()
            y_min = df_legs["SPARC"].min()
            y_range = y_max - y_min
            axes[1].set_ylim(y_min - (y_range * 0.05), y_max + (y_range * 0.35))

            x_ticks = [l.get_text() for l in axes[1].get_xticklabels()]
            for x_idx, label_text in enumerate(x_ticks):
                # We specifically labeled this as "SPARC_Lower_Limbs" in the pre-processing dataframe
                match = distances_df[(distances_df['Severity'] == label_text) & (distances_df['Metric'] == 'SPARC_Lower_Limbs')]
                if not match.empty:
                    ks = match.iloc[0]['KS_Stat']
                    h = match.iloc[0]['Hellinger']
                    worst_score = max(ks, h)
                    
                    axes[1].text(x_idx, y_max + (y_range * 0.15), f"K: {ks:.2f}\nH: {h:.2f}",
                            ha='center', va='bottom', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor=get_color(worst_score), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

        axes[1].legend(title="Data Source", loc="lower right")

        plt.tight_layout()
        sparc_cat_plot_path = out_dir / "03d_sparc_categories.png"
        plt.savefig(sparc_cat_plot_path, dpi=300)
        plt.close()
        print(f"Saved SPARC Category breakdown plot to: {sparc_cat_plot_path}")

    # -----------------------------
    # 24 INDIVIDUAL JOINTS FOR SPARC
    # -----------------------------
    all_joints = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
        'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
        'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    
    joint_records = []
    overall_metrics = raw_dist.get("Overall", {})
    for joint_name in all_joints:
        for val in overall_metrics.get(f"GT_SPARC_{joint_name}", []):
            joint_records.append({"Joint": joint_name, "Source": "Ground Truth", "SPARC": val})
        for val in overall_metrics.get(f"Gen_SPARC_{joint_name}", []):
            joint_records.append({"Joint": joint_name, "Source": "Generated", "SPARC": val})
            
    if joint_records:
        df_joints = pd.DataFrame(joint_records)

        gt_only = df_joints[df_joints["Source"] == "Ground Truth"]
        joint_order = gt_only.groupby("Joint")["SPARC"].median().sort_values(ascending=True).index

        plt.figure(figsize=(12, 10))
        sns.boxplot(
            data=df_joints, 
            y="Joint", 
            x="SPARC", 
            hue="Source",
            order=joint_order,
            palette=palette, 
            showfliers=False
        )

        plt.title("SPARC Smoothness across all 24 SMPL Joints (Ordered by GT smoothness)", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("SPARC Value (Higher = Smoother)")
        plt.ylabel("")
        plt.legend(title="Data Source", loc="lower right")
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        sparc_joint_plot_path = out_dir / "03e_sparc_all_24_joints.png"
        plt.savefig(sparc_joint_plot_path, dpi=300)
        plt.close()
        print(f"Saved 24-Joint SPARC breakdown plot to: {sparc_joint_plot_path}")


if __name__ == "__main__":
    model_folder = "JointModel-MLP-Baseline"
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
    parser.add_argument(
        "--hide_distances", action="store_true", 
        help="Flag to disable the KS & Hellinger distance balloons in the plots."
    )
    args = parser.parse_args()

    smpl_path = Path(args.smpl)
    output_dir = Path(args.output)

    if not smpl_path.exists():
        print(f"Error: Could not find SMPL evaluation JSON at: {smpl_path}")
    else:
        print(f"Loading cached SMPL evaluation data from: {smpl_path}")
        
        distances_df = None
        if not args.hide_distances:
            print("Computing KS & Hellinger Distances for SMPL Metrics...")
            with open(smpl_path, 'r') as f:
                data_json = json.load(f)
            
            raw_dist = data_json.get("raw_distributions", {})
            gt_comp = defaultdict(dict)
            gen_comp = defaultdict(dict)
            target_sparc_joints = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']

            for sev_key, metrics in raw_dist.items():
                # Map "Overall" to "overall" and "Class X" to "X" to match the H36M Comparator structure
                c_key = "overall" if sev_key == "Overall" else sev_key.replace("Class ", "")
                
                # Arm Swing (Forced to np.array to prevent TypeError during NaN filtering)
                gt_comp[c_key]["Left Arm ROM"] = np.array(np.degrees(metrics.get("GT_ROM_L", [])))
                gen_comp[c_key]["Left Arm ROM"] = np.array(np.degrees(metrics.get("Gen_ROM_L", [])))
                
                gt_comp[c_key]["Right Arm ROM"] = np.array(np.degrees(metrics.get("GT_ROM_R", [])))
                gen_comp[c_key]["Right Arm ROM"] = np.array(np.degrees(metrics.get("Gen_ROM_R", [])))
                
                # The explicit np.array cast added here!
                gt_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("GT_Symmetry_Index", []))
                gen_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("Gen_Symmetry_Index", []))
                
                # SPARC Lower Limbs (Pooled)
                gt_legs = []
                gen_legs = []
                for j in target_sparc_joints:
                    gt_legs.extend(metrics.get(f"GT_SPARC_{j}", []))
                    gen_legs.extend(metrics.get(f"Gen_SPARC_{j}", []))
                gt_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gt_legs)
                gen_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gen_legs)

            # Generate dataframe containing distances
            comparator = DistributionComparator()
            results = comparator.compare(gt_comp, gen_comp)
            distances_df = comparator._format_results_to_dataframe(results)

        plot_smpl_mpjae(smpl_path, output_dir)
        plot_arm_swing_metrics(smpl_path, output_dir, distances_df=distances_df)
        plot_sparc_metrics(smpl_path, output_dir, distances_df=distances_df)
        print(f"\nSuccessfully generated SMPL plots in: {output_dir}")
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.spatial.transform import Rotation

from thesis.src.evaluate_distributions import DistributionComparator
from thesis.src.utils.geometry_utils import pose_to_rmat


SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

SMPL_CATEGORIES = [
    'Overall', 'Lower Body', 'Upper Body', 'Hips', 
    'Knees', 'Ankles', 'Shoulders', 
    'Left Body', 'Right Body'
]

CLINICAL_METRICS_SPECS = [
    {
        "name": "Ankle Bradykinesia",
        "gt_key": "GT_Ankle_Bradykinesia",
        "gen_key": "Gen_Ankle_Bradykinesia",
        "filename": "smpl_bradykinesia_worse_ankle_comparison.png",
        "ylabel": "Physical Amplitude AUC (deg/s·Hz)",
        "clinical_note": "Lower = More Severe"
    },
    {
        "name": "Spine Rigidity",
        "gt_key": "GT_Spine_Rigidity",
        "gen_key": "Gen_Spine_Rigidity",
        "filename": "smpl_spine_rigidity_comparison.png",
        "ylabel": "Physical Amplitude AUC (deg/s·Hz)",
        "clinical_note": "Lower = More Severe"
    },
    {
        "name": "Ankle SI",
        "gt_key": "GT_Ankle_SI",
        "gen_key": "Gen_Ankle_SI",
        "filename": "smpl_asymmetry_ankle_si_comparison.png",
        "ylabel": "Robinson Symmetry Index (%)",
        "clinical_note": "Higher = More Asymmetric"
    },
    {
        "name": "Wrist Smoothness AUC",
        "gt_key": "GT_Wrist_Smoothness_AUC",
        "gen_key": "Gen_Wrist_Smoothness_AUC",
        "filename": "smpl_smoothness_wrist_auc_comparison.png",
        "ylabel": "Relative Jitter Energy (%)",
        "clinical_note": "Higher = More Severe"
    },
    {
        "name": "Hand Smoothness AUC",
        "gt_key": "GT_Hand_Smoothness_AUC",
        "gen_key": "Gen_Hand_Smoothness_AUC",
        "filename": "smpl_smoothness_hand_auc_comparison.png",
        "ylabel": "Relative Jitter Energy (%)",
        "clinical_note": "Higher = More Severe"
    }
]

# MPJAE VISUALIZATIONS
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

    # Broad Categories Breakdown
    cat_records = []
    for cls_key, metrics_dict in raw_dist.items():
        for cat_name in SMPL_CATEGORIES:
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

        # Overall distributions across broad categories (Boxplot)
        df_cat_overall = df_cat[df_cat["Severity Class"] == "Overall"]
        sns.boxplot(
            data=df_cat_overall, x="Category", y="MPJAE (deg)", 
            ax=axes[0], order=SMPL_CATEGORIES, color="lightcoral", 
            showfliers=False, width=0.5
        )
        axes[0].set_title("6D Pose Reconstruction Error by Body Region (Overall Dataset)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Angular Error (degrees)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis='x', rotation=30)

        # RCategory trends across clinical severity classes
        df_cat_classes = df_cat[df_cat["Severity Class"] != "Overall"]
        cls_order = sorted(df_cat_classes["Severity Class"].unique())
        sns.barplot(
            data=df_cat_classes, x="Category", y="MPJAE (deg)", hue="Severity Class", 
            ax=axes[1], order=SMPL_CATEGORIES, hue_order=cls_order, 
            palette="muted", errorbar="se"
        )
        axes[1].set_title("Mean Angular Error by Region across Severity Classes", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Mean MPJAE (degrees)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].legend(title="Severity Class", loc="upper right")

        plt.tight_layout()
        cat_plot_path = out_dir / "smpl_mpjae_categories.png"
        plt.savefig(cat_plot_path, dpi=300)
        plt.close()
        print(f"Saved SMPL Category breakdown plot to: {cat_plot_path}")

    # Individual 24 Joints Breakdown
    joint_records = []
    overall_metrics = raw_dist.get("Overall", {})
    for joint_name in SMPL_JOINT_NAMES:
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
            data=df_joints, y="Joint", x="MPJAE (deg)", 
            order=joint_order, palette="vlag_r", showfliers=False
        )

        # Vertical dashed red line for Overall Mean MPJAE across all joints
        overall_mean = np.degrees(np.mean(overall_metrics.get("Overall", [0])))
        plt.axvline(
            overall_mean, color="red", linestyle="--", linewidth=1.8, 
            label=f"Overall Mean: {overall_mean:.2f}°"
        )

        plt.title("MPJAE across all 24 SMPL Joints (Ordered by Median Error)", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Angular Error (degrees)")
        plt.ylabel("")
        plt.legend(loc="upper right")
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        joint_plot_path = out_dir / "smpl_mpjae_all_24_joints.png"
        plt.savefig(joint_plot_path, dpi=300)
        plt.close()
        print(f"Saved 24-Joint breakdown plot to: {joint_plot_path}")


# BRADYKINESIA AND SMOOTHNESS VISUALIZATIONS
def plot_clinical_metric_distributions(data, output_dir, distances_df=None, show_outliers=False):
    """Plots standalone Ground Truth vs. Generated distribution figures for each clinical metric."""
    if isinstance(data, (str, Path)):
        with open(data, 'r') as f:
            data = json.load(f)

    raw_dist = data.get("raw_distributions", {})
    if not raw_dist:
        print("No raw distributions found in JSON.")
        return

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_color(score):
        if score < 0.10: return '#85e085' # Green
        if score < 0.20: return '#ffe680' # Yellow
        if score < 0.40: return '#ffb366' # Orange
        return '#ff6666' # Red

    palette = {"Ground Truth": "cornflowerblue", "Generated": "salmon"}

    for spec in CLINICAL_METRICS_SPECS:
        metric_name = spec["name"]
        gt_key = spec["gt_key"]
        gen_key = spec["gen_key"]

        records = []
        for sev_key, metrics_dict in raw_dist.items():
            cls_name = "Overall" if sev_key.lower() == "overall" else (f"Class {sev_key}" if not str(sev_key).startswith("Class ") else str(sev_key))
            for val in metrics_dict.get(gt_key, []):
                records.append({"Severity Class": cls_name, "Source": "Ground Truth", "Value": val})
            for val in metrics_dict.get(gen_key, []):
                records.append({"Severity Class": cls_name, "Source": "Generated", "Value": val})

        if not records:
            continue

        df_metric = pd.DataFrame(records)
        cls_order = ["Overall"] + sorted([c for c in df_metric["Severity Class"].unique() if c != "Overall"])

        fig, ax = plt.subplots(figsize=(8.5, 6))

        sns.boxplot(
            data=df_metric, x="Severity Class", y="Value", hue="Source",
            order=cls_order, palette=palette, showfliers=show_outliers,
            width=0.5, ax=ax
        )

        ax.set_title(
            f"{metric_name} (Ground Truth vs. Generated)\n[{spec['clinical_note']}]", 
            fontsize=13, 
            fontweight='bold', 
            pad=25
        )
        ax.set_ylabel(spec["ylabel"], fontsize=11)
        ax.set_xlabel("Clinical Severity Class", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Plot distance badges if available
        if distances_df is not None:
            y_min_auto, y_max_auto = ax.get_ylim()
            y_range = max(y_max_auto - y_min_auto, 1e-5)
            ax.set_ylim(y_min_auto - (y_range * 0.05), y_max_auto + (y_range * 0.30))

            x_ticks = [l.get_text() for l in ax.get_xticklabels()]
            for x_idx, label_text in enumerate(x_ticks):
                match = distances_df[(distances_df['Severity'] == label_text) & (distances_df['Metric'] == metric_name)]
                if not match.empty:
                    ks = match.iloc[0]['KS_Stat']
                    w = match.iloc[0]['Wasserstein']
                    ax.text(
                        x_idx, y_max_auto + (y_range * 0.10), f"K: {ks:.2f}\nW: {w:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=get_color(ks), edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9)
                    )

        ax.legend(title="Data Source", loc="upper right")
        plt.tight_layout()
        save_path = out_dir / spec["filename"]
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} plot to: {save_path}")

# optional plotting of the frequency spectra for bradykinesia and smoothness metrics
def plot_clinical_spectra(gt_npz_path, gen_npz_path, labels_path, output_dir, fps: int = 30, nfft: int = 2048):
    """Generates overlaid Ground Truth vs. Generated spectral profile comparisons."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)["key_to_severity"]

    gt_data = np.load(gt_npz_path, allow_pickle=True)
    gen_data = np.load(gen_npz_path, allow_pickle=True)
    gt_dict = gt_data['arr_0'].item() if 'arr_0' in gt_data.files else {k: gt_data[k] for k in gt_data.files}
    gen_dict = gen_data['arr_0'].item() if 'arr_0' in gen_data.files else {k: gen_data[k] for k in gen_data.files}

    RAD2DEG = 180.0 / np.pi
    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']
    classes = [0, 1, 2, 3]
    valid_f = (f >= 0.0) & (f <= 8.0)
    common_keys = [k for k in gt_dict.keys() if k in gen_dict.keys() and not k.endswith('_trans')]

    def _compute_angular_speed(pose_seq):
        rot_mats = pose_to_rmat(pose_seq)
        if hasattr(rot_mats, "numpy"):
            rot_mats = rot_mats.numpy()
        T, J, _, _ = rot_mats.shape
        R_rel = np.matmul(np.swapaxes(rot_mats[:-1], -1, -2), rot_mats[1:])
        rotvecs = Rotation.from_matrix(R_rel.reshape(-1, 3, 3)).as_rotvec().reshape(T - 1, J, 3)
        return np.linalg.norm(rotvecs * fps, axis=-1)

    gt_spectra = {"ankle": {c: [] for c in classes}, "spine": {c: [] for c in classes},
                  "wrist": {c: [] for c in classes}, "hand": {c: [] for c in classes}}
    gen_spectra = {"ankle": {c: [] for c in classes}, "spine": {c: [] for c in classes},
                   "wrist": {c: [] for c in classes}, "hand": {c: [] for c in classes}}

    for k in common_keys:
        sev = labels_dict.get(k, -1)
        if sev not in classes:
            continue

        gt_a_t = _compute_angular_speed(gt_dict[k])
        gen_a_t = _compute_angular_speed(gen_dict[k])

        for data_source, a_t, target_dict in [("GT", gt_a_t, gt_spectra), ("Gen", gen_a_t, gen_spectra)]:
            T = a_t.shape[0]
            if T < 2:
                continue

            a_phys = {}
            psd_norm = {}
            for j in [6, 7, 8, 9, 20, 21, 22, 23]:
                sig = a_t[:, j] - np.mean(a_t[:, j])
                A_raw = np.abs(np.fft.rfft(sig, n=nfft))
                a_phys[j] = (2.0 / T) * A_raw * RAD2DEG
                psd_raw = (A_raw ** 2) / nfft
                psd_norm[j] = psd_raw / (np.sum(psd_raw) + 1e-8)

            # Worse ankle (Joints 7 & 8)
            auc_l = np.sum(a_phys[7][valid_f])
            auc_r = np.sum(a_phys[8][valid_f])
            target_dict["ankle"][sev].append(a_phys[7] if auc_l <= auc_r else a_phys[8])

            # Axial spine (Joints 6 & 9)
            target_dict["spine"][sev].append(0.5 * (a_phys[6] + a_phys[9]))

            # Worse wrist (Joints 20 & 21)
            jit_l = np.sum(psd_norm[20][valid_f])
            jit_r = np.sum(psd_norm[21][valid_f])
            target_dict["wrist"][sev].append(psd_norm[20] if jit_l >= jit_r else psd_norm[21])

            # Worse hand (Joints 22 & 23)
            jit_lh = np.sum(psd_norm[22][valid_f])
            jit_rh = np.sum(psd_norm[23][valid_f])
            target_dict["hand"][sev].append(psd_norm[22] if jit_lh >= jit_rh else psd_norm[23])

    def _render_spectrum_comparison(gt_curves, gen_curves, title, ylabel, save_filename):
        plt.figure(figsize=(10.5, 5.5))
        for c in classes:
            if len(gt_curves[c]) > 0:
                plt.plot(f[valid_f], np.mean(gt_curves[c], axis=0)[valid_f], color=colors[c],
                         linestyle='-', linewidth=2.0, label=f'Class {c} (GT)')
            if len(gen_curves[c]) > 0:
                plt.plot(f[valid_f], np.mean(gen_curves[c], axis=0)[valid_f], color=colors[c],
                         linestyle='--', linewidth=1.8, label=f'Class {c} (Gen)')

        plt.axvspan(0.5, 3.0, color='green', alpha=0.08, label='Voluntary Locomotion (0.5–3.0 Hz)')
        plt.axvspan(3.0, 8.0, color='red', alpha=0.08, label='Jitter Band (3.0–8.0 Hz)')
        plt.title(title, fontweight='bold', fontsize=12)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        plt.legend(loc="upper right", fontsize=8.5, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / save_filename, dpi=300)
        plt.close()
        print(f"Saved spectrum plot to: {out_dir / save_filename}")

    _render_spectrum_comparison(
        gt_spectra["ankle"], gen_spectra["ankle"],
        "Bradykinesia Harmonic Profile: Worse Ankle (GT vs. Gen)\n[Physical Amplitude Spectrum: deg/s]",
        "Harmonic Amplitude (deg/s)", "smpl_spectrum_worse_ankle_bradykinesia_comparison.png"
    )
    _render_spectrum_comparison(
        gt_spectra["spine"], gen_spectra["spine"],
        "Axial Rigidity Profile: Trunk Spine (GT vs. Gen)\n[Physical Amplitude Spectrum: deg/s]",
        "Harmonic Amplitude (deg/s)", "smpl_spectrum_axial_spine_rigidity_comparison.png"
    )
    _render_spectrum_comparison(
        gt_spectra["wrist"], gen_spectra["wrist"],
        "Smoothness Degradation: Worse Wrist (GT vs. Gen)\n[Relative Energy Distribution]",
        "Fractional Power / Bin", "smpl_spectrum_worse_wrist_smoothness_comparison.png"
    )
    _render_spectrum_comparison(
        gt_spectra["hand"], gen_spectra["hand"],
        "Smoothness Degradation: Worse Hand (GT vs. Gen)\n[Relative Energy Distribution]",
        "Fractional Power / Bin", "smpl_spectrum_worse_hand_smoothness_comparison.png"
    )


if __name__ == "__main__":
    model_folder = "JointModel-MLP-Baseline"
    base_dir = f"thesis/data/processed/{model_folder}/evaluation"

    parser = argparse.ArgumentParser(description="Visualize SMPL evaluation distributions.")
    parser.add_argument("--smpl", type=str, default=f"{base_dir}/smpl_evaluation.json")
    parser.add_argument("-o", "--output", type=str, default=f"thesis/visualizations/{model_folder}")
    parser.add_argument("--hide_distances", action="store_true", help="Disable KS & Wasserstein distance balloons.")
    parser.add_argument("--show_outliers", action="store_true", help="Show flier points on box plots.")
    parser.add_argument("--plot_spectra", action="store_true", help="Render physical and relative spectral curves (off by default).")
    parser.add_argument("--gt_npz", type=str, default=None, help="Path to ground_truth_6d.npz (required if --plot_spectra is set).")
    parser.add_argument("--gen_npz", type=str, default=None, help="Path to generated_6d.npz (required if --plot_spectra is set).")
    parser.add_argument("--labels", type=str, default=None, help="Path to gen_labels.json (required if --plot_spectra is set).")
    args = parser.parse_args()

    smpl_path = Path(args.smpl)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not smpl_path.exists():
        print(f"Error: Could not find SMPL evaluation JSON at: {smpl_path}")
        exit(1)

    print(f"Loading cached SMPL evaluation data from: {smpl_path}")
    with open(smpl_path, 'r') as f:
        data_json = json.load(f)

    distances_df = None
    if not args.hide_distances:
        print("Computing KS & Wasserstein Distances for SMPL Clinical Metrics...")
        raw_dist = data_json.get("raw_distributions", {})
        gt_comp = defaultdict(dict)
        gen_comp = defaultdict(dict)

        for sev_key, metrics in raw_dist.items():
            c_key = "overall" if sev_key.lower() == "overall" else sev_key.replace("Class ", "")
            for spec in CLINICAL_METRICS_SPECS:
                m_name = spec["name"]
                gt_k = spec["gt_key"]
                gen_k = spec["gen_key"]
                if gt_k in metrics and gen_k in metrics:
                    gt_comp[c_key][m_name] = np.array(metrics[gt_k])
                    gen_comp[c_key][m_name] = np.array(metrics[gen_k])

        comparator = DistributionComparator()
        results = comparator.compare(gt_comp, gen_comp)
        distances_df = comparator._format_results_to_dataframe(results)

    # Render standardized plots
    plot_smpl_mpjae(smpl_path, output_dir)
    plot_clinical_metric_distributions(smpl_path, output_dir, distances_df=distances_df, show_outliers=args.show_outliers)

    # Optional spectra plotting
    if args.plot_spectra:
        gt_npz = Path(args.gt_npz) if args.gt_npz else smpl_path.parent.parent / "6D_SMPL" / "ground_truth_6d.npz"
        gen_npz = Path(args.gen_npz) if args.gen_npz else smpl_path.parent.parent / "6D_SMPL" / "generated_6d.npz"
        lbls = Path(args.labels) if args.labels else smpl_path.parent.parent / "h36m" / "gen_labels.json"

        if gt_npz.exists() and gen_npz.exists() and lbls.exists():
            print("Rendering optional clinical harmonic spectra...")
            plot_clinical_spectra(gt_npz, gen_npz, lbls, output_dir)
        else:
            print("Could not locate npz datasets for spectra plotting. Pass --gt_npz, --gen_npz, and --labels.")

    print(f"\nAll SMPL visuals successfully updated in: {output_dir}\n")
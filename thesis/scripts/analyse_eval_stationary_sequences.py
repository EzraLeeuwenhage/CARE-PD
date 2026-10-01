import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from smplx.body_models import SMPL

from thesis.src.dataloader import get_dataloader
from thesis.src.utils.geometry_utils import batched_forward_to_h36m
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator


def main():
    out_dir = Path("thesis/visualizations/01_analyses/stationary_pathology/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CONFIGURE DATALOADER (filter_z_travel=False to load stationary sequences)
    # -------------------------------------------------------------
    raw_smpl_path = Path("thesis/data/raw/PD-GaM/3D_SMPL/PD-GaM_3D_SMPL_rot_trans_canonical.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")

    cfg = {
        'data': {
            'smpl_path': raw_smpl_path,
            'severity_labels_path': labels_path,
            'patient_prefix': 'all',
        },
        'windowing': {
            'total_window_size': 60,
            'prefix_length': 15,
            'step_size': 45,
            'max_sequence_len': 200,   # High ceiling so sequences are not sliced
            'full_seq_step_size': 200,
            'min_z_travel': 0.5,
            'filter_z_travel': False,  # MUST BE False to prevent dataloader from discarding stationary sequences
        },
        'training': {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'eval_split': 0.0,         # 100% of data included
            'test_split': 0.0,
            'overfit_severity_class': -1,
        },
        'model': {
            'generation_mode': 'one_shot',  # Instructs get_dataloader to use FullSequenceSMPLDataset
        }
    }

    print("\n--- Initializing FullSequenceSMPLDataset via DataLoader ---")
    loader = get_dataloader(cfg, mode='train')

    # -------------------------------------------------------------
    # 2. INITIALIZE SMPL -> H36M CONVERSION PIPELINE & EVALUATORS
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SMPL Neutral Model and H36M Regressor on {device}...")
    smpl_model = SMPL(
        model_path='thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl', 
        num_betas=10
    ).eval().to(device)
    h36m_regressor = torch.tensor(
        np.load('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy'), 
        dtype=torch.float32
    ).to(device)

    # Set min_z_travel=0.0 so H36M evaluator processes stationary sequences without zeroing them
    h36m_evaluator = H36MEvaluator(fps=30, min_z_travel=0.0)
    smpl_evaluator = SMPLEvaluator(fps=30)

    # -------------------------------------------------------------
    # 3. EXTRACT & ISOLATE STATIONARY SEQUENCES (< 0.5m Z-travel)
    # -------------------------------------------------------------
    stationary_records = []
    total_sequences = 0

    print(f"\nProcessing {len(loader)} total sequences to isolate stationary gait...")
    for batch in tqdm(loader, desc="Filtering & Evaluating Stationary Gait"):
        total_sequences += 1
        key = batch['key'][0]
        actual_len = batch['seq_len'][0].item()
        sev = batch['severity'][0].item()

        # Slice to actual sequence length
        pose = batch['pose'][:, :actual_len]
        trans = batch['trans'][:, :actual_len]

        # Check total anterior-posterior displacement
        z_travel = abs(trans[0, -1, 2] - trans[0, 0, 2]).item()
        if z_travel >= 0.5:
            continue  # Ignore mobile sequences

        # --- Pipeline SMPL -> H36M Forward Kinematics Conversion ---
        pose_dev = pose.to(device)
        trans_dev = trans.to(device)
        h36m_joints_list = batched_forward_to_h36m([pose_dev], [trans_dev], smpl_model, h36m_regressor, device)
        h36m_joints = h36m_joints_list[0]  # Shape: (T, 17, 3)

        # --- A. H36M Heel-Strike Independent Metric Extraction ---
        h36m_metrics = h36m_evaluator._extract_sequence_metrics(h36m_joints, clip_id=key)
        global_speed = h36m_metrics["mean_global_walking_speed"]
        cadence = h36m_metrics["mean_cadence"]

        # --- B. SMPL Clinical Metric Extraction (SO(3) Angular Velocities) ---
        clinical_metrics = smpl_evaluator.compute_clinical_metrics(pose[0])

        stationary_records.append({
            "Key": key,
            "Severity": sev,
            "Z_Travel_m": z_travel,
            "Seq_Len": actual_len,
            "Global_Walking_Speed": global_speed,
            "Mean_Cadence": cadence,
            "Ankle_Bradykinesia": clinical_metrics["ankle_bradykinesia"],
            "Spine_Rigidity": clinical_metrics["spine_rigidity"],
            "Ankle_SI": clinical_metrics["ankle_si"],
            "Wrist_Smoothness_AUC": clinical_metrics["wrist_smoothness_auc"],
            "Hand_Smoothness_AUC": clinical_metrics["hand_smoothness_auc"]
        })

    df = pd.DataFrame(stationary_records)
    print(f"\nSuccessfully isolated {len(df)} stationary sequences out of {total_sequences} total sequences.")
    print("Class breakdown of stationary sequences:")
    print(df["Severity"].value_counts().sort_index())

    print("\n" + "=" * 80)
    print("STATIONARY KEYS DICTIONARY:")
    print("=" * 80)
    print("STATIONARY_KEYS = {")
    for sev_c in sorted(df["Severity"].unique()):
        keys_list = df[df["Severity"] == sev_c]["Key"].tolist()
        print(f"    {sev_c}: {keys_list},")
    print("}")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------
    # 4. TERMINAL SUMMARY TABLE
    # -------------------------------------------------------------
    mean_stats = df.groupby("Severity").mean(numeric_only=True)
    std_stats = df.groupby("Severity").std(numeric_only=True)
    classes = [0, 1, 2, 3]

    print("\n" + "=" * 135)
    print("STATIONARY COHORT EVALUATION (Z-DISPLACEMENT < 0.5m) - PER-CLASS MEANS ± STD")
    print("=" * 135)
    header = (
        f"{'Class':<6} | {'N':<4} | {'Global Speed (m/s)':<19} | {'Cadence (steps/min)':<20} | "
        f"{'Ankle Brady (°/s·Hz)':<22} | {'Spine Rigid (°/s·Hz)':<22} | {'Ankle SI (%)':<13} | {'Wrist Jitter (%)':<17}"
    )
    print(header)
    print("-" * len(header))
    for c in classes:
        if c in mean_stats.index:
            n_count = len(df[df["Severity"] == c])
            gs = f"{mean_stats.loc[c, 'Global_Walking_Speed']:.3f}±{std_stats.loc[c, 'Global_Walking_Speed']:.3f}"
            cad = f"{mean_stats.loc[c, 'Mean_Cadence']:.1f}±{std_stats.loc[c, 'Mean_Cadence']:.1f}"
            ab = f"{mean_stats.loc[c, 'Ankle_Bradykinesia']:.2f}±{std_stats.loc[c, 'Ankle_Bradykinesia']:.2f}"
            sr = f"{mean_stats.loc[c, 'Spine_Rigidity']:.2f}±{std_stats.loc[c, 'Spine_Rigidity']:.2f}"
            asi = f"{mean_stats.loc[c, 'Ankle_SI']:.1f}±{std_stats.loc[c, 'Ankle_SI']:.1f}%"
            wj = f"{mean_stats.loc[c, 'Wrist_Smoothness_AUC']:.2f}±{std_stats.loc[c, 'Wrist_Smoothness_AUC']:.2f}%"
            print(f" {c:<5} | {n_count:<4} | {gs:<19} | {cad:<20} | {ab:<22} | {sr:<22} | {asi:<13} | {wj:<17}")
    print("=" * 135 + "\n")

    # -------------------------------------------------------------
    # 5. VISUALIZATIONS (EXACT PIPELINE STYLING)
    # -------------------------------------------------------------
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']

    metric_specs = [
        ("Ankle_Bradykinesia", "Worse Ankle Bradykinesia (deg/s·Hz)", "Lower = More Severe", "stationary_smpl_bradykinesia_worse_ankle.png"),
        ("Spine_Rigidity", "Axial Spine Rigidity (deg/s·Hz)", "Lower = More Severe", "stationary_smpl_spine_rigidity.png"),
        ("Ankle_SI", "Ankle Robinson Symmetry Index (%)", "Higher = More Asymmetric", "stationary_smpl_asymmetry_ankle_si.png"),
        ("Wrist_Smoothness_AUC", "Worse Wrist Jitter Energy (%)", "Higher = More Severe", "stationary_smpl_smoothness_wrist_auc.png"),
        ("Hand_Smoothness_AUC", "Worse Hand Jitter Energy (%)", "Higher = More Severe", "stationary_smpl_smoothness_hand_auc.png"),
        ("Global_Walking_Speed", "Global Walking Speed (m/s)", "Lower = More Severe", "stationary_h36m_global_walking_speed.png"),
        ("Mean_Cadence", "Mean Cadence (steps/min)", "Stepping Rhythm", "stationary_h36m_mean_cadence.png"),
    ]

    # --- A. Master 7-Panel Pathology Dashboard (2x4 Grid) ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()

    for idx, (col_name, plot_label, clinical_interp, _) in enumerate(metric_specs):
        ax = axes_flat[idx]
        box_data = [df[df['Severity'] == c][col_name].dropna().values for c in classes]
        bp = ax.boxplot(
            box_data, 
            labels=[f"Class {c}\n(N={len(box_data[c])})" for c in classes],
            patch_artist=True, 
            showfliers=False, 
            widths=0.45
        )
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_title(f"{plot_label}\n({clinical_interp})", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Hide the unused 8th subplot in the 2x4 grid
    axes_flat[7].axis('off')

    fig.suptitle(
        "Stationary Gait Pathology Profile (Z-Displacement < 0.5m)\n"
        "[Heel-Strike Independent Kinematics & SO(3) Clinical Metrics]",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    dashboard_path = out_dir / "stationary_master_clinical_dashboard.png"
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    print(f"Saved Master Stationary Dashboard to: {dashboard_path}")

    # --- B. Individual Standalone Plots ---
    for col_name, plot_label, clinical_interp, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        box_data = [df[df['Severity'] == c][col_name].dropna().values for c in classes]
        bp = ax.boxplot(
            box_data, 
            labels=[f"Class {c}\n(N={len(box_data[c])})" for c in classes],
            patch_artist=True, 
            showfliers=False, 
            widths=0.45
        )
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
            
        ax.set_title(f"{plot_label}\n[Stationary Cohort: {clinical_interp}]", fontsize=12, fontweight='bold', pad=12)
        ax.set_ylabel(plot_label.split('(')[-1].replace(')', ''), fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = out_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved standalone plot to: {save_path}")

    print(f"\nAll stationary evaluation artifacts successfully saved to:\n  {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
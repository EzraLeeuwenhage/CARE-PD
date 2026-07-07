import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import linregress

def load_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
        return data['arr_0'].item()
    return {k: np.array(data[k]) for k in data.files}

def analyze_floating(npz_path, labels_path, fps=30):
    print(f"Loading data from {npz_path}...")
    data = load_data(npz_path)
    
    with open(labels_path, "r") as f:
        key_to_severity = json.load(f)["key_to_severity"]
        
    records = []
    trajectories = {} # Store raw trajectories for plotting the worst offenders
    
    PELVIS_IDX = 0
    # Left and Right ankle indices in H36M
    L_ANKLE_IDX = 6 
    R_ANKLE_IDX = 3

    for clip_id, seq in data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        
        severity = None
        for reg_key, reg_score in key_to_severity.items():
            if reg_key in base_key or base_key in reg_key:
                severity = int(reg_score)
                break
                
        if severity is None:
            continue
            
        T = seq.shape[0]
        time_seconds = T / fps
        
        # Extract Y trajectories
        pelvis_y = seq[:, PELVIS_IDX, 1]
        
        # Find the lowest foot Y coordinate per frame
        lowest_foot_y = np.minimum(seq[:, L_ANKLE_IDX, 1], seq[:, R_ANKLE_IDX, 1])
        
        # Fit linear regression to find the systematic drift
        # Multiply slope by fps to get drift in meters/second
        pelvis_slope, _, _, _, _ = linregress(np.arange(T), pelvis_y)
        pelvis_drift_rate = pelvis_slope * fps 
        
        foot_slope, _, _, _, _ = linregress(np.arange(T), lowest_foot_y)
        foot_drift_rate = foot_slope * fps
        
        records.append({
            "clip_id": clip_id,
            "severity": severity,
            "sequence_length": T,
            "time_seconds": time_seconds,
            "pelvis_drift_m_s": pelvis_drift_rate,
            "foot_drift_m_s": foot_drift_rate
        })
        
        trajectories[clip_id] = {"pelvis": pelvis_y, "foot": lowest_foot_y}

    return pd.DataFrame(records), trajectories

def visualize_floating_analysis(df, trajectories, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Drift vs Sequence Length (Scatter)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="sequence_length", y="foot_drift_m_s", hue="severity", palette="deep", alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title("Is Floating Correlated with Sequence Length?")
    plt.ylabel("Lowest Foot Vertical Drift Rate (m/s)")
    plt.xlabel("Sequence Length (Frames)")
    plt.tight_layout()
    plt.savefig(out_dir / "floating_vs_length.png", dpi=300)
    plt.close()

    # 2. Drift vs Severity Class (Boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df, x="severity", y="pelvis_drift_m_s", ax=axes[0], palette="muted")
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title("Pelvis Upward Drift by Severity")
    axes[0].set_ylabel("Drift Rate (m/s)")

    sns.boxplot(data=df, x="severity", y="foot_drift_m_s", ax=axes[1], palette="muted")
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title("True Floating (Lowest Foot Drift) by Severity")
    axes[1].set_ylabel("Drift Rate (m/s)")

    plt.tight_layout()
    plt.savefig(out_dir / "floating_by_severity.png", dpi=300)
    plt.close()

    # 3. Plot the raw trajectories of the top 5 worst floaters
    top_floaters = df.nlargest(5, "foot_drift_m_s")
    
    plt.figure(figsize=(12, 6))
    for _, row in top_floaters.iterrows():
        clip_id = row['clip_id']
        y_traj = trajectories[clip_id]["foot"]
        # Normalize to start at 0 so we can compare drift clearly
        y_traj_normalized = y_traj - y_traj[0] 
        plt.plot(y_traj_normalized, label=f"Class {row['severity']} | T={row['sequence_length']} | {clip_id[:15]}...")

    plt.title("Raw Trajectories of the Top 5 Worst Floating Sequences")
    plt.ylabel("Change in Lowest Foot Height (meters)")
    plt.xlabel("Frames")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "worst_floaters_trajectories.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    npz_path = Path("thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")
    output_dir = Path("thesis/visualizations/floating_analysis")
    
    print("Analyzing sequences for floating/drifting...")
    df, trajectories = analyze_floating(npz_path, labels_path)
    
    print("\nDrift Summary (Meters per Second):")
    summary = df.groupby("severity")["foot_drift_m_s"].agg(["mean", "std", "max"])
    print(summary)
    
    print(f"\nGenerating visualizations in {output_dir}...")
    visualize_floating_analysis(df, trajectories, output_dir)
    print("Done!")
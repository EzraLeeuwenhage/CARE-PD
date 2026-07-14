import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_veering(npz_path, labels_path):
    print(f"Loading data from: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
        data = data['arr_0'].item()
    else:
        data = {k: np.array(data[k]) for k in data.files}

    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        key_to_severity = json.load(f)["key_to_severity"]

    records = []
    PELVIS = 0
    X_AXIS = 0
    
    for clip_id, seq in data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        
        if base_key in key_to_severity:
            severity = int(key_to_severity[base_key])
        else:
            continue
            
        pelvis_x = seq[:, PELVIS, X_AXIS]
        
        # Subtract pelvis_x[0] to get relative position to start
        net_veer = pelvis_x[-1] - pelvis_x[0]
        max_abs_veer = np.max(np.abs(pelvis_x - pelvis_x[0]))
        
        records.append({
            "Severity Class": severity,
            "Net Lateral Displacement (m)": net_veer,
            "Max Absolute Veering (m)": max_abs_veer
        })

    df = pd.DataFrame(records)

    df["Class_Label"] = df["Severity Class"].apply(lambda x: f"Class {x}")
    df = df.sort_values("Severity Class")
    class_counts = df["Class_Label"].value_counts()
    df["Class_Label"] = df["Class_Label"].apply(lambda x: f"{x}\n(N={class_counts[x]})")
    
    print(f"Processed {len(df)} valid sequences.")
    
    # Create plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.violinplot(
        data=df, 
        x="Class_Label", 
        y="Net Lateral Displacement (m)", 
        ax=axes[0], 
        palette="muted", 
        inner="quartile"
    )
    axes[0].set_title("Net Lateral Veering (Left/Right Bias)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("X-Axis Displacement (meters)")
    axes[0].set_xlabel("")
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    sns.violinplot(
        data=df, 
        x="Class_Label", 
        y="Max Absolute Veering (m)", 
        ax=axes[1], 
        palette="muted", 
        inner="quartile"
    )
    axes[1].set_title("Magnitude of Veering (Absolute Max Deviation)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Absolute Deviation (meters)")
    axes[1].set_xlabel("")
    
    plt.suptitle("Gait Veering Distributions by Severity Class", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    npz_path = Path("thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")
    
    if not npz_path.exists() or not labels_path.exists():
        print("Data paths not found. Please ensure you are running this from the project root.")
    else:
        analyze_veering(npz_path, labels_path)
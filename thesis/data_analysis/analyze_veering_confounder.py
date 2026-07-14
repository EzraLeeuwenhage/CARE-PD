import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

def analyze_veering_confounders(npz_path, labels_path):
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
        data = data['arr_0'].item()
    else:
        data = {k: np.array(data[k]) for k in data.files}

    with open(labels_path, 'r') as f:
        key_to_severity = json.load(f)["key_to_severity"]

    records = []
    PELVIS = 0
    X_AXIS = 0
    Z_AXIS = 2
    
    for clip_id, seq in data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        
        if base_key in key_to_severity:
            severity = int(key_to_severity[base_key])
        else:
            continue
            
        pelvis_x = seq[:, PELVIS, X_AXIS]
        pelvis_z = seq[:, PELVIS, Z_AXIS]
        
        max_abs_veer = np.max(np.abs(pelvis_x - pelvis_x[0]))
        z_distance = np.abs(pelvis_z[-1] - pelvis_z[0])
        
        # Avoid division by zero for completely stationary sequences
        if z_distance < 0.01:
            continue
            
        normalized_veer = max_abs_veer / z_distance
        
        records.append({
            "Severity Class": severity,
            "Max Absolute Veering (m)": max_abs_veer,
            "Total Z-Distance (m)": z_distance,
            "Normalized Veering (Veer per Meter)": normalized_veer
        })

    df = pd.DataFrame(records)
    
    # Statistical analysis of correlation 
    corr, p_val = pearsonr(df["Total Z-Distance (m)"], df["Max Absolute Veering (m)"])
    print(f"\n[STATISTICS]")
    print(f"Correlation between Z-Distance and Raw Veering: {corr:.3f} (p={p_val:.3e})")
    if corr > 0.5:
        print("-> STRONG CONFOUNDER DETECTED: Veering is heavily dependent on distance walked.")

    # Correlation between Severity Class and Raw Veering
    corr_sev_veer, p_val_sev_veer = pearsonr(df["Severity Class"], df["Max Absolute Veering (m)"])
    print(f"Correlation between Severity Class and Raw Veering: {corr_sev_veer:.3f} (p={p_val_sev_veer:.3e})")

    # Correlation between Severity Class and Normalized Veering (to prove the confounder correction)
    corr_sev_norm, p_val_sev_norm = pearsonr(df["Severity Class"], df["Normalized Veering (Veer per Meter)"])
    print(f"Correlation between Severity Class and Normalized Veering: {corr_sev_norm:.3f} (p={p_val_sev_norm:.3e})")
    
    # Create plots

    df["Class_Label"] = df["Severity Class"].apply(lambda x: f"Class {x}")
    df = df.sort_values("Severity Class")
    class_counts = df["Class_Label"].value_counts()
    df["Class_Label"] = df["Class_Label"].apply(lambda x: f"{x}\n(N={class_counts[x]})")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.violinplot(
        data=df, x="Class_Label", y="Total Z-Distance (m)", 
        ax=axes[0], palette="Blues", inner="quartile"
    )
    axes[0].set_title("1. Z-Distance by Severity", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("")
    
    sns.scatterplot(
        data=df, x="Total Z-Distance (m)", y="Max Absolute Veering (m)", 
        hue="Severity Class", palette="viridis", ax=axes[1], alpha=0.6, edgecolor=None
    )
    axes[1].set_title(f"2. The Confounder (Corr: {corr:.2f})", fontsize=12, fontweight='bold')
    
    sns.violinplot(
        data=df, x="Class_Label", y="Normalized Veering (Veer per Meter)", 
        ax=axes[2], palette="Reds", inner="quartile"
    )
    axes[2].set_title("3. True Veering (Normalized per Meter)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("")
    
    plt.suptitle("Disentangling Gait Veering from Distance Walked", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    npz_path = Path("thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")
    analyze_veering_confounders(npz_path, labels_path)
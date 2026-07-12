import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_npz_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
        return data['arr_0'].item()
    return {k: np.array(data[k]) for k in data.files}

def extract_class3_metrics(npz_path, labels_path):
    print("Loading data...")
    data = load_npz_data(npz_path)
    
    with open(labels_path, "r") as f:
        key_to_severity = json.load(f)["key_to_severity"]
        
    class3_data = []
    plot_id = 1
    
    PELVIS_IDX = 0
    Z_AXIS_IDX = 2

    for clip_id, seq in data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        
        severity = None
        for reg_key, reg_score in key_to_severity.items():
            if reg_key in base_key or base_key in reg_key:
                severity = int(reg_score)
                break
                
        if severity == 3:
            num_frames = seq.shape[0]
            if num_frames == 0:
                continue
                
            start_z = seq[0, PELVIS_IDX, Z_AXIS_IDX]
            end_z = seq[-1, PELVIS_IDX, Z_AXIS_IDX]
            z_distance = abs(end_z - start_z)
            
            class3_data.append({
                "id": plot_id,
                "key": clip_id,
                "frames": num_frames,
                "z_dist": z_distance
            })
            plot_id += 1
            
    return class3_data

def visualize_class3_scatter(class3_data, output_path):
    if not class3_data:
        print("No Class 3 sequences found.")
        return
        
    frames = [item["frames"] for item in class3_data]
    z_dists = [item["z_dist"] for item in class3_data]
    ids = [item["id"] for item in class3_data]
    
    # plt.figure(figsize=(12, 8))
    # plt.scatter(frames, z_dists, color='firebrick', alpha=0.7, edgecolors='black', s=50)
    
    # for i, txt in enumerate(ids):
    #     plt.annotate(str(txt), 
    #                  (frames[i], z_dists[i]), 
    #                  xytext=(5, 5), 
    #                  textcoords='offset points',
    #                  fontsize=9)
                     
    # plt.title("Class 3 Sequences: Sequence Length vs Z-Axis Pelvis Travel", fontsize=14, fontweight='bold', pad=15)
    # plt.xlabel("Sequence Length (Frames)", fontsize=12)
    # plt.ylabel("Absolute Pelvis Travel in Z-Direction (meters)", fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)
    
    # out_file = Path(output_path)
    # out_file.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_file, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"\nSaved plot to {out_file}")
    
    print("\nMapping of Plot IDs to Sequence Keys:")
    print("-" * 80)
    print(f"{'Plot ID':<10} | {'Sequence Length':<17} | {'Z-Axis Distance':<20} | {'Sequence Key'}")
    print("-" * 80)
    for item in sorted(class3_data, key=lambda x: x['z_dist'], reverse=True):
        print(f"{item['id']:<10} | {item['frames']:<17} | {item['z_dist']:<20.3f} | {item['key']}")


def analyze_chunked_class3_stats(npz_path, labels_path, chunk_size=60):
    print(f"\nAnalyzing data in {chunk_size}-frame chunks...")
    data = load_npz_data(npz_path)
    
    with open(labels_path, "r") as f:
        key_to_severity = json.load(f)["key_to_severity"]
        
    chunked_data = []
    
    PELVIS_IDX = 0
    Z_AXIS_IDX = 2

    for clip_id, seq in data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        
        severity = None
        for reg_key, reg_score in key_to_severity.items():
            if reg_key in base_key or base_key in reg_key:
                severity = int(reg_score)
                break
                
        if severity == 3:
            num_frames = seq.shape[0]
            
            for start_idx in range(0, num_frames - chunk_size + 1, chunk_size):
                chunk = seq[start_idx:start_idx + chunk_size]
                
                start_z = chunk[0, PELVIS_IDX, Z_AXIS_IDX]
                end_z = chunk[-1, PELVIS_IDX, Z_AXIS_IDX]
                z_distance = abs(end_z - start_z)
                
                chunk_key = f"{clip_id} (frames {start_idx}-{start_idx+chunk_size})"
                
                chunked_data.append({
                    "key": chunk_key,
                    "z_dist": z_distance
                })

    if not chunked_data:
        print("No Class 3 chunks found.")
        return

    z_dists = [item["z_dist"] for item in chunked_data]
    mean_z = np.mean(z_dists)
    median_z = np.median(z_dists)
    min_z = np.min(z_dists)
    max_z = np.max(z_dists)

    print("\nSummary Statistics for 60-Frame Chunks:")
    print("-" * 50)
    print(f"Total Chunks: {len(chunked_data)}")
    print(f"Mean Z-Travel:   {mean_z:.4f} m")
    print(f"Median Z-Travel: {median_z:.4f} m")
    print(f"Min Z-Travel:    {min_z:.4f} m")
    print(f"Max Z-Travel:    {max_z:.4f} m")

    print("\nMapping of Chunks Sorted by Z-Axis Travel (Descending):")
    print("-" * 80)
    print(f"{'Z-Axis Distance':<20} | {'Chunk Key'}")
    print("-" * 80)
    for item in sorted(chunked_data, key=lambda x: x['z_dist'], reverse=True):
        print(f"{item['z_dist']:<20.3f} | {item['key']}")


if __name__ == "__main__":
    npz_path = Path("thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")
    output_path = Path("thesis/visualizations/class3_z_travel_scatter.png")
    
    metrics = extract_class3_metrics(npz_path, labels_path)
    visualize_class3_scatter(metrics, output_path)
    analyze_chunked_class3_stats(npz_path, labels_path, chunk_size=60)
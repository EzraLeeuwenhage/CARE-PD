import numpy as np
import joblib
from pathlib import Path

from thesis.utils.sixD2smpl import convert_6d_to_smpl
from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m

def convert_6d_npz_to_smpl_pkl(npz_path, pkl_path):
    print(f"--- 1. Loading 6D SMPL data from {npz_path} ---")
    data = np.load(npz_path, allow_pickle=True)
    formatted_data = {}
    
    # Filter for ignoring global translation keys
    pose_keys = [k for k in data.files if not k.endswith('_trans')]
    
    for k in pose_keys:
        pose_6d = data[k]
        trans = data[f"{k}_trans"]
        
        # Parse subject_id and walk_id
        parts = k.split('__')
        subject_id = parts[0]
        walk_id = parts[1]
        
        pose_3d = convert_6d_to_smpl(pose_6d[None, ...])[0] 
        pose_flat = pose_3d.reshape(-1, 72).astype(np.float32)
        
        if subject_id not in formatted_data:
            formatted_data[subject_id] = {}
            
        formatted_data[subject_id][walk_id] = {
            'pose': pose_flat,
            'trans': trans.astype(np.float32),
            'beta': np.zeros((1, 10), dtype=np.float32) 
        }
        
    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(formatted_data, pkl_path)
    print(f"Saved intermediate SMPL .pkl to {pkl_path}")

def compare_h36m_npz(new_npz_path, baseline_npz_path):
    print(f"\n--- 3. Comparing Data ---")
    print(f"New Pipeline: {new_npz_path}")
    print(f"Old Baseline: {baseline_npz_path}")
    
    new_data = np.load(new_npz_path, allow_pickle=True)
    baseline_data = np.load(baseline_npz_path, allow_pickle=True)
    
    # Handle different dict structures
    if hasattr(new_data, 'files') and len(new_data.files) == 1 and new_data.files[0] == 'arr_0':
        new_data = new_data['arr_0'].item()
    else:
        new_data = {k: np.array(new_data[k]) for k in new_data.files}
        
    if hasattr(baseline_data, 'files') and len(baseline_data.files) == 1 and baseline_data.files[0] == 'arr_0':
        baseline_data = baseline_data['arr_0'].item()
    else:
        baseline_data = {k: np.array(baseline_data[k]) for k in baseline_data.files}
        
    keys_new = set(new_data.keys())
    keys_base = set(baseline_data.keys())
    common_keys = keys_new.intersection(keys_base)
    
    print(f"Found {len(common_keys)} matching sequences to compare.") # should be 1700
    
    max_errors = []
    mean_errors = []
    max_euc_errors = []
    mean_euc_errors = []
    
    for k in common_keys:
        seq_new = new_data[k]
        seq_base = baseline_data[k]
        
        # use min length for future downsampling cases
        min_len = min(seq_new.shape[0], seq_base.shape[0])
        seq_new = seq_new[:min_len]
        seq_base = seq_base[:min_len]
        
        # Calculate max and mean absolute errors for individual axes
        diff = np.abs(seq_new - seq_base)
        max_errors.append(np.max(diff))
        mean_errors.append(np.mean(diff))

        # Calculate absolute error using Euclidean distance
        euc_diff = np.linalg.norm(seq_new - seq_base, axis=-1) # Shape: (T, num_joints)
        max_euc_errors.append(np.max(euc_diff))
        mean_euc_errors.append(np.mean(euc_diff))
        
    overall_max = np.max(max_errors)
    overall_mean = np.mean(mean_errors)

    overall_max_euc = np.max(max_euc_errors)
    overall_mean_euc = np.mean(mean_euc_errors)
    
    print(f"\n================ VALIDATION RESULTS ================")
    print(f"Mean Absolute Error (m): {overall_mean:.8f}")
    print(f"Max Absolute Error (m):  {overall_max:.8f}")

    print(f"Max 3D Spatial Euclidean Error (m):  {overall_max_euc:.8f}")
    print(f"Mean 3D Spatial Euclidean Error (m): {overall_mean_euc:.8f}")
    
    # account for small possible rounding errors
    if overall_max < 1e-2: 
        print("SUCCESS: The pipeline results match closely enough.")
    else:
        print("FAIL: Difference between pipeline results is too large to be rounding error.")


if __name__ == "__main__":
    input_6d_npz = "thesis/data/raw/PD-GaM/6D_SMPL/PD-GaM_6D_SMPL_rot_trans_canonical.npz"
    baseline_h36m_npz = "thesis/data/raw/PD-GaM/h36m/PD-GaM_h36m_3d_world_no_slope_no_veering.npz"
    
    # Define temp staging paths
    temp_pkl = "thesis/data/raw/PD-GaM/temp_validation/temp_pipeline_smpl.pkl"
    temp_h36m_dir = "thesis/data/raw/PD-GaM/temp_validation/h36m"
 
    # build SMPL from 6D SMPL, then convert SMPL to H36M with post-processing version
    convert_6d_npz_to_smpl_pkl(input_6d_npz, temp_pkl)
    print(f"\n--- 2. Pushing through smpl2h36m.py ---")
    out_npz = convert_smpl_to_h36m(temp_pkl, temp_h36m_dir)
    
    # out_npz = "thesis/data/raw/PD-GaM/temp_validation/h36m/temp_pipeline_smpl_NEW_h36m_3d_world.npz"
    
    # Compare H36M against known pre-processing H36M version
    compare_h36m_npz(out_npz, baseline_h36m_npz)
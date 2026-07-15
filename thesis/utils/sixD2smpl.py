import numpy as np
import joblib
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def convert_6d_to_smpl(pose_6d_tensor):
    """Converts 6D continuous rotations back to 3D Axis-Angle representations."""
    if isinstance(pose_6d_tensor, np.ndarray):
        pose_6d_tensor = torch.tensor(pose_6d_tensor, dtype=torch.float32)

    # Use shape[-2] instead of shape[1] so it works regardless of batch dimension
    # Drop the empty 25th joint if it exists
    if pose_6d_tensor.shape[-2] == 25:
        pose_6d_tensor = pose_6d_tensor[..., :24, :]
        
    original_shape = pose_6d_tensor.shape
    
    # Gram-Schmidt Orthogonalization
    v1 = pose_6d_tensor[..., :3]
    v2 = pose_6d_tensor[..., 3:]
    
    x = torch.nn.functional.normalize(v1, dim=-1)
    y_raw = v2 - (torch.sum(x * v2, dim=-1, keepdim=True) * x)
    y = torch.nn.functional.normalize(y_raw, dim=-1)
    z = torch.cross(x, y, dim=-1)
    
    # Stack along dim=-2 (because CARE-PD SMPL -> 6D conversion slices on rows aswell)
    rot_mats = torch.stack([x, y, z], dim=-2)
    rot_mats_flat = rot_mats.view(-1, 3, 3).cpu().numpy()
    
    rotations = R.from_matrix(rot_mats_flat)
    axis_angles_flat = rotations.as_rotvec()
    
    # Dynamically reshape based on input dimensions (e.g., T, 24, 3 or B, T, 24, 3)
    smpl_pose = axis_angles_flat.reshape(*original_shape[:-1], 3)
    return smpl_pose


def build_smpl_pkl_from_6d_smpl(generated_pose_6d, generated_trans, output_filepath, subject_id="GEN", walk_prefix="gen_walk"):
    """Converts raw generated 6D SMPL tensors into base SMPL .pkl file.
    
    Takes batch input of shape (B, T, 24, 6) and (B, T, 3) and outputs same structure as the original CARE-PD SMPL data.
    """
    if torch.is_tensor(generated_pose_6d):
        generated_pose_6d = generated_pose_6d.detach().cpu().numpy()
    if torch.is_tensor(generated_trans):
        generated_trans = generated_trans.detach().cpu().numpy()
        
    batch_size = generated_pose_6d.shape[0]
    formatted_data = {subject_id: {}}
    
    for i in range(batch_size):
        seq_6d = generated_pose_6d[i] # (T, 24, 6)
        seq_trans = generated_trans[i] # (T, 3)
        
        # Convert 6D to axis-angle and flatten
        seq_smpl_3d = convert_6d_to_smpl(seq_6d) # (T, 24, 3)
        seq_pose_flat = seq_smpl_3d.reshape(-1, 72).astype(np.float32)
        
        # Create neutral beta shape parameters, just like CARE-PD dataset
        neutral_betas = np.zeros((1, 10), dtype=np.float32)

        # Create unique walk ID
        walk_id = f"{walk_prefix}_{i:03d}"
        
        formatted_data[subject_id][walk_id] = {
            'pose': seq_pose_flat,
            'trans': seq_trans.astype(np.float32),
            'beta': neutral_betas
        }
        
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(formatted_data, out_path)
    print(f"Saved generated SMPL data to: {out_path}")


def validate_reconstruction(num_samples=100):
    """Validates that 6D -> SMPL reverse conversion is lossless over a random sample of sequences."""
    import pickle
    import random

    print("Loading validation data...")
    pkl_path = Path("thesis/data/raw/PD-GaM/PD-GaM.pkl")
    npz_path = Path("thesis/data/raw/PD-GaM/6D_SMPL/6D_SMPL_30f_or_longer.npz")
    
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # FILTER OUT THE TRANSLATION KEYS so the validator only tests poses
    all_keys = [k for k in npz_data.files if not k.endswith('_trans')]
    
    # Ensure we don't sample more keys than actually exist
    num_samples = min(num_samples, len(all_keys))
    sample_keys = random.sample(all_keys, num_samples)
    
    print(f"\nValidating {num_samples} randomly selected sequences...")
    
    total_mae = []
    max_errors = []
    skipped = 0
    
    for key in sample_keys:
        # Robust parsing for standard and downsampled keys
        parts = key.split('__')
        subject_id = parts[0]
        rest = parts[1]
        
        if '_down' in rest:
            walk_id, down_str = rest.rsplit('_down', 1)
            down = int(down_str)
        else:
            walk_id = rest
            down = 0
            
        down_sample_rate = 1 # PD-GaM default
            
        pose_6d = npz_data[key]
        smpl_hat = convert_6d_to_smpl(pose_6d)  # Shape: (T, 24, 3)
        
        try:
            raw_pose = pkl_data[subject_id][walk_id]['pose']
        except KeyError:
            print(f"Warning: Could not find '{subject_id} - {walk_id}' in .pkl. Skipping.")
            skipped += 1
            continue

        raw_pose = raw_pose.reshape(-1, 24, 3)
        raw_pose_sliced = raw_pose[down::down_sample_rate, ...]
        
        min_frames = min(smpl_hat.shape[0], raw_pose_sliced.shape[0])
        smpl_hat = smpl_hat[:min_frames, :24, :] 
        raw_pose_sliced = raw_pose_sliced[:min_frames, :24, :]
        
        # Calculate error with rotation matrices
        R_hat = R.from_rotvec(smpl_hat.reshape(-1, 3)).as_matrix()
        R_raw = R.from_rotvec(raw_pose_sliced.reshape(-1, 3)).as_matrix()
        
        matrix_mae = np.mean(np.abs(R_hat - R_raw))
        matrix_max = np.max(np.abs(R_hat - R_raw))
        
        total_mae.append(matrix_mae)
        max_errors.append(matrix_max)
        
    overall_mean_mae = np.mean(total_mae)
    overall_max_error = np.max(max_errors)
    
    print(f"\n--- Aggregated Validation Results ({len(total_mae)} sequences) ---")
    print(f"Sequences Skipped (Not Found):      {skipped}")
    print(f"Overall Matrix Mean Absolute Error: {overall_mean_mae:.8f}")
    print(f"Absolute Worst-Case Max Error:      {overall_max_error:.8f}")
    
    if overall_mean_mae < 1e-4 and overall_max_error < 1e-3:
        print("\nSUCCESS: Reverse conversion is practically lossless across the dataset.")
    else:
        print("\nFAIL: Error is higher than allowed threshold.")

if __name__ == "__main__":
    validate_reconstruction(num_samples=250)
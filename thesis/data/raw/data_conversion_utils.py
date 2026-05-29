import numpy as np
import joblib
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def convert_6d_to_smpl(pose_6d_tensor):
    """Converts 6D continuous rotations back to 3D Axis-Angle representations."""
    if isinstance(pose_6d_tensor, np.ndarray):
        pose_6d_tensor = torch.tensor(pose_6d_tensor, dtype=torch.float32)

    # Drop the empty 25th joint if it exists (because it contains no information)
    if pose_6d_tensor.shape[1] == 25:
        pose_6d_tensor = pose_6d_tensor[:, :24, :]
        
    T, V, _ = pose_6d_tensor.shape
    
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
    
    smpl_pose = axis_angles_flat.reshape(T, V, 3)
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
        seq_smpl_3d = convert_6d_to_smpl(seq_6d) # (T, 24, 3)t
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


def validate_reconstruction():
    """Validates that 6D -> SMPL reverse conversion is lossless by comparing to raw .pkl data."""
    import pickle

    print("Loading validation data...")
    pkl_path = Path("thesis/data/raw/PD-GaM/PD-GaM.pkl")
    npz_path = Path("thesis/data/raw/PD-GaM/6D_SMPL/6D_SMPL_30f_or_longer.npz")
    
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Just pick some key for testing
    test_key = npz_data.files[0]
    print(f"\nValidating key: {test_key}")
    
    # Parse key to find raw data sequence
    parts = test_key.split('__')
    subject_id = parts[0]
    rest = parts[1]
    
    # Only works for PD-GaM (no framerate conversion) for now
    walk_id = rest
    down = 0
    down_sample_rate = 1
        
    pose_6d = npz_data[test_key]
    smpl_hat = convert_6d_to_smpl(pose_6d)  # Shape: (T, 24, 3)
    
    # Get original raw SMPL data for the same sequence
    raw_pose = pkl_data[subject_id][walk_id]['pose']  # Shape: (T, 72)
    raw_pose = raw_pose.reshape(-1, 24, 3)  # Reshape to match our format
    
    # Apply temporal slicing to match frame rate if needed
    raw_pose_sliced = raw_pose[down::down_sample_rate, ...]
    
    # Handle any frame mismatch, drop 25th empty joint data if present
    min_frames = min(smpl_hat.shape[0], raw_pose_sliced.shape[0])
    smpl_hat = smpl_hat[:min_frames, :24, :] 
    raw_pose_sliced = raw_pose_sliced[:min_frames, :24, :]
    
    # Calculate error with rotation matrices, which accounts for multiple equivalent axis-angle representations
    R_hat = R.from_rotvec(smpl_hat.reshape(-1, 3)).as_matrix()
    R_raw = R.from_rotvec(raw_pose_sliced.reshape(-1, 3)).as_matrix()
    
    print(f"\nValidation Results")
    print(f"Reconstructed Shape: {smpl_hat.shape}")
    print(f"Original Sliced Shape: {raw_pose_sliced.shape}")
    
    # Matrix error is true physical difference in 3D space
    matrix_mae = np.mean(np.abs(R_hat - R_raw))
    matrix_max = np.max(np.abs(R_hat - R_raw))
    
    print(f"Matrix Mean Absolute Error: {matrix_mae:.8f}")
    print(f"Matrix Max Absolute Error:  {matrix_max:.8f}")
    
    if matrix_mae < 1e-4:
        print("\nSUCCESS: Reverse conversion is practically lossless.")
    else:
        print("\nFAIL: Error is higher than allowed.")


if __name__ == "__main__":
    validate_reconstruction()
import numpy as np
import joblib
import torch
from pathlib import Path
from thesis.src.utils.geometry_utils import convert_6d_to_smpl

def save_smpl_pkl(generated_pose, generated_trans, output_filepath, subject_id="GEN", walk_prefix="gen_walk"):
    """
    Converts raw generated SMPL tensors (3D Axis-Angle or 6D Continuous) directly into a base SMPL .pkl file.
    
    Takes batch input of shape (B, T, 24, D) and (B, T, 3) and outputs the same hierarchical 
    dictionary structure as the original CARE-PD SMPL dataset.
    
    Args:
        generated_pose: Tensor of shape (B, T, 24, 3) or (B, T, 24, 6)
        generated_trans: Tensor of shape (B, T, 3)
        output_filepath: Destination path for the .pkl file
        subject_id: Root dictionary key for the subject (default: "GEN")
        walk_prefix: Prefix for individual sequence keys (default: "gen_walk")
    """
    dim = generated_pose.shape[-1]
    
    if dim == 6:
        # Convert 6D continuous rotations back to 3D axis-angle
        generated_pose = convert_6d_to_smpl(generated_pose)
    elif dim == 3:
        if torch.is_tensor(generated_pose):
            generated_pose = generated_pose.detach().cpu().numpy()
            
    if torch.is_tensor(generated_trans):
        generated_trans = generated_trans.detach().cpu().numpy()
        
    batch_size = generated_pose.shape[0]
    formatted_data = {subject_id: {}}
    
    for i in range(batch_size):
        seq_3d = generated_pose[i] # (T, 24, 3)
        seq_trans = generated_trans[i] # (T, 3)
        
        # 3D is already axis-angle, so we just flatten it to (T, 72)
        seq_pose_flat = seq_3d.reshape(-1, 72).astype(np.float32)
        
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
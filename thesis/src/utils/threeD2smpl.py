import numpy as np
import joblib
import torch
from pathlib import Path

def build_smpl_pkl_from_3d_smpl(generated_pose_3d, generated_trans, output_filepath, subject_id="GEN", walk_prefix="gen_walk"):
    """
    Converts raw generated 3D SMPL tensors directly into base SMPL .pkl file.
    Takes batch input of shape (B, T, 24, 3) and (B, T, 3).
    """
    if torch.is_tensor(generated_pose_3d):
        generated_pose_3d = generated_pose_3d.detach().cpu().numpy()
    if torch.is_tensor(generated_trans):
        generated_trans = generated_trans.detach().cpu().numpy()
        
    batch_size = generated_pose_3d.shape[0]
    formatted_data = {subject_id: {}}
    
    for i in range(batch_size):
        seq_3d = generated_pose_3d[i] # (T, 24, 3)
        seq_trans = generated_trans[i] # (T, 3)
        
        # 3D is already axis-angle, so we just flatten it to (T, 72)
        seq_pose_flat = seq_3d.reshape(-1, 72).astype(np.float32)
        neutral_betas = np.zeros((1, 10), dtype=np.float32)
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
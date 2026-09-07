"""
Unified preprocessing logic for SMPL input.
Converts raw SMPL sequences to Canonicalized 3D Axis-Angle and 6D Continuous Rotations.
"""

import os
import sys
import torch
import joblib
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thesis.src.care_pd.conversion_utils import (
    _DEVICE,
    generate_smpl_in_world,
    transform_seq_so_it_has_no_slope_h36m,
    qbetween_np,
    qrot_np,
    quat_to_scipy,
    axis_angle_to_matrix,
    matrix_to_rotation_6d
)

def canonicalize_smpl_sequence(pose_world, h36m_joints_world, smpl_model, h36m_regressor):
    """Canonicalizes the SMPL sequence to a standard orientation and position in the world frame."""
    T = pose_world.shape[0]
    
    # H36M slope correction
    h36m_curr, rot_mats_slope = transform_seq_so_it_has_no_slope_h36m(
        h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial=4, return_rot_matrices=True
    )

    floor_height = h36m_curr.min(axis=0).min(axis=0)[1]
    h36m_curr[:, :, 1] -= floor_height

    root_pos_init = h36m_curr[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    h36m_curr = h36m_curr - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = 1, 4, 14, 11
    across1 = h36m_curr[0, r_hip] - h36m_curr[0, l_hip]
    across2 = h36m_curr[0, sdr_r] - h36m_curr[0, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target_for_world = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target_for_world)
    root_quat_init_full = np.ones(h36m_curr.shape[:-1] + (4,)) * root_quat_init
    h36m_curr = qrot_np(root_quat_init_full, h36m_curr)

    # H36M veering correction
    first_frame = h36m_curr[0, 0]
    middle_frame = h36m_curr[T // 2, 0]

    walking_direction = middle_frame - first_frame
    walking_direction[1] = 0 
    if np.linalg.norm(walking_direction) > 1e-5:
        walking_direction = walking_direction / np.linalg.norm(walking_direction) 
        correction_quat = qbetween_np(walking_direction[np.newaxis, :], target_for_world)
        correction_quat_full = np.ones(h36m_curr.shape[:-1] + (4,)) * correction_quat
        h36m_curr = qrot_np(correction_quat_full, h36m_curr)
    else:
        correction_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

    R_face_z_direction = quat_to_scipy(root_quat_init)
    R_veering = quat_to_scipy(correction_quat)
    R_global_static = R_veering * R_face_z_direction 

    pose_aligned = pose_world.copy()

    for i in range(T):
        R_i = R_global_static * R.from_matrix(rot_mats_slope[i])
        root_rot = R.from_rotvec(pose_aligned[i, 0, :])
        new_root_rot = R_i * root_rot
        pose_aligned[i, 0, :] = new_root_rot.as_rotvec()

    global_orient = torch.tensor(pose_aligned[:, 0:1, :], dtype=torch.float32).reshape(T, -1).to(_DEVICE)
    body_pose = torch.tensor(pose_aligned[:, 1:24, :], dtype=torch.float32).reshape(T, -1).to(_DEVICE)
    betas = torch.zeros((T, 10), dtype=torch.float32).to(_DEVICE)
    zero_pose = torch.zeros((T, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand = torch.zeros((T, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_exp = torch.zeros((T, 10), dtype=torch.float32).to(_DEVICE)
    
    out_aligned = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                             jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                             left_hand_pose=zero_hand, right_hand_pose=zero_hand, expression=zero_exp)
                             
    local_h36m = vertices2joints(h36m_regressor, out_aligned.vertices).cpu().detach().numpy()
    trans_aligned = h36m_curr[:, 0, :] - local_h36m[:, 0, :]

    return pose_aligned, trans_aligned

def compute_datasets(cfg):
    """Processes raw SMPL sequences and saves them in both 3D and 6D formats simultaneously."""
    out_3d_path = cfg.OUT_PATH / '3D_SMPL' / 'PD-GaM_3D_SMPL_rot_trans_canonical.npz'
    out_6d_path = cfg.OUT_PATH / '6D_SMPL' / 'PD-GaM_6D_SMPL_rot_trans_canonical.npz'
    out_3d_path.parent.mkdir(parents=True, exist_ok=True)
    out_6d_path.parent.mkdir(parents=True, exist_ok=True)
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    
    result_3d = dict()
    result_6d = dict()
    
    for subject_id in tqdm(all_smpls, desc="Building Unified Datasets"):
        for walk_id in all_smpls[subject_id]:
            smpl_data = all_smpls[subject_id][walk_id]
            
            if smpl_data['pose'].shape[0] < 30 or 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                walk_name = f"{subject_id}__{walk_id}" if down_sample_rate == 1 else f"{subject_id}__{walk_id}_down{down}"

                out_world, pose_world, _ = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down)
                h36m_joints_world = vertices2joints(h36m_regressor, out_world.vertices).cpu().detach().numpy()
                
                # Retrieve canonical 3D representation
                pose_3d, trans_aligned = canonicalize_smpl_sequence(
                    pose_world, h36m_joints_world, smpl_model, h36m_regressor
                )
                
                if pose_3d.shape[0] >= 30:
                    # Save 3D Format
                    result_3d[walk_name] = pose_3d      
                    result_3d[f"{walk_name}_trans"] = trans_aligned
                    
                    # Convert to 6D continuous rotations (T, 24, 6) WITHOUT 25th padding joint
                    pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(torch.tensor(pose_3d))).cpu().numpy()
                    
                    result_6d[walk_name] = pose_6d
                    result_6d[f"{walk_name}_trans"] = trans_aligned
            
    np.savez(out_3d_path, **result_3d)
    np.savez(out_6d_path, **result_6d)
    print(f"Successfully generated {out_3d_path} and {out_6d_path}.")

def validate_canonicalization(gen_3d_path, gen_6d_path, num_samples=100):
    """
    Validates the newly generated 3D and 6D datasets directly against the 
    known sources of truth to guarantee mathematical parity.
    """
    print("\nValidating Generated Datasets")
    
    gt_3d_path = Path("thesis/data/raw/PD-GaM/3D_SMPL/PD-GaM_3D_SMPL_rot_trans_canonical.npz")
    gt_6d_path = Path("thesis/data/raw/PD-GaM/6D_SMPL/PD-GaM_6D_SMPL_rot_trans_canonical.npz")

    for rep, gen_path, gt_path in [("3D", gen_3d_path, gt_3d_path), ("6D", gen_6d_path, gt_6d_path)]:
        if not gt_path.exists():
            print(f"[!] Warning: Source of truth not found at {gt_path}. Skipping {rep} validation.")
            continue
            
        print(f"\nLoading {rep} datasets into memory...")
        gen_data = np.load(gen_path, allow_pickle=True)
        gt_data = np.load(gt_path, allow_pickle=True)
        
        all_keys = [k for k in gen_data.files if not k.endswith('_trans')]
        
        if num_samples is None or num_samples == -1:
            sample_keys = all_keys
            print(f"Validating the ENTIRE {rep} dataset ({len(sample_keys)} sequences)...")
        else:
            num_samples_actual = min(num_samples, len(all_keys))
            import random
            sample_keys = random.sample(all_keys, num_samples_actual)
            print(f"Validating {num_samples_actual} randomly selected {rep} sequences...")
            
        total_pose_mae, total_trans_mae = [], []
        skipped = 0
        
        for key in tqdm(sample_keys, desc=f"Evaluating {rep} Math"):
            if key not in gt_data:
                skipped += 1
                continue
                
            pose_gen = gen_data[key]
            trans_gen = gen_data[key + "_trans"]
            pose_gt = gt_data[key]
            trans_gt = gt_data[key + "_trans"]
            
            # Align lengths (handles any minor frame slicing differences)
            min_frames = min(pose_gen.shape[0], pose_gt.shape[0])
            pose_gen, trans_gen = pose_gen[:min_frames], trans_gen[:min_frames]
            pose_gt, trans_gt = pose_gt[:min_frames], trans_gt[:min_frames]

            # Drop the legacy 25th padding joint from the source of truth if it exists
            if pose_gt.shape[-2] == 25:
                pose_gt = pose_gt[..., :24, :]
            
            # Direct element-wise Mean Absolute Error
            pose_mae = np.mean(np.abs(pose_gen - pose_gt))
            trans_mae = np.mean(np.abs(trans_gen - trans_gt))
            
            total_pose_mae.append(pose_mae)
            total_trans_mae.append(trans_mae)
            
        if not total_pose_mae:
            print(f"[!] Validation failed for {rep}: No keys matched between datasets.")
            continue
            
        overall_pose_mae = np.mean(total_pose_mae)
        overall_trans_mae = np.mean(total_trans_mae)
        
        print(f"\n--- Aggregated Validation Results for {rep} ({len(total_pose_mae)} sequences) ---")
        print(f"Sequences Skipped: {skipped}")
        print(f"Pose MAE:          {overall_pose_mae:.8f}")
        print(f"Trans MAE:         {overall_trans_mae:.8f}")
        
        if overall_pose_mae < 1e-3 and overall_trans_mae < 1e-3:
            print(f"SUCCESS: {rep} generation perfectly matches the source of truth!")
        else:
            print(f"FAIL: {rep} geometric error exceeds floating-point tolerance.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SMPL Canonicalization (3D and 6D).")
    parser.add_argument("-i", "--input", type=str, default="thesis/data/raw/PD-GaM/PD-GaM.pkl",
                        help="Path to the raw input SMPL .pkl file.")
    parser.add_argument("-o", "--output_dir", type=str, default="thesis/data/raw/PD-GaM/temp_validation",
                        help="Safe directory to output the generated NPZ files.")
    parser.add_argument("--skip_val", action="store_true", help="Skip the validation check.")
    parser.add_argument("-n", "--num_samples", type=int, default=-1, 
                        help="Number of sequences to validate (-1 for all).")
    args = parser.parse_args()
    
    cfg = SimpleNamespace()
    cfg.H36M_J_REG = Path('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = Path(args.input)
    cfg.OUT_PATH = Path(args.output_dir)
    cfg.fps = 30
    cfg.exfps = 30
    
    # Define expected output paths
    out_3d_path = cfg.OUT_PATH / '3D_SMPL' / 'PD-GaM_3D_SMPL_rot_trans_canonical.npz'
    out_6d_path = cfg.OUT_PATH / '6D_SMPL' / 'PD-GaM_6D_SMPL_rot_trans_canonical.npz'
    
    compute_datasets(cfg)
    
    if not args.skip_val:
        validate_canonicalization(out_3d_path, out_6d_path, num_samples=args.num_samples)
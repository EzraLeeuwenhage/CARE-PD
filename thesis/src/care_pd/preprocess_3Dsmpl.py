""" NEW PREPROCESSING LOGIC FOR 3D SMPL MODEL INPUT.

Converts raw SMPL sequences to Canonicalized 3D Axis-Angle sequences and validates them. 
"""

import os
import sys
import torch
import joblib
import numpy as np
import random
from pathlib import Path
from tqdm.auto import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thesis.src.care_pd.conversion_utils import (
    _DEVICE,
    generate_smpl_in_world,
    transform_seq_so_it_has_no_slope_h36m,
    qbetween_np,
    qrot_np,
    quat_to_scipy
)

def canonicalize_smpl_sequence(pose_world, h36m_joints_world, smpl_model, h36m_regressor):
    """
    Canonicalizes the SMPL sequence to a standard orientation and position in the world frame.
    Bases translations on the desired h36m outcome to ensure the same result as original CARE-PD data.
    """
    T = pose_world.shape[0]
    
    # H36M slope correction
    h36m_curr, rot_mats_slope = transform_seq_so_it_has_no_slope_h36m(
        h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial=4, return_rot_matrices=True
    )

    # H36M alignment of head/chest to global z-direction
    floor_height = h36m_curr.min(axis=0).min(axis=0)[1]
    h36m_curr[:, :, 1] -= floor_height

    root_pos_init = h36m_curr[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    h36m_curr = h36m_curr - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = 1, 4, 14, 11 # Hardcoded H36M indices
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
    middle_frame_idx = T // 2
    middle_frame = h36m_curr[middle_frame_idx, 0]

    walking_direction = middle_frame - first_frame
    walking_direction[1] = 0 
    if np.linalg.norm(walking_direction) > 1e-5:
        walking_direction = walking_direction / np.linalg.norm(walking_direction) 
        correction_quat = qbetween_np(walking_direction[np.newaxis, :], target_for_world)
        correction_quat_full = np.ones(h36m_curr.shape[:-1] + (4,)) * correction_quat
        h36m_curr = qrot_np(correction_quat_full, h36m_curr)
    else:
        correction_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

    # Apply H36M corrections to ROOT JOINT orientation
    R_face_z_direction = quat_to_scipy(root_quat_init)
    R_veering = quat_to_scipy(correction_quat)
    R_global_static = R_veering * R_face_z_direction 

    pose_aligned = pose_world.copy()

    for i in range(T):
        R_i = R_global_static * R.from_matrix(rot_mats_slope[i])
        root_rot = R.from_rotvec(pose_aligned[i, 0, :])
        new_root_rot = R_i * root_rot
        pose_aligned[i, 0, :] = new_root_rot.as_rotvec()

    # Compute exact location of h36m pelvis by regressing SMPL model to h36m data
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


def compute_3D_canonical_representation(cfg):
    """ Processes raw SMPL sequences and saves them natively in 3D axis-angle space. """
    cfg.OUT_PATH_f = cfg.OUT_PATH / f'PD-GaM_3D_SMPL_rot_trans_canonical.npz'
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    print(f'Number of subjects: {len(all_smpls)}')
    
    result = dict()
    for subject_id in tqdm(all_smpls, desc="Canonicalizing 3D SMPL Data"):
        for walk_id in all_smpls[subject_id]:
            smpl_data = all_smpls[subject_id][walk_id]
            
            if smpl_data['pose'].shape[0] < 30 or 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                walk_name = f"{subject_id}__{walk_id}" if down_sample_rate == 1 else f"{subject_id}__{walk_id}_down{down}"

                out_world, pose_world, _ = generate_smpl_in_world(
                    smpl_model, smpl_data, down_sample_rate, down
                )
                
                h36m_joints_world = vertices2joints(h36m_regressor, out_world.vertices).cpu().detach().numpy()
                
                # Apply sequence normalization without converting to 6D
                pose_world_aligned, trans_world_aligned = canonicalize_smpl_sequence(
                    pose_world, h36m_joints_world, smpl_model, h36m_regressor
                )
                
                if pose_world_aligned.shape[0] >= 30:
                    result[walk_name] = pose_world_aligned      # Shape: (T, 24, 3)
                    result[f"{walk_name}_trans"] = trans_world_aligned
                else:
                    print(f"Discarding {walk_name} due to insufficient frames.")
            
    np.savez(cfg.OUT_PATH_f, **result)
    print(f"Saved {cfg.OUT_PATH_f} with {len(result)//2} valid 3D SMPL sequences.")
    return cfg.OUT_PATH_f


def forward_3d_to_h36m(pose_3d, trans, smpl_model, h36m_regressor, device):
    """
    Converts 3D Axis-Angle SMPL representations to H36M coordinates for validation.
    pose_3d: (T, 24, 3)
    trans: (T, 3)
    """
    T = pose_3d.shape[0]
    
    global_orient = torch.tensor(pose_3d[:, 0:1, :], dtype=torch.float32).reshape(T, -1).to(device)
    body_pose     = torch.tensor(pose_3d[:, 1:24, :], dtype=torch.float32).reshape(T, -1).to(device)
    world_trans_t = torch.tensor(trans, dtype=torch.float32).to(device)
    
    betas = torch.zeros((T, 10), dtype=torch.float32).to(device)
    zero_pose = torch.zeros((T, 3), dtype=torch.float32).to(device)
    zero_hand = torch.zeros((T, 15, 3), dtype=torch.float32).to(device)

    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                         jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                         left_hand_pose=zero_hand, right_hand_pose=zero_hand,
                         expression=betas)
        
        vertices_world = out.vertices + world_trans_t[:, None, :]
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
        
    return h36m_joints.cpu().numpy()


def validate_canonicalization(cfg, generated_npz_path, num_samples=100):
    """ Validates that the saved 3D SMPL arrays reconstruct the correct Canonical H36M data. """
    print("\n--- Validating 3D SMPL to H36M Mapping ---")
    truth_h36m_path = Path("thesis/data/raw/PD-GaM/h36m/PD-GaM_h36m_3d_world_no_slope_no_veering.npz")
    
    if not truth_h36m_path.exists():
        print(f"[!] Warning: Ground Truth file not found at {truth_h36m_path}.")
        print("Please ensure your preprocessed H36M source of truth exists, or update the path in this script.")
        return

    print("Loading datasets into memory...")
    npz_data_3d = np.load(generated_npz_path, allow_pickle=True)
    npz_data_true_h36m = np.load(truth_h36m_path, allow_pickle=True)
    
    all_keys = [k for k in npz_data_3d.files if not k.endswith('_trans')]

    if num_samples is None or num_samples == -1:
        sample_keys = all_keys
        print(f"\nValidating the ENTIRE dataset ({len(sample_keys)} sequences)...")
    else:
        num_samples = min(num_samples, len(all_keys))
        sample_keys = random.sample(all_keys, num_samples)
        print(f"\nValidating {num_samples} randomly selected sequences...")
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    total_mae = []
    max_errors = []
    skipped = 0
    
    for key in tqdm(sample_keys, desc="Evaluating Math"):
        if key not in npz_data_true_h36m:
            skipped += 1
            continue
            
        pose_3d = npz_data_3d[key]                 # (T, 24, 3)
        trans = npz_data_3d[key + "_trans"]        # (T, 3)
        h36m_true = npz_data_true_h36m[key]        # (T, 17, 3)
        
        # Make sure shapes align (downsampled data logic check)
        min_frames = min(pose_3d.shape[0], h36m_true.shape[0])
        pose_3d = pose_3d[:min_frames]
        trans = trans[:min_frames]
        h36m_true = h36m_true[:min_frames]
        
        # Map back to H36M
        h36m_hat = forward_3d_to_h36m(pose_3d, trans, smpl_model, h36m_regressor, _DEVICE)
        
        # Calculate coordinate differences in meters
        mae = np.mean(np.abs(h36m_hat - h36m_true))
        max_err = np.max(np.abs(h36m_hat - h36m_true))
        
        total_mae.append(mae)
        max_errors.append(max_err)

    if not total_mae:
        print("[!] Validation failed: No keys matched between the two datasets.")
        return

    overall_mean_mae = np.mean(total_mae)
    overall_max_error = np.max(max_errors)
    
    print(f"\n--- Aggregated Validation Results ({len(total_mae)} sequences) ---")
    print(f"Sequences Skipped (Not Found):      {skipped}")
    print(f"Overall H36M MAE (meters):          {overall_mean_mae:.8f}")
    print(f"Absolute Worst-Case Max Error:      {overall_max_error:.8f}")
    
    # 1e-4 meters = 0.1 millimeters tolerance for floating point calculations
    if overall_mean_mae < 1e-4 and overall_max_error < 5e-3:
        print("\nSUCCESS: 3D Axis-Angle mapping perfectly preserves the H36M canonicalization!")
    else:
        print("\nFAIL: Geometric error exceeds floating-point tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL sequences to 3D Canonical Axis-Angle format.")
    parser.add_argument("-i", "--input", type=str, default="thesis/data/raw/PD-GaM/PD-GaM.pkl")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--skip_val", action="store_true", help="Skip the H36M validation check.")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    cfg = SimpleNamespace()
    
    cfg.H36M_J_REG = Path('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = input_path
    
    if args.output:
        cfg.OUT_PATH = Path(args.output)
    else:
        cfg.OUT_PATH = input_path.parent / '3D_SMPL'
        
    cfg.db = 'PD-GaM'     
    cfg.exfps = 30
    cfg.fps = 30
    
    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    generated_path = compute_3D_canonical_representation(cfg)
    
    if not args.skip_val:
        validate_canonicalization(cfg, generated_path, num_samples=None)
""" TODO: cite care pd repo for original code, changes marked with 'Adapted' """

import os
import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thesis.src.care_pd.conversion_utils import (
    _DEVICE,
    generate_smpl_in_world,
    transform_seq_so_it_has_no_slope_h36m,
    get_6D_rep_from_24x3_pose,
    qbetween_np,
    qrot_np,
    quat_to_scipy
)

def canonicalize_smpl_sequence(pose_world, h36m_joints_world, smpl_model, h36m_regressor):
    """
    Canonicalizes the SMPL sequence to a standard orientation and position in the world frame.
    
    Bases translations on the desired h36m outcome to ensure the same result as original CARE-PD data
    translations directly applied on the h36m data. 
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
        # Identity quaternion (w, x, y, z) if no veering
        correction_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

    # Apply H36M corrections to ROOT JOINT orientation
    R_face_z_direction = quat_to_scipy(root_quat_init)
    R_veering = quat_to_scipy(correction_quat)
    R_global_static = R_veering * R_face_z_direction 

    pose_aligned = pose_world.copy()

    for i in range(T):
        # Compute total rotation matrix for frame `i` slope correction
        R_i = R_global_static * R.from_matrix(rot_mats_slope[i])

        # apply the rotation to the root joint at this frame
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
    
    # Run SMPL with zero global translation
    out_aligned = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                             jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                             left_hand_pose=zero_hand, right_hand_pose=zero_hand, expression=zero_exp)
                             
    local_h36m = vertices2joints(h36m_regressor, out_aligned.vertices).cpu().detach().numpy()
    
    # The required global translation is exactly the difference between the target pelvis and the local pelvis
    trans_aligned = h36m_curr[:, 0, :] - local_h36m[:, 0, :]

    return pose_aligned, trans_aligned


def compute_6D_motionclip_representation_from_pkl_SMPL_params(cfg):
    cfg.OUT_PATH_f = cfg.OUT_PATH / f'PD-GaM_6D_SMPL_rot_trans_canonical.npz'
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    print(f'Number of walks: {len(all_smpls)}')
    
    result = dict()
    for subject_id in tqdm(all_smpls):
        for walk_id in all_smpls[subject_id]:
            smpl_data = all_smpls[subject_id][walk_id]
            
            if smpl_data['pose'].shape[0] < 30:
                continue
            if 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                if down_sample_rate == 1: 
                    walk_name = str(subject_id) + '__' + str(walk_id)   
                else:
                    walk_name = str(subject_id) + '__' + str(walk_id) + f'_down{down}' 

                out_world, pose_world, _ = generate_smpl_in_world(
                    smpl_model, smpl_data, down_sample_rate, down
                ) # (num_frames, 24, 3) axis-angle per frame
                vertices_world = out_world.vertices # (n_frames, n_vertices, 3)
                
                h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
                pose_world_aligned, trans_world_aligned = canonicalize_smpl_sequence(
                    pose_world, h36m_joints_world, smpl_model, h36m_regressor
                )

                pose6d = get_6D_rep_from_24x3_pose(torch.tensor(pose_world_aligned)) # shape (T, 25, 6)
                
                if pose6d.shape[0] >= 30:
                    result[walk_name] = pose6d
                    result[f"{walk_name}_trans"] = trans_world_aligned
                else:
                    print(f"Discarding {walk_name} because it is less than 30 frames {pose6d.shape[0]}")
            
    np.savez(cfg.OUT_PATH_f, **result)
    print(f"Saved {cfg.OUT_PATH_f} with {len(result)} sequences: {len(result)/2} rotations {len(result)/2} trans seqs")
     
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL .pkl sequences to 6D .npz format.")
    parser.add_argument("-i", "--input", type=str, default="thesis/data/raw/PD-GaM/PD-GaM.pkl",
                        help="Path to the input SMPL .pkl file.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to the output directory. Defaults to two levels up + /6D_SMPL.")
    
    args = parser.parse_args()
    print(args) 
    
    input_path = Path(args.input)
    cfg = SimpleNamespace()
    
    # Path setup
    cfg.H36M_J_REG = Path('./thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('./thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = input_path
    
    if args.output:
        cfg.OUT_PATH = Path(args.output)
    else:
        cfg.OUT_PATH = input_path.parent.parent / '6D_SMPL'
        
    print(f"Input Data: {cfg.DATA_DIR}")
    print(f"Output Dir: {cfg.OUT_PATH}")
    
    # HARDCODED logic for PD-GaM
    cfg.db = 'PD-GaM'     
    cfg.exfps = 30
    cfg.fps = 30
    
    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    compute_6D_motionclip_representation_from_pkl_SMPL_params(cfg)
""" TODO: cite care pd repo for original code, changes marked with 'Adapted' """

# patch deprecated 'chumpy' package for compatibility with python 3.11+
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str


import os
import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# TODO: remove dependencies of CARE-PD repo, so thesis code is run independently
from const.const import _DEVICE, SUPPORTED_DATASETS, DATASET_ORIGINAL_FPS
from const import path
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

""" 
Slope correction logic from CARE-PD repo
TODO: Add proper citation for CARE-PD repo and authorship of slope correction code.
 """
def detect_standing_still(seq):
    sacrum_locs = seq[:, 0, :]
    sdiff = np.diff(sacrum_locs, axis=0)
    sdiff = np.pad(sdiff, ((0,1),(0,0)), mode='edge') # repeat last value of sdiff one more time so it's the same size as seq
    sdiff = np.linalg.norm(sdiff, axis=1)
    standing_still_mask = sdiff < 0.007 # I found this threshold from watching the values of real sequences
    return standing_still_mask

def get_hip_vec_estimate_h36m(pose):
    return pose[1, :] - pose[4, :]

def get_height_vec_estimate_h36m(pose):
    lank = pose[6, :]
    rank = pose[3, :]
    lknee = pose[5, :]
    rknee = pose[2, :]
    height_vector = pose[0, :] - np.mean([lank, rank, lknee, rknee], axis=0)
    return height_vector

def get_perpendicular_in_dir_of_nose(vec1, vec2, pose):
    res = np.cross(vec1,vec2)
    dir_of_nose = pose[9, :] - np.mean([1.5*pose[8, :], 0.5*pose[10, :]], axis=0)
    if np.dot(res, dir_of_nose) < 0: res *= -1
    return res / np.linalg.norm(res) # vector of length 1

def get_rotation_matrix(vec1, vec2=np.array(([[1, 0, 0], [0, 0, 1], [0, 1, 0]])), weights=[0.8, 1, 0.6]):
    """get rotation matrix between two sets of vectors using scipy vector sets of shape N x 3"""
    r = R.align_vectors(vec2, vec1, weights=weights)
    return r[0].as_matrix()

def transform_seq_so_it_has_no_slope_h36m(seq, n_frames_est_mov_dir=15, window_size=90, polynomial=4, return_rot_matrices=False):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_h36m(pose)
        height_vector  = get_height_vec_estimate_h36m(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_nose(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        if standing_still_mask[i] or \
           np.linalg.norm(m_dir[i]) < n*0.004 or \
           i >= len(m_dir) - n: 
            m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    centered_seq_rotated = []
    rotation_matrices = []
    for i,mov_dir_vec in enumerate(movement_direction):
        mov_dir_vec_in_xz_plane = mov_dir_vec.copy()
        mov_dir_vec_in_xz_plane[1] = 0
        mov_dir_vec_in_xz_plane /= np.linalg.norm(mov_dir_vec_in_xz_plane)

        to_align_with = np.stack([mov_dir_vec_in_xz_plane, hip_vectors[i], [0, 1, 0]])
        rot_mat = get_rotation_matrix(
            vec1=np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]),
            vec2=to_align_with,
            weights=[1, np.inf, 1]
        )

        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        offset = seq[i, 0, :]
        offset[1] = 0
        rotated += offset
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        rotation_matrices.append(rot_mat)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    if return_rot_matrices:
        return centered_seq_rotated, np.stack(rotation_matrices)
    else:
        return centered_seq_rotated

def generate_smpl_in_world(smpl_model, sequence, down_sample_rate, down):
    frame_number = sequence['pose'].shape[0]
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)
    if betas.shape[0] != frame_number:
        betas = np.tile(betas, (frame_number, 1))
        

    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)  # (num_frames, 1, 3)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)  # (num_frames, 23, 3)
    betas         = torch.tensor(betas, dtype=torch.float32)  # (num_frames, 10)
    
    if down_sample_rate > 1:
        global_orient = global_orient[down::down_sample_rate,...]  # (num_frames, 24, 3)  # start from down and they select every down_sample_rate
        body_pose     = body_pose[down::down_sample_rate,...]  # (num_frames, 10)
        betas         = betas[down::down_sample_rate,...] 
        world_trans         = world_trans[down::down_sample_rate,...]
        frame_number  = body_pose.shape[0]

    # Ensure everything is on the same device
    global_orient = global_orient.reshape(frame_number, -1).to(_DEVICE)
    body_pose = body_pose.reshape(frame_number, -1).to(_DEVICE)
    betas = betas.reshape(frame_number, -1).to(_DEVICE)
    world_trans = torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE)  # Ensure on same device

    # Zero values for face, hands, and expression
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(_DEVICE)

    # Generate SMPL output
    out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, 
                     jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                     left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose,
                     expression=zero_expression)

    # Apply global translation (world_trans) to the output vertices
    out.vertices += world_trans[:, None, :]  # Broadcasting (num_frames, 1, 3) to (num_frames, num_vertices, 3)

    return out

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()

def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)

def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1, dim=-1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))

def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


""" 
Adapted code, based on CARE-PD repo, for converting SMPL data to H36M format.
 """
def main_world_only(cfg):
    """Streamlined function to extract 3D world coordinates and skip all camera/image projections."""
    if cfg.slope_correction:
        ext = '_slopeCorrected'
    else:
        ext = ''
        
    base_name = cfg.DATA_DIR.stem
    cfg.OUT_PATH_world = cfg.OUT_PATH / f'{base_name}_3d_world{ext}.npz'
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    result_world = dict()
    
    for subject_id in tqdm(all_smpls, desc=f"Converting {base_name} to 3D World Coords"):
        for walk_id in all_smpls[subject_id]:
            smpl_data = all_smpls[subject_id][walk_id]
            if 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                walk_name = f"{subject_id}__{walk_id}" if down_sample_rate == 1 else f"{subject_id}__{walk_id}_down{down}"
                if smpl_data['pose'].shape[0] < 30:
                    print(f"Discarding {walk_name} because it is less than 30 frames {smpl_data['pose'].shape[0]}")
                    continue
                    
                out_world = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down)
                vertices_world = out_world.vertices 
                h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
                
                # TODO: always apply this? 
                if cfg.slope_correction:
                    h36m_joints_world = transform_seq_so_it_has_no_slope_h36m(h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial = 4)

                '''Put on Floor'''
                floor_height = h36m_joints_world.min(axis=0).min(axis=0)[1]
                h36m_joints_world[:, :, 1] -= floor_height
                
                '''XZ at origin'''
                root_pos_init = h36m_joints_world[0]
                root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
                h36m_joints_world = h36m_joints_world - root_pose_init_xz
                
                '''All initially face Z+'''
                r_hip, l_hip, sdr_r, sdr_l = cfg.face_joint_indx
                across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
                across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
                across = across1 + across2
                across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
                forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
                forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
                
                target_for_world = np.array([[0, 0, 1]])
                root_quat_init = qbetween_np(forward_init, target_for_world)
                root_quat_init = np.ones(h36m_joints_world.shape[:-1] + (4,)) * root_quat_init
                h36m_joints_world = qrot_np(root_quat_init, h36m_joints_world)
                
                # removed this: in CARE-PD repo it's only done for cam img, not for world
                # '''Correct for curved walking direction'''
                # if cfg.db in ['T-SDU-PD', 'PD-GaM']:
                #     first_frame = h36m_joints_world[0, 0]
                #     middle_frame_idx = h36m_joints_world.shape[0] // 2
                #     middle_frame = h36m_joints_world[middle_frame_idx, 0]
                    
                #     walking_direction = middle_frame - first_frame
                #     walking_direction[1] = 0 
                #     if np.linalg.norm(walking_direction) > 1e-5:
                #         walking_direction = walking_direction / np.linalg.norm(walking_direction) 
                #         correction_quat = qbetween_np(walking_direction[np.newaxis, :], target_for_world)
                #         correction_quat = np.ones(h36m_joints_world.shape[:-1] + (4,)) * correction_quat
                #         h36m_joints_world = qrot_np(correction_quat, h36m_joints_world)
                
                result_world[walk_name] = h36m_joints_world

    np.savez(cfg.OUT_PATH_world, **result_world)
    return cfg.OUT_PATH_world

def convert_smpl_to_h36m(input_filename, output_dir=None):
    """Wrapper for SMPL to H36M conversion.
    
    Expects input_filename to be a full path (e.g., thesis/data/processed/PD-GaM/SMPL/file.pkl)
    """
    input_path = Path(input_filename)
    cfg = SimpleNamespace()
    
    try:
        cfg.BASE_DIR = Path(path.PROJECT_ROOT)
    except NameError:
        cfg.BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        
    cfg.H36M_J_REG = Path('./data/preprocessing/common/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('./data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = input_path
    
    # Use provided output directory, or default to the relative ../../h36m path
    if output_dir:
        cfg.OUT_PATH = Path(output_dir)
    else:
        cfg.OUT_PATH = input_path.parent.parent / 'h36m' 
    
    print(f"Input Data: {cfg.DATA_DIR}")
    print(f"Output Dir: {cfg.OUT_PATH}")
    
    # HARDCODED logic for PD-GaM (change for other datasets if needed)
    cfg.db = 'PD-GaM' 
    cfg.slope_correction = False
    cfg.exfps = 30
    cfg.fps = 30
    
    # H36M Face Index Format
    cfg.face_joint_indx = [1, 4, 14, 11]
        
    cfg.H = 1000
    cfg.W = 1000
    
    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    out_path = main_world_only(cfg)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL .pkl sequences to H36M .npz format.")
    parser.add_argument("-i", "--input", type=str, default="thesis/data/raw/PD-GaM/PD-GaM.pkl",
                        help="Path to the input SMPL .pkl file.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to the output directory. Defaults to two levels up + /h36m.")
    
    args = parser.parse_args()
    
    convert_smpl_to_h36m(input_filename=args.input, output_dir=args.output)
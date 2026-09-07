import numpy as np
import torch
from smplx.lbs import vertices2joints
from scipy.spatial.transform import Rotation as R
from thesis.src.care_pd.conversion_utils import axis_angle_to_matrix

def gram_schmidt_to_rmat(pose_tensor):
    """
    Gram-Schmidt to convert 6d pose tensors to (..., 3, 3) rotation matrices.
    Constructs the rotation matrix by stacking rows due to original CARE-PD formatting.
    """
    if isinstance(pose_tensor, np.ndarray):
        pose_tensor = torch.tensor(pose_tensor, dtype=torch.float32)

    v1 = pose_tensor[..., :3]
    v2 = pose_tensor[..., 3:]
    
    x = torch.nn.functional.normalize(v1, dim=-1)
    y_raw = v2 - (torch.sum(x * v2, dim=-1, keepdim=True) * x)
    y = torch.nn.functional.normalize(y_raw, dim=-1)
    z = torch.cross(x, y, dim=-1)
    
    # Stack into (..., 3, 3) rotation matrices
    # and handle x,y,z as rows because CARE-PD formatted 6D rotations as first 2 rows
    return torch.stack([x, y, z], dim=-2)

def convert_6d_to_smpl(pose_6d_tensor):
    """
    Converts 6D continuous rotations back to 3D Axis-Angle representations.
    Dynamically handles batch dimensions (e.g., T, 24, 6 or B, T, 24, 6).
    """
    if isinstance(pose_6d_tensor, np.ndarray):
        pose_6d_tensor = torch.tensor(pose_6d_tensor, dtype=torch.float32)
        
    original_shape = pose_6d_tensor.shape
    rot_mats = gram_schmidt_to_rmat(pose_6d_tensor)
    
    rot_mats_flat = rot_mats.reshape(-1, 3, 3).cpu().numpy()
    rotations = R.from_matrix(rot_mats_flat)
    axis_angles_flat = rotations.as_rotvec()
    
    # Dynamically reshape based on input dimensions (e.g., T, 24, 3 or B, T, 24, 3)
    smpl_pose = axis_angles_flat.reshape(*original_shape[:-1], 3)
    return smpl_pose

def pose_to_rmat(pose_tensor):
    """
    Dynamically infers the representation based on the last dimension size
    and converts either 3D axis-angle or 6D continuous rotations to rotation matrices.
    """
    if isinstance(pose_tensor, np.ndarray):
        pose_tensor = torch.tensor(pose_tensor, dtype=torch.float32)
        
    dim = pose_tensor.shape[-1]
    
    if dim == 3:
        return axis_angle_to_matrix(pose_tensor)
    elif dim == 6:
        return gram_schmidt_to_rmat(pose_tensor)
    else:
        raise ValueError(f"Expected last dimension to be 3 (axis-angle) or 6 (continuous), got {dim}")

def forward_to_h36m(pose_tensor, trans, smpl_model, h36m_regressor, device):
    """
    Directly converts a single sequence of poses (3D or 6D) and translations 
    to 3D H36M coordinates. Executes entirely in memory without intermediate files.
    """
    dim = pose_tensor.shape[-1]
    if dim == 6:
        smpl_pose = convert_6d_to_smpl(pose_tensor)
        if torch.is_tensor(smpl_pose):
            smpl_pose = smpl_pose.detach().cpu().numpy()
    elif dim == 3:
        smpl_pose = pose_tensor.detach().cpu().numpy() if torch.is_tensor(pose_tensor) else pose_tensor
    else:
        raise ValueError(f"Unknown pose dimension {dim}")
        
    T = smpl_pose.shape[0]
    
    # Extract global orientation and body pose, format for SMPL layer
    global_orient = torch.as_tensor(smpl_pose[:, 0:1, :], dtype=torch.float32, device=device).reshape(T, -1)
    body_pose     = torch.as_tensor(smpl_pose[:, 1:24, :], dtype=torch.float32, device=device).reshape(T, -1)
    world_trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    
    # Create neutral shape/expression placeholders
    betas = torch.zeros((T, 10)).to(device)
    zero_pose = torch.zeros((T, 3)).to(device)
    zero_hand = torch.zeros((T, 15, 3)).to(device)

    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                         jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                         left_hand_pose=zero_hand, right_hand_pose=zero_hand,
                         expression=betas)

        vertices_world = out.vertices + world_trans_t[:, None, :]
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
        
    return h36m_joints.cpu().numpy()

def batched_to_h36m(pose_tensor, trans, smpl_model, h36m_regressor, device, chunk_size=256):
    """Processes (Batch, Time, Joints, D) tensors efficiently in GPU chunks."""
    N, T, J, D = pose_tensor.shape
    out_h36m = np.zeros((N, T, 17, 3), dtype=np.float32)
    
    for i in range(0, N, chunk_size):
        p_chunk = pose_tensor[i:i+chunk_size]
        t_chunk = trans[i:i+chunk_size]
        B = p_chunk.shape[0]
        
        # Process the entire chunk simultaneously on the GPU
        h36m_flat = forward_to_h36m(
            p_chunk.reshape(B * T, J, D), 
            t_chunk.reshape(B * T, 3), 
            smpl_model, h36m_regressor, device
        )
        
        # Reshape back to individual sequences and store
        out_h36m[i:i+chunk_size] = h36m_flat.reshape(B, T, 17, 3)
        
    return out_h36m
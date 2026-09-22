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

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts (..., 3, 3) rotation matrices to (..., 4) unit quaternions [w, x, y, z] on GPU.
    Uses Shepperd's algorithm for numerical stability near singular boundaries.
    """
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22

    q0 = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    q1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 0.5
    q2 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) * 0.5
    q3 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) * 0.5

    cond0 = (trace >= m00) & (trace >= m11) & (trace >= m22)
    cond1 = (m00 >= m11) & (m00 >= m22)
    cond2 = m11 >= m22

    w = torch.where(cond0, q0, torch.where(cond1, (m21 - m12) / (4.0 * q1), 
                    torch.where(cond2, (m02 - m20) / (4.0 * q2), (m10 - m01) / (4.0 * q3))))
    x = torch.where(cond0, (m21 - m12) / (4.0 * q0), 
                    torch.where(cond1, q1, torch.where(cond2, (m01 + m10) / (4.0 * q2), (m02 + m20) / (4.0 * q3))))
    y = torch.where(cond0, (m02 - m20) / (4.0 * q0), 
                    torch.where(cond1, (m01 + m10) / (4.0 * q1), torch.where(cond2, q2, (m12 + m21) / (4.0 * q3))))
    z = torch.where(cond0, (m10 - m01) / (4.0 * q0), 
                    torch.where(cond1, (m02 + m20) / (4.0 * q1), torch.where(cond2, (m12 + m21) / (4.0 * q2), q3)))

    q = torch.stack([w, x, y, z], dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts (..., 4) unit quaternions [w, x, y, z] to (..., 3) axis-angle vectors on GPU.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch GPU conversion from (..., 3, 3) rotation matrices to (..., 3) axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(rot_mats))


def convert_6d_to_3d_axis_angle_smpl(pose_6d_tensor):
    """
    Converts 6D continuous rotations back to 3D Axis-Angle representations.
    Dynamically handles batch dimensions (e.g., T, 24, 6 or B, T, 24, 6) entirely on GPU.
    """
    is_numpy = isinstance(pose_6d_tensor, np.ndarray)
    if is_numpy:
        pose_6d_tensor = torch.tensor(pose_6d_tensor, dtype=torch.float32)

    rot_mats = gram_schmidt_to_rmat(pose_6d_tensor)
    smpl_pose = matrix_to_axis_angle(rot_mats)

    return smpl_pose.cpu().numpy() if is_numpy else smpl_pose

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
        # 6D conversion currently requires Numpy/Scipy, so go to CPU
        smpl_pose = torch.as_tensor(convert_6d_to_3d_axis_angle_smpl(pose_tensor), dtype=torch.float32, device=device)
    elif dim == 3:
        # remain on GPU for axis-angle to rotation matrix conversion
        smpl_pose = torch.as_tensor(pose_tensor, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown pose dimension {dim}")
        
    T = smpl_pose.shape[0]
    
    global_orient = smpl_pose[:, 0:1, :].reshape(T, -1)
    body_pose     = smpl_pose[:, 1:24, :].reshape(T, -1)
    world_trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    
    betas = torch.zeros((T, 10), device=device)
    zero_pose = torch.zeros((T, 3), device=device)
    zero_hand = torch.zeros((T, 15, 3), device=device)

    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                         jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                         left_hand_pose=zero_hand, right_hand_pose=zero_hand,
                         expression=betas)

        vertices_world = out.vertices + world_trans_t[:, None, :]
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
        
    return h36m_joints.cpu().numpy()

def batched_forward_to_h36m(pose_tensor, trans, smpl_model, h36m_regressor, device, chunk_frames=2048):
    """
    High-throughput batched forward mapping of SMPL poses and translations to H36M joints.
    Efficiently processes variable-length sequences using temporal concatenation and frame-budgeted GPU chunks.
    Accepts either lists of variable-length sequence tensors or a fixed 4D tensor (Batch, Time, Joints, D).
    """
    is_ragged = isinstance(pose_tensor, (list, tuple))

    if is_ragged:
        processed_poses = []
        processed_trans = []
        lengths = []
        for p, t in zip(pose_tensor, trans):
            p_t = torch.as_tensor(p, dtype=torch.float32)
            t_t = torch.as_tensor(t, dtype=torch.float32)

            # Squeeze leading singleton batch dimensions if present
            if p_t.ndim == 4 and p_t.shape[0] == 1:
                p_t = p_t.squeeze(0)
            if t_t.ndim == 3 and t_t.shape[0] == 1:
                t_t = t_t.squeeze(0)

            lengths.append(p_t.shape[0])
            processed_poses.append(p_t)
            processed_trans.append(t_t)

        pose_flat = torch.cat(processed_poses, dim=0)
        trans_flat = torch.cat(processed_trans, dim=0)
    else:
        pose_t = torch.as_tensor(pose_tensor, dtype=torch.float32)
        trans_t = torch.as_tensor(trans, dtype=torch.float32)
        N, T, J, D = pose_t.shape
        lengths = [T] * N
        pose_flat = pose_t.reshape(N * T, J, D)
        trans_flat = trans_t.reshape(N * T, 3)

    dim = pose_flat.shape[-1]
    if dim == 6:
        smpl_pose_flat = torch.as_tensor(convert_6d_to_3d_axis_angle_smpl(pose_flat), dtype=torch.float32, device=device)
    elif dim == 3:
        smpl_pose_flat = pose_flat.to(device)
    else:
        raise ValueError(f"Unknown pose dimension {dim}")

    trans_flat = trans_flat.to(device)
    total_frames = smpl_pose_flat.shape[0]
    h36m_chunks = []

    for start_idx in range(0, total_frames, chunk_frames):
        end_idx = min(start_idx + chunk_frames, total_frames)
        p_sub = smpl_pose_flat[start_idx:end_idx]
        t_sub = trans_flat[start_idx:end_idx]
        sub_T = p_sub.shape[0]

        global_orient = p_sub[:, 0:1, :].reshape(sub_T, -1)
        body_pose = p_sub[:, 1:24, :].reshape(sub_T, -1)

        betas = torch.zeros((sub_T, 10), device=device)
        zero_pose = torch.zeros((sub_T, 3), device=device)
        zero_hand = torch.zeros((sub_T, 15, 3), device=device)

        with torch.no_grad():
            out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                             jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                             left_hand_pose=zero_hand, right_hand_pose=zero_hand,
                             expression=betas)

            vertices_world = out.vertices + t_sub[:, None, :]
            h36m_joints = vertices2joints(h36m_regressor, vertices_world)

        h36m_chunks.append(h36m_joints.cpu().numpy())

    h36m_all = np.concatenate(h36m_chunks, axis=0) if h36m_chunks else np.empty((0, 17, 3), dtype=np.float32)

    if is_ragged:
        split_points = np.cumsum(lengths)[:-1]
        return np.split(h36m_all, split_points)
    else:
        return h36m_all.reshape(N, T, 17, 3)
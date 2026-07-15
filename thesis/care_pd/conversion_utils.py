"""TODO: Add proper citation for CARE-PD repo and authorship of code. """

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str

import torch
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################
# SMPL2H36M utils
######################
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


######################
# SMPL2sixD utils
######################
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def get_6D_rep_from_24x3_pose(pose):
    pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).detach().cpu().numpy()
    pose6d=np.pad(pose6d, ((0,0), (0,1), (0,0))) # Adding [0,0,0,0,0,0] for translation
    return pose6d

# Helper to convert custom Quats [w, x, y, z] to SciPy Rotations [x, y, z, w]
def quat_to_scipy(q):
    return R.from_quat([q[0, 1], q[0, 2], q[0, 3], q[0, 0]])

######################
# Shared utils
######################
def generate_smpl_in_world(smpl_model, sequence, down_sample_rate, down):
    """Adapted to fit both SMPL2 h36m and 6D conversion functionality."""
    frame_number = sequence['pose'].shape[0]

    if sequence['beta'].shape[0] != frame_number:
        sequence['beta'] = np.tile(sequence['beta'], (frame_number, 1))
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)

    pose_world    = pose_world[down::down_sample_rate, ...]  
    betas         = betas[down::down_sample_rate, ...] 
    world_trans   = world_trans[down::down_sample_rate, ...]
    frame_number = pose_world.shape[0]

    # keep copies of original pose for later use in canonicalization
    pose_world_out = pose_world.copy()
    world_trans_out = world_trans.copy()    

    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32).reshape(frame_number, -1).to(_DEVICE)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32).reshape(frame_number, -1).to(_DEVICE)
    betas         = torch.tensor(betas, dtype=torch.float32).reshape(frame_number, -1).to(_DEVICE)
    world_trans_t = torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE)

    # Zeros for face, hands, and expression joints
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(_DEVICE)

    # Generate SMPL output
    out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, 
                     jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                     left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose,
                     expression=zero_expression)

    # Apply global translation (world_trans) to the output vertices
    out.vertices += world_trans_t[:, None, :]  # Broadcasting (num_frames, 1, 3) to (num_frames, num_vertices, 3)

    return out, pose_world_out, world_trans_out
import torch


def generate_motion_prior_from_prefix(prefix_pose, prefix_trans, num_frames, prior_noise_scale=1.0, generator=None, use_cumsum=False):
    """
    Generates x_0 using an STFlow-inspired kinematic random walk.
    Translation uses full drift + noise. Pose uses zero-drift + noise.

    - use_cumsum=False: Stationary Gaussian noise (Option A, independent variance per frame).
    - use_cumsum=True:  Cumulative noise walk via cumsum() across frames.
    """
    device = prefix_trans.device
    batch_size = prefix_trans.shape[0]
    
    # Global root translation prior
    trans_velocity = prefix_trans[:, 1:, :] - prefix_trans[:, :-1, :]
    mu_trans = trans_velocity.mean(dim=1, keepdim=True)
    sigma_trans = trans_velocity.std(dim=1, keepdim=True) * prior_noise_scale
    sigma_trans = torch.nan_to_num(sigma_trans, 1e-4)
    
    time_steps = torch.arange(1, num_frames + 1, device=device, dtype=prefix_trans.dtype).view(1, -1, 1)
    
    trans_noise = torch.randn(batch_size, num_frames, 3, device=device, dtype=prefix_trans.dtype, generator=generator)
    if use_cumsum:
        trans_noise = torch.cumsum(trans_noise, dim=1)

    trans_0 = prefix_trans[:, -1:, :] + (time_steps * mu_trans) + (sigma_trans * trans_noise)
    
    # Joint rotation pose prior
    pose_vel = prefix_pose[:, 1:, :, :] - prefix_pose[:, :-1, :, :]
    sigma_pose = pose_vel.std(dim=1, keepdim=True) * prior_noise_scale
    sigma_pose = torch.nan_to_num(sigma_pose, 1e-4)
    
    # Dynamically extract spatial dimensions from the prefix tensor
    _, _, num_joints, pose_dim = prefix_pose.shape
    
    pose_noise = torch.randn(batch_size, num_frames, num_joints, pose_dim, device=device, dtype=prefix_pose.dtype, generator=generator)
    if use_cumsum:
        pose_noise = torch.cumsum(pose_noise, dim=1)

    pose_0 = prefix_pose[:, -1:, :, :] + (sigma_pose * pose_noise)
    
    return {
        'pose': pose_0,
        'trans': trans_0
    }

def generate_label_prior(batch_size, num_classes, device, generator=None):
    """Generates uniform random noise for categorical label state space."""
    return torch.randint(0, num_classes, (batch_size,), device=device, generator=generator)

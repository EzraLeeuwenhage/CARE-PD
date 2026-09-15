import sys
import subprocess
import torch
from pathlib import Path


def generate_motion_prior_from_prefix(prefix_pose, prefix_trans, num_frames, s_scale=1.0):
    """
    Generates x_0 using an STFlow-inspired kinematic random walk.
    Translation uses full drift + noise. Pose uses zero-drift + noise.
    """
    device = prefix_trans.device
    batch_size = prefix_trans.shape[0]
    
    # Global translation prior
    trans_velocity = prefix_trans[:, 1:, :] - prefix_trans[:, :-1, :]
    mu_trans = trans_velocity.mean(dim=1, keepdim=True)
    sigma_trans = trans_velocity.std(dim=1, keepdim=True) * s_scale
    sigma_trans = torch.nan_to_num(sigma_trans, 1e-4) 
    
    time_steps = torch.arange(1, num_frames + 1, device=device).view(1, -1, 1)
    z_trans = torch.randn(batch_size, num_frames, 3, device=device)
    random_walk_trans = torch.cumsum(z_trans, dim=1)
    trans_0 = prefix_trans[:, -1:, :] + (time_steps * mu_trans) + (sigma_trans * random_walk_trans)
    
    # Pose prior
    pose_vel = prefix_pose[:, 1:, :, :] - prefix_pose[:, :-1, :, :]
    sigma_pose = pose_vel.std(dim=1, keepdim=True) * s_scale
    sigma_pose = torch.nan_to_num(sigma_pose, 1e-4)
    
    # Dynamically extract spatial dimensions from the prefix tensor
    _, _, num_joints, pose_dim = prefix_pose.shape
    
    z_pose = torch.randn(batch_size, num_frames, num_joints, pose_dim, device=device)
    random_walk_pose = torch.cumsum(z_pose, dim=1)
    pose_0 = prefix_pose[:, -1:, :, :] + (sigma_pose * random_walk_pose)
    
    return {
        'pose': pose_0,
        'trans': trans_0
    }


if __name__ == "__main__":
    from thesis.src.utils.pipeline_utils import load_config
    from thesis.src.dataloader import get_dataloader
    from thesis.src.utils.smpl_io import save_smpl_pkl
    from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m

    print("Initializing Dataloader...")
    cfg = load_config("thesis/configs/baseline.yaml")
    loader = get_dataloader(cfg, mode='test')
    prefix, target, severity = next(iter(loader))
    
    # Extract just the first sample from the batch
    prefix_single = {
        'pose': prefix['pose'][0:1],
        'trans': prefix['trans'][0:1]
    }
    target_single = {
        'pose': target['pose'][0:1],
        'trans': target['trans'][0:1]
    }
    
    print("Generating Prior from Prefix...")
    x_0 = generate_motion_prior_from_prefix(prefix_single, target_single)
    
    # Concat true prefix with generated suffix
    full_seq_6d = torch.cat([prefix_single['pose'], x_0['pose']], dim=1)
    full_seq_trans = torch.cat([prefix_single['trans'], x_0['trans']], dim=1)
    
    print(f"   Full Sequence 6D Pose Shape:  {full_seq_6d.shape}")
    print(f"   Full Sequence Trans Shape:    {full_seq_trans.shape}")
    
    temp_pkl_path = "thesis/data/processed/test_gen_prior/SMPL/example_generated_prior.pkl"
    final_h36m_dir = "thesis/data/processed/test_gen_prior/h36m/"
    final_npz_path = Path(final_h36m_dir) / "example_generated_prior_h36m_3d_world.npz"
    
    print("\nConverting 6D -> SMPL (.pkl)...")
    save_smpl_pkl(
        generated_pose=full_seq_6d,
        generated_trans=full_seq_trans,
        output_filepath=temp_pkl_path,
        subject_id="TEST",
        walk_prefix="prior_walk"
    )
    
    print("\nConverting SMPL -> H36M (.npz)...")
    convert_smpl_to_h36m(input_filename=temp_pkl_path)
    
    print("\nMaking Visualization...")
    command = [
        sys.executable,
        "utility/viz_seqs.py",
        "-n", str(final_npz_path),
        "-f", "h36m"
    ]
    
    print(f"Executing: {' '.join(command)}\n")
    subprocess.run(command, check=True)
import sys
import subprocess
import torch
from pathlib import Path


def generate_prior_from_prefix(prefix_dict, target_dict, s_scale=1.0):
    """
    Generates x_0 using an STFlow-inspired kinematic random walk.
    Translation uses full drift + noise. Pose uses zero-drift + noise.
    """
    device = prefix_dict['trans'].device
    batch_size = prefix_dict['trans'].shape[0]
    target_frames = target_dict['trans'].shape[1]
    
    # 1) Global translation prior
    trans_prefix = prefix_dict['trans']
    trans_velocity = trans_prefix[:, 1:, :] - trans_prefix[:, :-1, :]
    mu_trans = trans_velocity.mean(dim=1, keepdim=True)
    sigma_trans = trans_velocity.std(dim=1, keepdim=True) * s_scale
    
    # handle edge case of zero variance
    sigma_trans = torch.nan_to_num(sigma_trans, 1e-4) 
    
    # compute random walk updates
    time_steps = torch.arange(1, target_frames + 1, device=device).view(1, -1, 1)
    z_trans = torch.randn(batch_size, target_frames, 3, device=device)
    rw_trans = torch.cumsum(z_trans, dim=1)
    trans_0 = trans_prefix[:, -1:, :] + (time_steps * mu_trans) + (sigma_trans * rw_trans)
    
    # 2) Pose prior
    pose_prefix = prefix_dict['pose']
    
    # compute variance in angular velocity
    pose_vel = pose_prefix[:, 1:, :, :] - pose_prefix[:, :-1, :, :]
    sigma_pose = pose_vel.std(dim=1, keepdim=True) * s_scale
    sigma_pose = torch.nan_to_num(sigma_pose, 1e-4)
    
    # compute random walk updates without drift
    z_pose = torch.randn_like(target_dict['pose'])
    rw_pose = torch.cumsum(z_pose, dim=1)
    pose_0 = pose_prefix[:, -1:, :, :] + (sigma_pose * rw_pose)
    
    return {
        'pose': pose_0,
        'trans': trans_0
    }


if __name__ == "__main__":
    from thesis.src.dataloader import get_dataloader
    from thesis.utils.sixD2smpl import build_smpl_pkl_from_6d_smpl
    from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m

    print("Initializing Dataloader...")
    loader = get_dataloader(config_path="thesis/configs/dataloader.yaml")
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
    x_0 = generate_prior_from_prefix(prefix_single, target_single)
    
    # Concat true prefix with generated suffix
    full_seq_6d = torch.cat([prefix_single['pose'], x_0['pose']], dim=1)
    full_seq_trans = torch.cat([prefix_single['trans'], x_0['trans']], dim=1)
    
    print(f"   Full Sequence 6D Pose Shape:  {full_seq_6d.shape}")
    print(f"   Full Sequence Trans Shape:    {full_seq_trans.shape}")
    
    temp_pkl_path = "thesis/data/processed/PD-GaM/SMPL/example_generated_prior.pkl"
    final_h36m_dir = "thesis/data/processed/PD-GaM/h36m/"
    final_npz_path = Path(final_h36m_dir) / "example_generated_prior__2_h36m_sideright_cam.npz"
    
    print("\nConverting 6D -> SMPL (.pkl)...")
    build_smpl_pkl_from_6d_smpl(
        generated_pose_6d=full_seq_6d, 
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
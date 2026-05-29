import sys
import subprocess
import yaml
import torch
import numpy as np
from pathlib import Path
from thesis.care_pd.smpl2h36m import convert_smpl_to_h36m
from thesis.data.raw.data_conversion_utils import build_smpl_pkl_from_6d_smpl
from thesis.src.model import FlowMatchingMLP
from thesis.src.dataloader import get_dataloader
from thesis.src.generate_prior import generate_prior_from_prefix


def load_config(config_path="thesis/configs/sample.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_generated_to_npz(full_seq_6d, full_seq_trans, output_dir, filename="generated_PD_walk.npz"):
    """Saves the generated 6D pose and translation tensors to given directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / filename

    pose_np = full_seq_6d.squeeze(0).cpu().detach().numpy()    # (Total_T, 24, 6)
    trans_np = full_seq_trans.squeeze(0).cpu().detach().numpy()  # (Total_T, 3)

    np.savez(
        full_path, 
        pose=pose_np, 
        trans=trans_np
    )
    
    print(f"\nSuccessfully saved generated sequence to: {full_path}")
    print(f"Saved Pose Shape:  {pose_np.shape}")
    print(f"Saved Trans Shape: {trans_np.shape}")


def euler_ode_solver(model, prefix_dict, x_0_dict, severity_score, num_steps=100):
    """Starts from the generated prior (x_0) and iteratively applies the model's 
    predicted velocity field to generate the target suffix (x_1).
    """
    device = prefix_dict['pose'].device
    batch_size = prefix_dict['pose'].shape[0]
    model.eval()
    
    x_t = {
        'pose': x_0_dict['pose'].clone(),
        'trans': x_0_dict['trans'].clone()
    }
    
    dt = 1.0 / num_steps
    print(f"\nStarting Euler ODE solver with {num_steps} steps...")
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            
            t_tensor = torch.full((batch_size, 1), t_val, device=device)
            velocity = model(x_t, prefix_dict, t_tensor, severity_score)
            
            # Euler step: x_{t+dt} = x_t + v * dt
            x_t['pose'] = x_t['pose'] + (velocity['pose'] * dt)
            x_t['trans'] = x_t['trans'] + (velocity['trans'] * dt)
            
            if (step + 1) % 20 == 0 or step == num_steps - 1:
                print(f"   Step {step + 1:03d}/{num_steps} (t={t_val + dt:.2f}) completed.")
                
    print("Generation complete! Reached t=1.0")
    return x_t


if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("\nInitializing Dataloader...")
    loader = get_dataloader(config_path=cfg['sampling']['dataloader_config'])
    prefix, target, severity = next(iter(loader))
    
    # Use single sample for this test
    prefix = {
        'pose': prefix['pose'][0:1].to(device),
        'trans': prefix['trans'][0:1].to(device)
    }
    target = {
        'pose': target['pose'][0:1].to(device),
        'trans': target['trans'][0:1].to(device)
    }
    severity_label = severity[0:1].to(device)
    
    # Generate Prior (x_0), get model and load model params
    print("\nGenerating Prior from Prefix...")
    x_0 = generate_prior_from_prefix(prefix, target)
    
    print("\nLoading Trained Model...")
    model = FlowMatchingMLP(config_path=cfg['sampling']['dataloader_config']).to(device)
    weights_path = Path(cfg['sampling']['weights_path'])
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))

    # Solve ODE
    print("\nGenerating Suffix via ODE Solver...")
    generated_suffix = euler_ode_solver(
        model=model, 
        prefix_dict=prefix, 
        x_0_dict=x_0, 
        severity_score=severity_label, 
        num_steps=cfg['sampling']['num_steps']
    )
    
    # Concat prefix and generated suffix
    print("\nConcatenating prefix and generated suffix...")
    full_seq_6d = torch.cat([prefix['pose'], generated_suffix['pose']], dim=1)
    full_seq_trans = torch.cat([prefix['trans'], generated_suffix['trans']], dim=1)
    print(f"   Full Sequence 6D Pose Shape:  {full_seq_6d.shape}")
    print(f"   Full Sequence Trans Shape:    {full_seq_trans.shape}")
    
    # Save to 6D_SMPL directory
    save_generated_to_npz(
        full_seq_6d=full_seq_6d, 
        full_seq_trans=full_seq_trans, 
        output_dir=cfg['sampling']['output_dir'],
        filename="example_generated_walk.npz"
    )

    temp_pkl_path = "thesis/data/processed/SMPL/example_generated_walk.pkl"
    final_h36m_dir = "thesis/data/processed/h36m/"
    final_npz_path = Path(final_h36m_dir) / "example_generated_walk__2_h36m_sideright_cam.npz"

    print("\nConverting 6D -> SMPL (.pkl)...")
    build_smpl_pkl_from_6d_smpl(
        generated_pose_6d=full_seq_6d, 
        generated_trans=full_seq_trans, 
        output_filepath=temp_pkl_path,
        subject_id="TEST",
        walk_prefix="generated_walk"
    )
    
    print("\nConverting SMPL -> H36M (.npz)...")
    convert_smpl_to_h36m(input_filename=temp_pkl_path)
    
    print("\nLaunching Visualization...")
    command = [
        sys.executable,
        "utility/viz_seqs.py",
        "-n", str(final_npz_path),
        "-f", "h36m"
    ]
    
    print(f"Executing: {' '.join(command)}\n")
    subprocess.run(command, check=True)
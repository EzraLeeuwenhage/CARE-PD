import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from thesis.src.generate_prior import generate_prior_from_prefix

def save_generated_to_npz(full_seq_6d, full_seq_trans, output_dir, filename="generated_PD_walk.npz"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / filename

    pose_np = full_seq_6d.squeeze(0).cpu().detach().numpy() # (Total_T, 24, 6)
    trans_np = full_seq_trans.squeeze(0).cpu().detach().numpy() # (Total_T, 3)

    np.savez(full_path, pose=pose_np, trans=trans_np)
    print(f"\nSuccessfully saved generated sequence to: {full_path}")
    print(f"Saved Pose Shape:  {pose_np.shape}")
    print(f"Saved Trans Shape: {trans_np.shape}")

@torch.no_grad()
def generate_trajectories(model, dataloader, num_steps, device, max_batches=-1, desc="Generating"):
    """Generates synthetic dataset using model and dataloader."""
    model.eval()
    
    all_gt_pose, all_gt_trans = [], []
    all_gen_pose, all_gen_trans = [], []
    all_severities = []
    
    for i, (prefix, target, severity) in enumerate(tqdm(dataloader, desc=desc, leave=False)):
        if max_batches > 0 and i >= int(max_batches):
            break
            
        prefix = {k: v.to(device) for k, v in prefix.items()}
        target = {k: v.to(device) for k, v in target.items()}
        severity = severity.to(device)
        
        gt_pose = torch.cat([prefix['pose'], target['pose']], dim=1).cpu()
        gt_trans = torch.cat([prefix['trans'], target['trans']], dim=1).cpu()
        all_gt_pose.append(gt_pose)
        all_gt_trans.append(gt_trans)
        
        x_0 = generate_prior_from_prefix(prefix, target)
        generated_suffix = model.generate_suffix(prefix, x_0, severity, num_steps=num_steps)
        
        gen_pose = torch.cat([prefix['pose'], generated_suffix['pose']], dim=1).cpu()
        gen_trans = torch.cat([prefix['trans'], generated_suffix['trans']], dim=1).cpu()
        all_gen_pose.append(gen_pose)
        all_gen_trans.append(gen_trans)
        
        all_severities.extend(severity.cpu().tolist())

    return {
        "gt": {"pose": torch.cat(all_gt_pose, dim=0), "trans": torch.cat(all_gt_trans, dim=0)},
        "gen": {"pose": torch.cat(all_gen_pose, dim=0), "trans": torch.cat(all_gen_trans, dim=0)},
        "severities": all_severities
    }

import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

from thesis.src.utils.pipeline_utils import unpack_inference_outputs


def save_generated_to_npz(full_seq_pose, full_seq_trans, output_dir, filename="generated_PD_walk.npz"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / filename

    pose_np = full_seq_pose.squeeze(0).cpu().detach().numpy() # (Total_T, 24, D)
    trans_np = full_seq_trans.squeeze(0).cpu().detach().numpy() # (Total_T, 3)

    np.savez(full_path, pose=pose_np, trans=trans_np)
    print(f"\nSuccessfully saved generated sequence to: {full_path}")
    print(f"Saved Pose Shape:  {pose_np.shape}")
    print(f"Saved Trans Shape: {trans_np.shape}")

@torch.no_grad()
def generate_trajectories(model, dataloader, num_steps, device, max_batches=-1, desc="Generating", 
                          is_joint_model=False, force_joint_conditioning=False):
    """Generates synthetic dataset using model and dataloader."""
    model.eval()

    all_gt_pose, all_gt_trans = [], []
    all_gen_pose, all_gen_trans = [], []
    all_gt_severities, all_gen_severities, all_prior_severities = [], [], []
    
    for i, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
        if max_batches > 0 and i >= int(max_batches):
            break
            
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch_size = batch['severity'].shape[0]
        
        # Store unpadded ground truth sequences
        for b_idx in range(batch_size):
            l = batch['seq_len'][b_idx].item()
            all_gt_pose.append(batch['pose'][b_idx:b_idx+1, :l].cpu())
            all_gt_trans.append(batch['trans'][b_idx:b_idx+1, :l].cpu())
        all_gt_severities.extend(batch['severity'].cpu().tolist())

        if is_joint_model and model.gen_mode == 'one_shot':
            outputs = model._run_oneshot_inference(batch, num_steps, force_joint_conditioning=force_joint_conditioning)
        elif is_joint_model and model.gen_mode == 'ar_rollout':
            outputs = model._run_ar_inference(batch, num_steps, force_joint_conditioning=force_joint_conditioning)
        elif not is_joint_model and model.gen_mode == 'one_shot':
            outputs = model._run_oneshot_inference(batch, num_steps)
        elif not is_joint_model and model.gen_mode == 'ar_rollout':
            outputs = model._run_ar_inference(batch, num_steps)

        gen_pose, gen_trans, gen_labels, y_0_prior = unpack_inference_outputs(outputs)

        if is_joint_model and force_joint_conditioning:
            # MGM-Cond: Force the severity score to match the ground truth prefix
            all_prior_severities.extend(batch['severity'].cpu().tolist())
            all_gen_severities.extend(gen_labels.cpu().tolist())
            
        elif is_joint_model and not force_joint_conditioning:
            # MGM-Joint: Let the model predict its own severity score via jump process
            all_prior_severities.extend(y_0_prior.cpu().tolist())
            all_gen_severities.extend(gen_labels.cpu().tolist())
        else:
            # CFM-Cond: Standard conditional model
            all_prior_severities.extend(batch['severity'].cpu().tolist())
            all_gen_severities.extend(batch['severity'].cpu().tolist())

        # Store unpadded synthetic sequences
        for b_idx in range(batch_size):
            l = batch['seq_len'][b_idx].item()
            all_gen_pose.append(gen_pose[b_idx:b_idx+1, :l].cpu())
            all_gen_trans.append(gen_trans[b_idx:b_idx+1, :l].cpu())

    return {
        "gt": {"pose": all_gt_pose, "trans": all_gt_trans},
        "gen": {"pose": all_gen_pose, "trans": all_gen_trans},
        "severities": all_gt_severities,
        "gen_severities": all_gen_severities,
        "prior_severities": all_prior_severities
    }
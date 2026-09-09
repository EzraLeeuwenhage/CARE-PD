"""
Standalone inference script to generate datasets from trained checkpoints.
Example: python -m thesis.scripts.inference --ckpt path/to/best.ckpt --format npz
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from thesis.src.model import ConditionalBaselineModel, JointBaselineModel
from thesis.src.dataloader import get_dataloader
from thesis.src.sample import generate_trajectories
from thesis.src.utils.pipeline_utils import load_config
from thesis.src.utils.smpl_io import save_smpl_pkl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint .ckpt")
    parser.add_argument("--config", type=str, default="thesis/configs/baseline_3d.yaml", help="Path to yaml config")
    parser.add_argument("--format", type=str, choices=["npz", "pkl"], default="npz", help="Output dataset format")
    parser.add_argument("--max_batches", type=int, default=-1, help="Max batches to generate (-1 for all test data)")
    parser.add_argument("--out_dir", type=str, default="thesis/data/synthetic_datasets", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    is_joint = cfg['model'].get('is_joint_model', False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint: {args.ckpt}")
    model_cls = JointBaselineModel if is_joint else ConditionalBaselineModel
    model = model_cls.load_from_checkpoint(args.ckpt, cfg=cfg).to(device)
    model.eval()

    test_loader = get_dataloader(cfg, mode='test', is_joint_model_train=False)

    print(f"Generating trajectories (Joint={is_joint})...")
    data_dict = generate_trajectories(
        model=model, dataloader=test_loader, num_steps=cfg['sampling']['num_steps'], 
        device=device, max_batches=args.max_batches, desc="Generating Synthetic Data", 
        is_joint_model=is_joint
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_{cfg['data'].get('representation', '3D').lower()}.{args.format}"

    if args.format == "npz":
        gen_pose = data_dict["gen"]["pose"].cpu().numpy()
        gen_trans = data_dict["gen"]["trans"].cpu().numpy()
        
        save_dict = {}
        for i in range(gen_pose.shape[0]):
            key = f"seq_{i:04d}"
            save_dict[key] = gen_pose[i]
            save_dict[f"{key}_trans"] = gen_trans[i]
            
        np.savez(out_path, **save_dict)
        print(f"Saved synthetic dataset to: {out_path}")
        
    elif args.format == "pkl":
        save_smpl_pkl(
            generated_pose=data_dict["gen"]["pose"], 
            generated_trans=data_dict["gen"]["trans"], 
            output_filepath=str(out_path), 
            subject_id="SYNTHETIC", 
            walk_prefix="walk"
        )

if __name__ == "__main__":
    main()
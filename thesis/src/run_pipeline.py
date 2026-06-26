import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from thesis.src.model import FlowMatchingMLP
from thesis.src.dataloader import SMPL6DDataset
from thesis.src.generate_prior import generate_prior_from_prefix
from thesis.src.sample import euler_ode_solver
from thesis.data.raw.data_conversion_utils import build_smpl_pkl_from_6d_smpl
from thesis.care_pd.smpl2h36m import convert_smpl_to_h36m
from thesis.src.evaluate import H36MEvaluator
from thesis.src.train import train

CONFIG_PATH = "thesis/configs/baseline.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def generate_full_dataset(model, dataset, cfg, device):
    print("\n--- PHASE 2: DATASET GENERATION ---")
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    model.eval()
    
    all_gt_pose, all_gt_trans = [], []
    all_gen_pose, all_gen_trans = [], []
    all_severities = []
    
    num_steps = cfg['sampling']['num_steps']
    
    for prefix, target, severity in tqdm(loader, desc="Generating Suffixes"):
        prefix = {k: v.to(device) for k, v in prefix.items()}
        target = {k: v.to(device) for k, v in target.items()}
        severity = severity.to(device)
        
        gt_pose = torch.cat([prefix['pose'], target['pose']], dim=1).cpu()
        gt_trans = torch.cat([prefix['trans'], target['trans']], dim=1).cpu()
        all_gt_pose.append(gt_pose)
        all_gt_trans.append(gt_trans)
        
        x_0 = generate_prior_from_prefix(prefix, target)
        generated_suffix = euler_ode_solver(model, prefix, x_0, severity, num_steps=num_steps)
        
        gen_pose = torch.cat([prefix['pose'], generated_suffix['pose']], dim=1).cpu()
        gen_trans = torch.cat([prefix['trans'], generated_suffix['trans']], dim=1).cpu()
        all_gen_pose.append(gen_pose)
        all_gen_trans.append(gen_trans)
        
        all_severities.extend(severity.cpu().tolist())

    return {
        "gt": {
            "pose": torch.cat(all_gt_pose, dim=0),
            "trans": torch.cat(all_gt_trans, dim=0)
        },
        "gen": {
            "pose": torch.cat(all_gen_pose, dim=0),
            "trans": torch.cat(all_gen_trans, dim=0)
        },
        "severities": all_severities
    }

def format_and_convert(data_dict, cfg):
    print("\n--- PHASE 3: FORMAT CONVERSION ---")
    out_dir = Path(cfg['paths']['output_dir'])
    
    # Pre-define paths
    smpl_dir = out_dir / "SMPL"
    h36m_dir = out_dir / "h36m"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    h36m_dir.mkdir(parents=True, exist_ok=True)
    
    gt_pkl = smpl_dir / "ground_truth.pkl"
    gen_pkl = smpl_dir / "generated.pkl"
    
    gt_h36m = h36m_dir / "ground_truth_3d_world.npz"
    gen_h36m = h36m_dir / "generated_3d_world.npz"
    
    # ground truth data, skip if exists (assuming using full dataset)
    if gt_h36m.exists() and gt_pkl.exists():
        print("Ground Truth H36M data already exists.")
    else:
        print("Formatting Ground Truth to SMPL...")
        build_smpl_pkl_from_6d_smpl(data_dict["gt"]["pose"], data_dict["gt"]["trans"], str(gt_pkl), "GT", "gt")
        print("Converting Ground Truth SMPL -> H36M (This takes a moment)...")
        convert_smpl_to_h36m(str(gt_pkl))
    
    # generated data
    print("Formatting Generated data to SMPL...")
    build_smpl_pkl_from_6d_smpl(data_dict["gen"]["pose"], data_dict["gen"]["trans"], str(gen_pkl), "GEN", "gen")
    print("Converting Generated SMPL -> H36M...")
    convert_smpl_to_h36m(str(gen_pkl))
    
    # label registry for evaluation
    gt_labels = {"key_to_severity": {}}
    gen_labels = {"key_to_severity": {}}
    
    for i, sev in enumerate(data_dict["severities"]):
        gt_labels["key_to_severity"][f"GT__gt_{i:03d}"] = sev
        gen_labels["key_to_severity"][f"GEN__gen_{i:03d}"] = sev
        
    gt_labels_path = h36m_dir / "gt_labels.json"
    gen_labels_path = h36m_dir / "gen_labels.json"
    
    with open(gt_labels_path, 'w') as f: json.dump(gt_labels, f)
    with open(gen_labels_path, 'w') as f: json.dump(gen_labels, f)
        
    return {
        "gt_h36m": gt_h36m,
        "gen_h36m": gen_h36m,
        "gt_labels": gt_labels_path,
        "gen_labels": gen_labels_path,
        "out_dir": out_dir
    }

def evaluate_pipeline(paths):
    print("\n--- PHASE 4: EVALUATION ---")
    evaluator = H36MEvaluator(fps=30)
    
    evaluator.evaluate_and_cache(
        npz_path=str(paths["gt_h36m"]),
        labels_path=str(paths["gt_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gt_h36m_distributions.pkl")
    )
    
    evaluator.evaluate_and_cache(
        npz_path=str(paths["gen_h36m"]),
        labels_path=str(paths["gen_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gen_h36m_distributions.pkl")
    )

if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Unified Pipeline on: {device.type.upper()}")
    
    dataset = SMPL6DDataset(config_path=CONFIG_PATH)
    model = FlowMatchingMLP(config_path=CONFIG_PATH).to(device)

    # optional train time optimization
    model = torch.compile(model)
    
    print("\n--- PHASE 1: TRAINING ---")
    loader = DataLoader(
        dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=cfg['training']['shuffle'],
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    train(model, loader, cfg, device=device)
    
    model.load_state_dict(torch.load(cfg['paths']['weights'], map_location=device))
    
    data_dict = generate_full_dataset(model, dataset, cfg, device)
    paths = format_and_convert(data_dict, cfg)
    evaluate_pipeline(paths)
    
    print("\nPipeline Finished Successfully!")
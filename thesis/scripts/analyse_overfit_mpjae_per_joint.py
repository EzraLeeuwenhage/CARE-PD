import math
from pathlib import Path
import numpy as np
import torch

from thesis.src.dataloader import get_dataloader
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.model_conditional import ConditionalBaselineModel
from thesis.src.utils.pipeline_utils import load_config

def dissect():
    # 1. Device detection (works on CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Running on device: {device}")

    # 2. Config & Dataloader setup
    config_path = Path("thesis/configs/overfit_baseline_3d.yaml")
    if not config_path.exists():
        config_path = Path("thesis/configs/baseline_3d.yaml")
    cfg = load_config(str(config_path))
    
    eval_loader = get_dataloader(cfg, mode='eval')
    dataset = eval_loader.dataset

    print("\n" + "="*70)
    print("      DISSECTING OVERFIT VALIDATION DATASET")
    print("="*70)
    print(f"Dataset type wrapped:         {type(dataset.dataset).__name__}")
    print(f"Indices in valid_indices:     {dataset.valid_indices}")
    
    for i, v_idx in enumerate(dataset.valid_indices):
        item = dataset.dataset.window_indices[v_idx]
        print(f"  Slot {i} (v_idx={v_idx}): Sequence '{item[0]}', start_idx = {item[1]} (frames [{item[1]:03d} -> {item[1]+60:03d}])")

    # 3. Checkpoint resolution
    model_name = cfg['model'].get('name', 'GenerativeModel')
    ckpt_dir = Path("thesis/data/processed") / model_name / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    
    if not ckpts:
        fallback = ckpt_dir / "best-epoch=99-val" / "mpjae_deg=7.68.ckpt"
        if fallback.exists():
            ckpts = [fallback]
            latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
            model = ConditionalBaselineModel.load_from_checkpoint(str(latest_ckpt), cfg=cfg).to(device)
        else: 
            print(f"\n[Warning] No checkpoint found in: {ckpt_dir.resolve()}")
            print("Note: If you trained on Google Colab, download the .ckpt file from your Google Drive into this folder, or run this script directly in Colab.")
            model = ConditionalBaselineModel(cfg).to(device)
    else:
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading model weights from: {latest_ckpt}")
        model = ConditionalBaselineModel.load_from_checkpoint(str(latest_ckpt), cfg=cfg).to(device)

    model.eval()
    evaluator = SMPLEvaluator()
    prefix_len = cfg['windowing']['prefix_length']

    print("\n" + "="*70)
    print("   WINDOW-BY-WINDOW MPJAE EVALUATION (Target Frames [15:60])")
    print("="*70)

    for slot_idx in range(len(dataset.valid_indices)):
        actual_v_idx = dataset.valid_indices[slot_idx]
        win_info = dataset.dataset.window_indices[actual_v_idx]
        start_frame = win_info[1]

        sample = dataset.dataset[actual_v_idx]
        batch = {
            'pose': sample['pose'].unsqueeze(0).to(device),
            'trans': sample['trans'].unsqueeze(0).to(device),
            'seq_len': sample['seq_len'].unsqueeze(0).to(device),
            'severity': sample['severity'].unsqueeze(0).to(device),
            'key': [sample['key']]
        }

        with torch.no_grad():
            gen_pose, _ = model._run_ar_inference(batch, num_steps=100)

            gt_target = batch['pose'][:, prefix_len:60]
            gen_target = gen_pose[:, prefix_len:60]

            # 12 hardest joints
            hard_mpjae_rad = evaluator.compute_mpjae(gt_target.cpu(), gen_target.cpu(), return_per_joint=False)
            hard_mpjae_deg = hard_mpjae_rad * (180.0 / math.pi)

            # All 24 individual joints
            per_joint_rad = evaluator.compute_mpjae(gt_target.cpu(), gen_target.cpu(), return_per_joint=True)
            per_joint_deg = per_joint_rad * (180.0 / math.pi)
            overall_24_deg = float(np.mean(per_joint_deg))

        trained_status = "TRAINED (Window 0)" if start_frame == 0 else "UNSEEN / UNTRAINED (Window 1)"
        print(f"\n--- WINDOW {slot_idx} (frames [{start_frame:03d} -> {start_frame+60:03d}]) [{trained_status}] ---")
        print(f"  * Mean MPJAE (12 Hardest Joints): {hard_mpjae_deg:.4f}°")
        print(f"  * Mean MPJAE (All 24 Joints):     {overall_24_deg:.4f}°")
        print(f"  * Knee Error:  L_Knee = {per_joint_deg[4]:.2f}°, R_Knee = {per_joint_deg[5]:.2f}°")
        print(f"  * Ankle Error: L_Ankle = {per_joint_deg[7]:.2f}°, R_Ankle = {per_joint_deg[8]:.2f}°")

if __name__ == "__main__":
    dissect()
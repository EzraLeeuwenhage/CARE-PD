import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import yaml
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from thesis.src.model import ConditionalBaselineModel, JointBaselineModel
from thesis.src.dataloader import get_dataloader
from thesis.src.sample import generate_trajectories
from thesis.utils.sixD2smpl import build_smpl_pkl_from_6d_smpl
from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator


CONFIG_PATH = "thesis/configs/baseline.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def format_and_convert(data_dict, cfg, is_joint_model=False):
    out_dir = Path(cfg['paths']['output_dir'])

    smpl_dir = out_dir / "SMPL"
    h36m_dir = out_dir / "h36m"
    sixd_dir = out_dir / "6D_SMPL"
    
    smpl_dir.mkdir(parents=True, exist_ok=True)
    h36m_dir.mkdir(parents=True, exist_ok=True)
    sixd_dir.mkdir(parents=True, exist_ok=True)
    
    gt_pkl = smpl_dir / "ground_truth.pkl"
    gen_pkl = smpl_dir / "generated.pkl"
    
    gt_h36m = h36m_dir / "ground_truth_3d_world.npz"
    gen_h36m = h36m_dir / "generated_3d_world.npz"
    
    gt_6d_npz = sixd_dir / "ground_truth_6d.npz"
    gen_6d_npz = sixd_dir / "generated_6d.npz"

    gt_dict, gen_dict = {}, {}
    gt_labels, gen_labels = {"key_to_severity": {}}, {"key_to_severity": {}}
    
    print("Formatting and caching raw 6D sequences...")
    for i, sev in enumerate(data_dict["severities"]):
        seq_key = f"seq_{i:03d}"

        # Save 6D SMPL sequences to NPZ files for both ground truth and generated data
        gt_dict[seq_key] = data_dict["gt"]["pose"][i].numpy()
        gt_dict[f"{seq_key}_trans"] = data_dict["gt"]["trans"][i].numpy()
        gen_dict[seq_key] = data_dict["gen"]["pose"][i].numpy()
        gen_dict[f"{seq_key}_trans"] = data_dict["gen"]["trans"][i].numpy()
        
        # Use generated severities if joint model, otherwise fallback to conditional/gt
        gen_sev = data_dict["gen_severities"][i] if is_joint_model else sev
        
        # Registry mapping for SMPLEvaluator (Matches 6D Seq keys)
        gt_labels["key_to_severity"][seq_key] = sev
        gen_labels["key_to_severity"][seq_key] = gen_sev
        
        # Registry mapping for H36MEvaluator (Matches converted .pkl keys)
        gt_labels["key_to_severity"][f"GT__gt_{i:03d}"] = sev
        gen_labels["key_to_severity"][f"GEN__gen_{i:03d}"] = gen_sev

    np.savez(gt_6d_npz, **gt_dict)
    np.savez(gen_6d_npz, **gen_dict)
    
    if gt_h36m.exists() and gt_pkl.exists():
        print("Ground Truth H36M data already exists.")
    else:
        print("Formatting Ground Truth to SMPL...")
        build_smpl_pkl_from_6d_smpl(data_dict["gt"]["pose"], data_dict["gt"]["trans"], str(gt_pkl), "GT", "gt")
        print("Converting Ground Truth SMPL -> H36M (This takes a moment)...")
        convert_smpl_to_h36m(str(gt_pkl), str(gt_h36m.parent), gt_h36m.name)
    
    print("Formatting Generated data to SMPL...")
    build_smpl_pkl_from_6d_smpl(data_dict["gen"]["pose"], data_dict["gen"]["trans"], str(gen_pkl), "GEN", "gen")
    print("Converting Generated SMPL -> H36M...")
    convert_smpl_to_h36m(str(gen_pkl), str(gen_h36m.parent), gen_h36m.name)
        
    gt_labels_path = h36m_dir / "gt_labels.json"
    gen_labels_path = h36m_dir / "gen_labels.json"
    
    with open(gt_labels_path, 'w') as f: json.dump(gt_labels, f)
    with open(gen_labels_path, 'w') as f: json.dump(gen_labels, f)
        
    return {
        "gt_6d": gt_6d_npz, "gen_6d": gen_6d_npz, "gt_h36m": gt_h36m,
        "gen_h36m": gen_h36m, "gt_labels": gt_labels_path,
        "gen_labels": gen_labels_path, "out_dir": out_dir
    }


def evaluate_pipeline(paths):
    evaluator = H36MEvaluator(fps=30)
    evaluator.evaluate_and_cache(
        npz_path=str(paths["gt_h36m"]),
        labels_path=str(paths["gt_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gt_h36m_distributions.pkl")
    )
    evaluator.evaluate_and_cache(
        npz_path=str(paths["gen_h36m"]),
        labels_path=str(paths["gen_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gen_h36m_distributions.pkl"),
        synthetic=True
    )

    smpl_evaluator = SMPLEvaluator()
    smpl_evaluator.evaluate_and_cache(
        gt_npz_path=paths["gt_6d"],
        gen_npz_path=paths["gen_6d"],
        labels_path=paths["gen_labels"],
        cache_output_path=str(paths["out_dir"] / "evaluation" / "smpl_mpjae_evaluation.json"),
        verbose=True
    )


if __name__ == "__main__":
    cfg = load_config()
    is_joint_model = cfg['model'].get('is_joint_model', False)
    model_name = cfg['model'].get('name', 'GenerativeModel')
    print(f"\nStarting model train-test pipeline for '{model_name}' (Joint Model: {is_joint_model})...")

    if not is_joint_model:
        model_class = ConditionalBaselineModel
        train_loader = get_dataloader(cfg, mode='train', is_joint_model_train=False)
        eval_loader = get_dataloader(cfg, mode='eval', is_joint_model_train=False)
        test_loader = get_dataloader(cfg, mode='test', is_joint_model_train=False)
    else:
        model_class = JointBaselineModel
        train_loader = get_dataloader(cfg, mode='train', is_joint_model_train=True)
        eval_loader = get_dataloader(cfg, mode='eval', is_joint_model_train=False)
        test_loader = get_dataloader(cfg, mode='test', is_joint_model_train=False)

    model = model_class(cfg)

    wandb_logger = WandbLogger(
        project="thesis",
        name=model_name,
        config=cfg
    )
    
    ckpt_dir = Path(cfg['paths']['output_dir']) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mpjae_rad", 
        mode="min", 
        save_top_k=1,
        dirpath=str(ckpt_dir),
        filename=f"{model_name}-best-{{epoch:02d}}-{{val/mpjae_rad:.4f}}"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg['training']['epochs'],
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=cfg['training'].get('eval_interval', 10),
        log_every_n_steps=cfg['training'].get('log_interval', 5)
    )

    print("\n--- PHASE 1: TRAINING ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    print("\n--- PHASE 2: DATASET GENERATION ---")
    best_model_path = checkpoint_callback.best_model_path

    print(f"Loading best checkpoint from: {best_model_path}")
    best_model = model_class.load_from_checkpoint(best_model_path, cfg=cfg)
    
    data_dict = generate_trajectories(
        model=best_model, 
        dataloader=test_loader, 
        num_steps=cfg['sampling']['num_steps'], 
        device=trainer.device,
        max_batches=-1,
        desc="Generating Final Test Set", 
        is_joint_model=is_joint_model
    )
    
    print("\n--- PHASE 3: FORMAT CONVERSION ---")
    paths = format_and_convert(data_dict, cfg, is_joint_model=is_joint_model)

    print("\n--- PHASE 4: EVALUATION ---")
    evaluate_pipeline(paths)
    
    print("\nPipeline Finished Successfully!")

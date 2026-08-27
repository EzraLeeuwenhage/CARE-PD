import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import yaml
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from thesis.src.callbacks import EpochAndValPrintCallback, WandBEvaluationCallback
from thesis.src.model import ConditionalBaselineModel, JointBaselineModel
from thesis.src.dataloader import get_dataloader
from thesis.src.sample import generate_trajectories
from thesis.utils.pipeline_utils import load_config, format_and_convert, evaluate_pipeline


CONFIG_PATH = "thesis/configs/baseline.yaml"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
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

    out_dir_path = Path(cfg['paths']['output_dir'])
    out_dir_path.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project="thesis",
        name=model_name,
        save_dir=str(out_dir_path),
        config=cfg
    )
    
    log_interval = cfg['training'].get('log_interval', 5)
    eval_interval = cfg['training'].get('eval_interval', 10)
    wandb_eval_interval = cfg['training'].get('wandb_eval_interval', 50)

    print_callback = EpochAndValPrintCallback(
        train_interval=log_interval, 
        val_interval=eval_interval
    )
    
    wandb_eval_callback = WandBEvaluationCallback(
        cfg=cfg, 
        eval_interval=wandb_eval_interval
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mpjae_rad", 
        mode="min", 
        save_top_k=1,
        dirpath=str(Path(cfg['paths']['output_dir']) / "checkpoints"),
        filename=f"best-{{epoch:02d}}-{{val/mpjae_rad:.4f}}"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[print_callback, checkpoint_callback, wandb_eval_callback],
        enable_progress_bar=False,
        max_epochs=cfg['training']['epochs'],
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=eval_interval,
    )

    print("\n--- PHASE 1: TRAINING ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    print("\n--- PHASE 2: DATASET GENERATION ---")
    best_model_path = checkpoint_callback.best_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading best checkpoint from: {best_model_path}")
    best_model = model_class.load_from_checkpoint(best_model_path, cfg=cfg).to(device)
    
    data_dict = generate_trajectories(
        model=best_model, 
        dataloader=test_loader, 
        num_steps=cfg['sampling']['num_steps'], 
        device=device,
        max_batches=-1,
        desc="Generating Final Test Set", 
        is_joint_model=is_joint_model,
    )

    if is_joint_model:
        print("\n--- PHASE 2.5: CONDITIONAL ADHERENCE (LABEL ACCURACY) ---")
        gt_sevs = np.array(data_dict["severities"])
        gen_sevs = np.array(data_dict["gen_severities"])
        
        test_label_acc = np.mean(gt_sevs == gen_sevs)
        correct_matches = np.sum(gt_sevs == gen_sevs)
        
        print(f"Final Test Label Accuracy: {test_label_acc:.4f} ({correct_matches}/{len(gt_sevs)} matches)")
        
        eval_dir = Path(cfg['paths']['output_dir']) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "test_label_accuracy.json", "w") as f:
            json.dump({"test_label_accuracy": float(test_label_acc)}, f, indent=4)
    
    print("\n--- PHASE 3: FORMAT CONVERSION ---")
    paths = format_and_convert(data_dict, cfg, is_joint_model=is_joint_model)

    print("\n--- PHASE 4: EVALUATION ---")
    evaluate_pipeline(paths)

    wandb_logger.experiment.finish()
    print("\nPipeline Finished Successfully!")

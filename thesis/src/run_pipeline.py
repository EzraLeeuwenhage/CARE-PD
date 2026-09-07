import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from pathlib import Path

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from thesis.src.callbacks import EpochAndValPrintCallback, WandBEvaluationCallback
from thesis.src.model import ConditionalBaselineModel, JointBaselineModel
from thesis.src.dataloader import get_dataloader
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.sample import generate_trajectories
from thesis.src.utils.pipeline_utils import load_config, format_and_convert, evaluate_and_plot_distributions

CONFIG_PATH = "thesis/configs/baseline_3d.yaml"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    model_name = cfg['model'].get('name', 'GenerativeModel')
    is_joint_model = cfg['model'].get('is_joint_model', False)
    min_z_travel = cfg['windowing'].get('min_z_travel', 0.5)
    overfit_severity_class = cfg['training'].get('overfit_severity_class', -1)

    out_dir_path = Path(cfg['paths']['output_dir'])
    out_dir_path.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project="thesis",
        name=model_name,
        save_dir=str(out_dir_path),
        config=cfg
    )
    
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
    
    log_interval = cfg['training'].get('log_interval', 5)
    val_interval = cfg['training'].get('val_interval', 10)
    wandb_val_interval = cfg['training'].get('wandb_val_interval', 50)

    print_callback = EpochAndValPrintCallback(train_interval=log_interval, val_interval=val_interval)
    wandb_eval_callback = WandBEvaluationCallback(cfg=cfg, eval_interval=wandb_val_interval)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mpjae_deg", 
        mode="min", 
        save_top_k=1,
        dirpath=str(out_dir_path / "checkpoints"),
        filename=f"best-{{epoch:02d}}-{{val/mpjae_deg:.2f}}"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[print_callback, checkpoint_callback, wandb_eval_callback],
        enable_progress_bar=False,
        max_epochs=cfg['training']['epochs'],
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=val_interval,
    )

    print("\n--- PHASE 0: BASELINE EVALUATION ---")
    trainer.validate(model, dataloaders=eval_loader)

    print("\n--- PHASE 1: TRAINING ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    print("\n--- PHASE 2: DATASET GENERATION ---")
    best_model_path = checkpoint_callback.best_model_path

    if overfit_severity_class == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading best checkpoint from: {best_model_path}")
        best_model = model_class.load_from_checkpoint(best_model_path, cfg=cfg).to(device)
        
        data_dict = generate_trajectories(
            model=best_model, dataloader=test_loader, num_steps=cfg['sampling']['num_steps'], 
            device=device, max_batches=-1, desc="Generating Final Test Set", is_joint_model=is_joint_model,
        )

        if is_joint_model:
            gt_sevs = np.array(data_dict["severities"])
            gen_sevs = np.array(data_dict["gen_severities"])
            test_label_acc = np.mean(gt_sevs == gen_sevs)
            print(f"Final Test Label Accuracy: {test_label_acc:.4f} ({np.sum(gt_sevs == gen_sevs)}/{len(gt_sevs)} matches)")
        
        print("\n--- PHASE 3: FINAL TEST EVALUATION & DISK STORAGE ---")
        memory_data = format_and_convert(data_dict, cfg, is_joint_model=is_joint_model, save_to_disk=True)
        dist_metrics, vis_dir = evaluate_and_plot_distributions(
            memory_data, 
            min_z_travel=min_z_travel, 
            is_joint_model=is_joint_model, 
            step_name="Final Test"
        )

        smpl_evaluator = SMPLEvaluator()
        mpjae_rad = smpl_evaluator.compute_mpjae(data_dict["gt"]["pose"], data_dict["gen"]["pose"])
        dist_metrics["eval_metrics/Overall_MPJAE_deg"] = mpjae_rad * (180.0 / np.pi)

        # Log to W&B
        if wandb_logger.experiment is not None:
            for img_path in vis_dir.glob("*.png"):
                dist_metrics[f"eval_visuals/{img_path.stem}"] = wandb.Image(str(img_path))
            wandb_logger.experiment.log(dist_metrics)
    else: 
        print("[OVERFIT MODE] Skipping Test Generation and Evaluation.")

    wandb_logger.experiment.finish()
    print("\nPipeline Finished Successfully!")
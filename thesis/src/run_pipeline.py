import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from pathlib import Path

import torch
torch.set_float32_matmul_precision('high')

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from thesis.src.callbacks import EpochAndValPrintCallback, WandBEvaluationCallback
from thesis.src.model_conditional import ConditionalBaselineModel
from thesis.src.model_joint import JointBaselineModel
from thesis.src.dataloader import get_dataloader
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.sample import generate_trajectories
from thesis.src.utils.pipeline_utils import (
    load_config, format_and_convert, evaluate_and_plot_distributions
)

CONFIG_PATH = "thesis/configs/overfit_baseline_3d.yaml"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    model_name = cfg['model'].get('name', 'GenerativeModel')
    is_joint_model = cfg['model'].get('is_joint_model', False)
    min_z_travel = cfg['windowing'].get('min_z_travel', 0.5)
    overfit_severity_class = cfg['training'].get('overfit_severity_class', -1)

    out_dir_path = Path(cfg['paths']['output_dir'])
    out_dir_path.mkdir(parents=True, exist_ok=True)

    local_wandb_dir = Path("/content/wandb_runtime")
    local_wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project="thesis",
        name=model_name,
        dir=str(local_wandb_dir),
        config=cfg,
        # reinit=True
    )
    wandb_logger = WandbLogger(experiment=run)

    print(f"\nStarting model train-test pipeline for '{model_name}' (Joint Model: {is_joint_model})...")

    if not is_joint_model:
        model_class = ConditionalBaselineModel
        train_loader = get_dataloader(cfg, mode='train')
        eval_loader = get_dataloader(cfg, mode='eval')
        test_loader = get_dataloader(cfg, mode='test')
    else:
        model_class = JointBaselineModel
        train_loader = get_dataloader(cfg, mode='train')
        eval_loader = get_dataloader(cfg, mode='eval')
        test_loader = get_dataloader(cfg, mode='test')

    # optionally load model and trainer state from checkpoint
    resume_ckpt = cfg['training'].get('resume_checkpoint', None)
    resume_trainer_state = cfg['training'].get('resume_trainer_state', False)

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[Resume Error] Checkpoint file not found at: {ckpt_path.resolve()}")
        print(f"\n[RESUME] Loading model weights from: {ckpt_path}\n")
        model = model_class.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    else:
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
        filename="best-{epoch:02d}-{val_mpjae_deg:.2f}",
        auto_insert_metric_name=False
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[print_callback, checkpoint_callback, wandb_eval_callback],
        enable_progress_bar=False,
        max_epochs=cfg['training']['epochs'],
        precision="32",
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=val_interval,
    )

    print("\n--- PHASE 0: BASELINE EVALUATION ---")
    trainer.validate(model, dataloaders=eval_loader, verbose=False)

    print("\n--- PHASE 1: TRAINING ---")
    fit_ckpt_path = str(resume_ckpt) if (resume_ckpt and resume_trainer_state) else None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader, ckpt_path=fit_ckpt_path)

    print("\n--- PHASE 2: DATASET GENERATION ---")
    best_model_path = checkpoint_callback.best_model_path
    target_ckpt = best_model_path if (best_model_path and Path(best_model_path).exists()) else resume_ckpt

    if overfit_severity_class == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if target_ckpt and Path(target_ckpt).exists():
            print(f"Loading checkpoint from: {target_ckpt}")
            best_model = model_class.load_from_checkpoint(str(target_ckpt), cfg=cfg).to(device)
        else:
            best_model = model.to(device)
        
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
        dist_metrics["test_metrics/Overall_MPJAE_deg"] = mpjae_rad * (180.0 / np.pi)
        dist_metrics["test_metrics/label_accuracy"] = test_label_acc

        for img_path in vis_dir.glob("*.png"):
            dist_metrics[f"test_visuals/{img_path.stem}"] = wandb.Image(str(img_path))
        
        run.log(dist_metrics, step=trainer.global_step)
    else: 
        print("[OVERFIT MODE] Skipping Test Generation and Evaluation.")

    run.finish()
    print("\nPipeline Finished Successfully!")
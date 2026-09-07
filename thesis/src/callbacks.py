import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import time

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from smplx.body_models import SMPL

from thesis.src.sample import generate_trajectories
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.generate_prior import generate_prior_from_prefix
from thesis.src.utils.geometry_utils import forward_to_h36m
from thesis.src.utils.pipeline_utils import format_and_convert, evaluate_and_plot_distributions
from thesis.src.utils.rendering.render_h36m_gif import render_three_way_gif


class EpochAndValPrintCallback(Callback):
    """Custom callback to print train and val metrics at specified epoch intervals."""
    def __init__(self, train_interval, val_interval):
        super().__init__()
        self.train_interval = train_interval
        self.val_interval = val_interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.train_interval == 0:
            loss = trainer.callback_metrics.get("train/loss_total")
            loss_val = f"{loss.item():.4f}" if loss is not None else "N/A"
            print(f"Epoch {epoch:04d}/{trainer.max_epochs} | Train Loss: {loss_val}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        epoch = trainer.current_epoch + 1
        if epoch % self.val_interval == 0:
            mpjae = trainer.callback_metrics.get("val/mpjae_deg")
            acc = trainer.callback_metrics.get("val/label_accuracy")
            
            mpjae_str = f"{mpjae.item():.2f} deg" if mpjae is not None else "N/A"
            
            if acc is not None:
                print(f" >>> VALIDATION Epoch {epoch:04d} | MPJAE: {mpjae_str} | Label Acc: {acc.item():.4f}")
            else:
                print(f" >>> VALIDATION Epoch {epoch:04d} | MPJAE: {mpjae_str}")


class WandBEvaluationCallback(Callback):
    """Evaluates generated distributions and Anchor GIFs entirely in RAM."""
    def __init__(self, cfg, eval_interval=50):
        super().__init__()
        self.cfg = cfg
        self.eval_interval = eval_interval
        self.is_joint_model = cfg['model'].get('is_joint_model', False)
        self.anchors = {}
        
        self.cache_dir = Path(cfg['paths']['output_dir']) / "wandb_eval_cache"
        self.vis_dir = self.cache_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.smpl_model = SMPL(model_path='thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl', num_betas=10).eval()
        self.h36m_regressor = torch.tensor(np.load('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy'), dtype=torch.float32)
        self.smpl_evaluator = SMPLEvaluator(fps=30)

    def _sample_anchors(self, trainer, pl_module):
        """Samples and caches 1 anchor sequence per class."""
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list): val_loader = val_loader[0]
        
        print("\n[W&B Callback] Sampling Anchor Sequences across Severity Classes...")
        candidates = defaultdict(list)
        for prefix, target, severity in val_loader:
            for b_idx in range(severity.shape[0]):
                sev_val = severity[b_idx].item()
                pref_single = {k: v[b_idx:b_idx+1].cpu() for k, v in prefix.items()}
                targ_single = {k: v[b_idx:b_idx+1].cpu() for k, v in target.items()}
                candidates[sev_val].append((pref_single, targ_single))
                
        for sev_val in sorted(candidates.keys()):
            pref_single, targ_single = random.choice(candidates[sev_val])
            pref_single = {k: v.to(pl_module.device) for k, v in pref_single.items()}
            targ_single = {k: v.to(pl_module.device) for k, v in targ_single.items()}
            x_0 = generate_prior_from_prefix(pref_single, targ_single)
            self.anchors[sev_val] = {"prefix": pref_single, "x_0": x_0, "target": targ_single, "severity": sev_val}
            print(f"  -> Locked Random Anchor for Severity Class {sev_val}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        
        epoch = trainer.current_epoch + 1
        is_baseline = (trainer.global_step == 0)
        
        if not is_baseline and epoch % self.eval_interval != 0:
            return
        
        val_start_time = time.time()
        
        if not self.anchors:
            self._sample_anchors(trainer, pl_module)

        for old_file in self.vis_dir.glob("*"):
            old_file.unlink()
        
        display_epoch = 0 if is_baseline else epoch
        print(f"\n--- [W&B Callback] Running Validation (Epoch {display_epoch}) ---")

        val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        
        gen_start = time.time()
        data_dict = generate_trajectories(
            model=pl_module, dataloader=val_loader, 
            num_steps=self.cfg['sampling']['num_steps'], device=pl_module.device, 
            max_batches=self.cfg['training'].get('eval_batches', -1),
            desc=f"W&B Eval Ep {display_epoch}", is_joint_model=self.is_joint_model,
        )
        print(f"  [Time] Trajectory Generation: {time.time() - gen_start:.2f}s")

        is_overfit = self.cfg['training'].get('overfit_severity_class', -1) >= 0

        # Compute MPJAE
        mpjae_rad = self.smpl_evaluator.compute_mpjae(data_dict["gt"]["pose"], data_dict["gen"]["pose"])
        mpjae_deg = mpjae_rad * (180.0 / np.pi)
    
        wandb_logs = {
            "epoch": display_epoch,
            "eval_metrics/Overall_MPJAE_deg": mpjae_deg
        }

        # Evaluate distributions
        if not is_overfit:
            conv_start = time.time()
            memory_data = format_and_convert(data_dict, self.cfg, self.is_joint_model, save_to_disk=False)
            print(f"  [Time] SMPL to H36M Batch Conversion: {time.time() - conv_start:.2f}s")

            metric_start = time.time()
            min_z = self.cfg['windowing'].get('min_z_travel', 0.5)
            memory_data["out_dir"] = self.cache_dir
            
            dist_metrics, vis_dir = evaluate_and_plot_distributions(
                memory_data, min_z, self.is_joint_model, step_name=f"Epoch {display_epoch}"
            )
            print(f"  [Time] Metric Extraction & Plots: {time.time() - metric_start:.2f}s")

            wandb_logs.update(dist_metrics)
            if wandb.run is not None:
                for img_path in vis_dir.glob("*.png"):
                    wandb_logs[f"eval_visuals/{img_path.stem}"] = wandb.Image(str(img_path))

        # Render Anchor GIFs
        gif_start = time.time()
        self.smpl_model = self.smpl_model.to(pl_module.device)
        self.h36m_regressor = self.h36m_regressor.to(pl_module.device)
        
        gif_paths = []
        for sev_val, anchor_data in self.anchors.items():
            if self.is_joint_model:
                gen_suffix, gen_severity = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], severity_score=None, num_steps=self.cfg['sampling']['num_steps']
                )
                gen_sev_val = gen_severity[0].item()
            else:
                gen_suffix = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], severity_score=torch.tensor([sev_val]).to(pl_module.device), num_steps=self.cfg['sampling']['num_steps']
                )
                gen_sev_val = None
                
            gt_full_pose = torch.cat([anchor_data["prefix"]['pose'], anchor_data["target"]['pose']], dim=1)[0]
            gt_full_trans = torch.cat([anchor_data["prefix"]['trans'], anchor_data["target"]['trans']], dim=1)[0]
            prior_full_pose = torch.cat([anchor_data["prefix"]['pose'], anchor_data["x_0"]['pose']], dim=1)[0]
            prior_full_trans = torch.cat([anchor_data["prefix"]['trans'], anchor_data["x_0"]['trans']], dim=1)[0]
            gen_full_pose = torch.cat([anchor_data["prefix"]['pose'], gen_suffix['pose']], dim=1)[0]
            gen_full_trans = torch.cat([anchor_data["prefix"]['trans'], gen_suffix['trans']], dim=1)[0]
            
            seq_gt = forward_to_h36m(gt_full_pose, gt_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            seq_prior = forward_to_h36m(prior_full_pose, prior_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            seq_gen = forward_to_h36m(gen_full_pose, gen_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            
            gif_path = self.vis_dir / f"anchor_class_{sev_val}_epoch_{display_epoch}.gif"
            render_three_way_gif(seq_gt, seq_prior, seq_gen, sev_val, gif_path, elev=55, azim=55, roll=135, gen_severity=gen_sev_val)
            gif_paths.append(gif_path)
        print(f"  [Time] Anchor GIF Rendering: {time.time() - gif_start:.2f}s")

        # Log all metrics and GIFs to W&B
        if wandb.run is not None:
            for p in gif_paths:
                wandb_logs[f"eval_videos/{p.stem}"] = wandb.Video(str(p), format="gif")
            trainer.logger.experiment.log(wandb_logs, step=trainer.global_step)

        print(f"  [Time] TOTAL Validation Routine: {time.time() - val_start_time:.2f}s\n")
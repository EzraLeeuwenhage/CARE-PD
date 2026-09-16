from unittest import case

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

from thesis.src.model_backbones import generate_x0
from thesis.src.generate_prior import generate_label_prior
from thesis.src.sample import generate_trajectories
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.utils.pipeline_utils import (
    format_and_convert, 
    evaluate_and_plot_distributions, 
    plot_physical_realism_tracking,
    render_anchor_gifs,
)


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
            print(f"Epoch {epoch:03d} | Train Loss: {loss_val}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.val_interval == 0:
            mpjae = trainer.callback_metrics.get("val/mpjae_deg")
            mpjae_val = f"{mpjae.item():.2f}" if mpjae is not None else "N/A"
            print(f"Epoch {epoch:03d} | Val MPJAE: {mpjae_val} deg")


class WandBEvaluationCallback(Callback):
    """Evaluates generated distributions and renders visualizations/ example sequence GIFs."""
    def __init__(self, cfg, eval_interval=50):
        super().__init__()
        self.cfg = cfg
        self.eval_interval = eval_interval
        self.is_joint_model = cfg['model'].get('is_joint_model', False)
        self.gen_mode = cfg['model'].get('generation_mode', 'ar_rollout')
        self.anchors = {}
        
        self.val_epochs = []
        self.floating_gen_hist = []
        self.floating_gt_hist = []
        self.foot_disp_gen_hist = []
        self.foot_disp_gt_hist = []
        
        self.cache_dir = Path(cfg['paths']['output_dir']) / "wandb_eval_cache"
        self.vis_dir = self.cache_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.smpl_model = SMPL(model_path='thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl', num_betas=10).eval()
        self.h36m_regressor = torch.tensor(np.load('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy'), dtype=torch.float32)
        self.smpl_evaluator = SMPLEvaluator(fps=30)

    def _sample_anchors(self, trainer, pl_module):
        """Samples and caches 1 anchor sequence per class.
        
        For one-shot generation, pre-computes x_0 for each anchor. 
        For AR rollout, x_0 cannot be pre-computed and will be generated per window.
        """
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list): 
            val_loader = val_loader[0]

        print("\n[W&B Callback] Sampling Anchor Sequences across Severity Classes...")
        candidates = defaultdict(list)
        for batch in val_loader:
            batch_size = batch['severity'].shape[0]
            for batch_idx in range(batch_size):
                sev_val = batch['severity'][batch_idx].item()
               
                single_sequence = {
                    k: (v[batch_idx:batch_idx+1].cpu() if torch.is_tensor(v) 
                        else (v[batch_idx] if isinstance(v, list) else v))
                    for k, v in batch.items()
                }
                candidates[sev_val].append(single_sequence)
                
        # Randomly select one anchor sequence per severity class
        for sev_val in sorted(candidates.keys()):
            single_sequence = {
                k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                for k, v in random.choice(candidates[sev_val]).items()
            }
            
            # Generate and store a fixed random seed
            seed = random.randint(0, 1024)
            gen = torch.Generator(device=pl_module.device).manual_seed(seed)
            
            # Pre-compute x_0 for one-shot, keep None for AR rollout (will be generated per window)
            x_0 = None
            if self.gen_mode == 'one_shot':
                x_0 = generate_x0(single_sequence, pl_module.prefix_len, pl_module.prior_noise_scale, generator=gen)
                
            y_0 = generate_label_prior(1, self.cfg['model'].get('num_classes', 4), pl_module.device, generator=gen)

            self.anchors[sev_val] = {
                "batch": single_sequence,
                "seed": seed,
                "x_0": x_0,
                "y_0": y_0
            }
            print(f"  -> Locked Random Anchor (Seed: {seed}) for Severity Class {sev_val}")

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
        flat_gt_pose = torch.cat(data_dict["gt"]["pose"], dim=1)
        flat_gen_pose = torch.cat(data_dict["gen"]["pose"], dim=1)
        
        mpjae_rad = self.smpl_evaluator.compute_mpjae(flat_gt_pose, flat_gen_pose)
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

            self.val_epochs.append(display_epoch)
            self.floating_gt_hist.append(dist_metrics["physical_realism/mean_floating_gt"])
            self.floating_gen_hist.append(dist_metrics["physical_realism/mean_floating_gen"])
            self.foot_disp_gt_hist.append(dist_metrics["physical_realism/mean_foot_disp_gt"])
            self.foot_disp_gen_hist.append(dist_metrics["physical_realism/mean_foot_disp_gen"])

            tracking_path = plot_physical_realism_tracking(
                self.val_epochs, self.floating_gt_hist, self.floating_gen_hist,
                self.foot_disp_gt_hist, self.foot_disp_gen_hist, self.vis_dir
            )

            wandb_logs.update(dist_metrics)
            if wandb.run is not None:
                wandb_logs["eval_visuals/physical_realism_tracking"] = wandb.Image(str(tracking_path))
                for img_path in vis_dir.glob("*.png"):
                    wandb_logs[f"eval_visuals/{img_path.stem}"] = wandb.Image(str(img_path))

        # Render Anchor GIFs
        gif_start = time.time()
        self.smpl_model = self.smpl_model.to(pl_module.device)
        self.h36m_regressor = self.h36m_regressor.to(pl_module.device)
        
        gif_paths = render_anchor_gifs(
            anchors=self.anchors,
            pl_module=pl_module,
            is_joint_model=self.is_joint_model,
            vis_dir=self.vis_dir,
            smpl_model=self.smpl_model,
            h36m_regressor=self.h36m_regressor,
            display_epoch=display_epoch
        )

        print(f"  [Time] Anchor GIF Rendering: {time.time() - gif_start:.2f}s")

        # Log all metrics and GIFs to W&B
        if wandb.run is not None:
            for p, sev_val in zip(gif_paths, self.anchors.keys()):
                wandb_logs[f"eval_videos/anchor_class_{sev_val}"] = wandb.Video(str(p), format="gif")
                 
            trainer.logger.experiment.log(wandb_logs, step=trainer.global_step)

        print(f"  [Time] TOTAL Validation Routine: {time.time() - val_start_time:.2f}s\n")
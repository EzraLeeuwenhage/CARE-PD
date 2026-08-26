import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from smplx.body_models import SMPL

from thesis.src.sample import generate_trajectories
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.evaluate_distributions import DistributionComparator
from thesis.src.generate_prior import generate_prior_from_prefix
from thesis.utils.pipeline_utils import forward_6d_to_h36m
from thesis.utils.visualize_h36m_metric_dist import (
    plot_dataset_summary_stats,
    plot_pd_feature_violins,
    plot_pd_feature_comparison_plots,
    prepare_dataframe,
    prepare_combined_dataframe
)
from thesis.utils.visualize_smpl_metric_dist import (
    plot_smpl_mpjae,
    plot_arm_swing_metrics,
    plot_sparc_metrics
)
from thesis.utils.render_h36m_gif import render_three_way_gif


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
        if trainer.sanity_checking:
            return
            
        epoch = trainer.current_epoch + 1
        if epoch % self.val_interval == 0:
            mpjae = trainer.callback_metrics.get("val/mpjae_rad")
            acc = trainer.callback_metrics.get("val/label_accuracy")
            
            mpjae_str = f"{mpjae.item():.4f} rad" if mpjae is not None else "N/A"
            
            if acc is not None:
                acc_str = f"{acc.item():.4f}"
                print(f" >>> VALIDATION Epoch {epoch:04d} | MPJAE: {mpjae_str} | Label Acc: {acc_str}")
            else:
                print(f" >>> VALIDATION Epoch {epoch:04d} | MPJAE: {mpjae_str}")


class WandBEvaluationCallback(Callback):
    """Evaluates generated distributions, logs scalar baselines, visual plots, and Anchor GIFs entirely in RAM."""
    def __init__(self, cfg, eval_interval=50):
        super().__init__()
        self.cfg = cfg
        self.eval_interval = eval_interval
        self.is_joint_model = cfg['model'].get('is_joint_model', False)
        self.force_joint_cond = cfg['sampling'].get('force_joint_conditioning', False)
        
        self.cache_dir = Path(cfg['paths']['output_dir']) / "wandb_eval_cache"
        self.vis_dir = self.cache_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.anchors = {}
        self.gt_6d_dict = None
        self.gt_h36m_dict = None
        self.gt_key_to_severity = None
        self.gt_h36m_data = None
        self.h36m_evaluator = H36MEvaluator(fps=30)
        self.smpl_evaluator = SMPLEvaluator(fps=30)
        self.comparator = DistributionComparator()

        self.smpl_model = SMPL(model_path='thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl', num_betas=10).eval()
        self.h36m_regressor = torch.tensor(np.load('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy'), 
                                           dtype=torch.float32)

    def on_train_start(self, trainer, pl_module):
        """Randomly samples and freezes exactly 1 Anchor sequence per severity class (0, 1, 2, 3)."""
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
            
        print("\n[W&B Callback] Randomly Sampling Anchor Sequences across Severity Classes...")
        
        candidates = defaultdict(list)
        for prefix, target, severity in val_loader:
            for b_idx in range(severity.shape[0]):
                sev_val = severity[b_idx].item()
                # Keep temps on CPU during sampling
                pref_single = {k: v[b_idx:b_idx+1].cpu() for k, v in prefix.items()}
                targ_single = {k: v[b_idx:b_idx+1].cpu() for k, v in target.items()}
                candidates[sev_val].append((pref_single, targ_single))
                
        # pick one random anchor per severity class
        for sev_val in sorted(candidates.keys()):
            pref_single, targ_single = random.choice(candidates[sev_val])

            # Move anchors to device
            pref_single = {k: v.to(pl_module.device) for k, v in pref_single.items()}
            targ_single = {k: v.to(pl_module.device) for k, v in targ_single.items()}
            
            x_0 = generate_prior_from_prefix(pref_single, targ_single)
            self.anchors[sev_val] = {
                "prefix": pref_single,
                "x_0": x_0,
                "target": targ_single,
                "severity": sev_val
            }
            print(f"  -> Locked Random Anchor for Severity Class {sev_val}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if trainer.sanity_checking or epoch % self.eval_interval != 0:
            return

        print(f"\n--- [W&B Callback] Running Validation (Epoch {epoch}) ---")
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        # Generate Synthetic Suffixes
        data_dict = generate_trajectories(
            model=pl_module, 
            dataloader=val_loader, 
            num_steps=self.cfg['sampling']['num_steps'], 
            device=pl_module.device,
            max_batches=self.cfg['training'].get('eval_batches', -1),
            desc=f"W&B Eval Ep {epoch}", 
            is_joint_model=self.is_joint_model,
            force_joint_conditioning=self.force_joint_cond
        )

        # Convert 6D to H36M and SMPL
        self.smpl_model = self.smpl_model.to(pl_module.device)
        self.h36m_regressor = self.h36m_regressor.to(pl_module.device)

        gen_6d_dict, gen_h36m_dict, gen_key_to_severity = {}, {}, {}
        gen_sevs_list = data_dict["gen_severities"] if self.is_joint_model else data_dict["severities"]

        # Only compute GT metrics once and cache in memory for future validation epochs
        if self.gt_6d_dict is None:
            self.gt_6d_dict, self.gt_h36m_dict, self.gt_key_to_severity = {}, {}, {}

            for i, gt_sev in enumerate(data_dict["severities"]):
                seq_key = f"seq_{i:03d}"
                self.gt_key_to_severity[seq_key] = gt_sev
                
                pose_gt_t = data_dict["gt"]["pose"][i]
                trans_gt_t = data_dict["gt"]["trans"][i]

                self.gt_6d_dict[seq_key] = pose_gt_t.cpu().numpy()
                self.gt_6d_dict[f"{seq_key}_trans"] = trans_gt_t.cpu().numpy()
                self.gt_h36m_dict[seq_key] = forward_6d_to_h36m(pose_gt_t, trans_gt_t, self.smpl_model, self.h36m_regressor, pl_module.device)

            # Extract GT metrics once and hold in memory forever
            self.gt_h36m_data, _ = self.h36m_evaluator.evaluate_from_memory(self.gt_h36m_dict, self.gt_key_to_severity)

        for i, gen_sev in enumerate(gen_sevs_list):
            seq_key = f"seq_{i:03d}"
            gen_key_to_severity[seq_key] = gen_sev

            pose_gen_t = data_dict["gen"]["pose"][i]
            trans_gen_t = data_dict["gen"]["trans"][i]

            gen_6d_dict[seq_key] = pose_gen_t.cpu().numpy()
            gen_6d_dict[f"{seq_key}_trans"] = trans_gen_t.cpu().numpy()
            gen_h36m_dict[seq_key] = forward_6d_to_h36m(pose_gen_t, trans_gen_t, self.smpl_model, self.h36m_regressor, pl_module.device)

        # Extract Generated Metrics
        gen_h36m_data, _ = self.h36m_evaluator.evaluate_from_memory(gen_h36m_dict, gen_key_to_severity)
        
        smpl_summary, smpl_cache_data = self.smpl_evaluator.evaluate_from_memory(
            self.gt_6d_dict, gen_6d_dict, self.gt_key_to_severity, verbose=False
        )

        # H36M Distribution Distances & Plots
        h36m_results = self.comparator.compare(self.gt_h36m_data, gen_h36m_data)
        h36m_dist_df = self.comparator._format_results_to_dataframe(h36m_results)
        
        gen_df = prepare_dataframe(gen_h36m_data)
        combined_df = prepare_combined_dataframe(self.gt_h36m_data, gen_h36m_data)
        
        plot_dataset_summary_stats(gen_df, self.vis_dir, prefix="gen_")
        plot_pd_feature_violins(gen_df, self.vis_dir, prefix="gen_")
        plot_pd_feature_comparison_plots(combined_df, h36m_dist_df, self.vis_dir)

        # SMPL Distribution Distances & Plots
        gt_comp, gen_comp = defaultdict(dict), defaultdict(dict)
        target_sparc_joints = ['L_Knee', 'R_Knee']
        
        for sev_key, metrics in smpl_cache_data.get("raw_distributions", {}).items():
            c_key = "overall" if sev_key == "Overall" else sev_key.replace("Class ", "")
            
            gt_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("GT_Symmetry_Index", []))
            gen_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("Gen_Symmetry_Index", []))
            
            gt_knees, gen_knees = [], []
            for j in target_sparc_joints:
                gt_knees.extend(metrics.get(f"GT_SPARC_{j}", []))
                gen_knees.extend(metrics.get(f"Gen_SPARC_{j}", []))
            gt_comp[c_key]["SPARC_Knees"] = np.array(gt_knees)
            gen_comp[c_key]["SPARC_Knees"] = np.array(gen_knees)
            
        smpl_dist_df = self.comparator._format_results_to_dataframe(self.comparator.compare(gt_comp, gen_comp))

        plot_smpl_mpjae(smpl_cache_data, self.vis_dir)
        plot_arm_swing_metrics(smpl_cache_data, self.vis_dir, distances_df=smpl_dist_df)
        plot_sparc_metrics(smpl_cache_data, self.vis_dir, distances_df=smpl_dist_df)

        # Render Side-by-Side Anchor GIFs (Prior vs Generated Suffix)
        gif_paths = []
        for sev_val, anchor_data in self.anchors.items():
            if self.is_joint_model:
                sev_tensor = torch.tensor([sev_val]).to(pl_module.device) if self.force_joint_cond else None
                gen_suffix, gen_severity = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], severity_score=sev_tensor, num_steps=self.cfg['sampling']['num_steps']
                )
                gen_sev_val = gen_severity[0].item()
            else:
                gen_suffix = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], severity_score=torch.tensor([sev_val]).to(pl_module.device), num_steps=self.cfg['sampling']['num_steps']
                )
                gen_sev_val = sev_val
                
            gt_full_pose = torch.cat([anchor_data["prefix"]['pose'], anchor_data["target"]['pose']], dim=1)[0]
            gt_full_trans = torch.cat([anchor_data["prefix"]['trans'], anchor_data["target"]['trans']], dim=1)[0]
            
            prior_full_pose = torch.cat([anchor_data["prefix"]['pose'], anchor_data["x_0"]['pose']], dim=1)[0]
            prior_full_trans = torch.cat([anchor_data["prefix"]['trans'], anchor_data["x_0"]['trans']], dim=1)[0]
            
            gen_full_pose = torch.cat([anchor_data["prefix"]['pose'], gen_suffix['pose']], dim=1)[0]
            gen_full_trans = torch.cat([anchor_data["prefix"]['trans'], gen_suffix['trans']], dim=1)[0]
            
            seq_gt = forward_6d_to_h36m(gt_full_pose, gt_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            seq_prior = forward_6d_to_h36m(prior_full_pose, prior_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            seq_gen = forward_6d_to_h36m(gen_full_pose, gen_full_trans, self.smpl_model, self.h36m_regressor, pl_module.device)
            
            gif_path = self.vis_dir / f"anchor_class_{sev_val}_epoch_{epoch}.gif"
            render_three_way_gif(seq_gt, seq_prior, seq_gen, sev_val, gif_path, elev=20, azim=45, roll=135, gen_severity=gen_sev_val)
            gif_paths.append(gif_path)

        # Log all metrics, distances, and visuals to W&B
        gt_floating = float(np.nanmean(self.gt_h36m_data["overall"]["floating"]))
        gen_floating = float(np.nanmean(gen_h36m_data["overall"]["floating"]))
        gt_foot_disp = float(np.nanmean(self.gt_h36m_data["overall"]["mean_stance_displacement"]))
        gen_foot_disp = float(np.nanmean(gen_h36m_data["overall"]["mean_stance_displacement"]))

        wandb_logs = {
            "physical_realism/floating_gen": gen_floating,
            "physical_realism/floating_gt": gt_floating,
            "physical_realism/floating_error_abs": abs(gen_floating - gt_floating),
            
            "physical_realism/foot_displacement_gen": gen_foot_disp,
            "physical_realism/foot_displacement_gt": gt_foot_disp,
            "physical_realism/foot_displacement_error_abs": abs(gen_foot_disp - gt_foot_disp),
            
            "eval_metrics/Overall_MPJAE_rad": smpl_summary.get("Overall", {}).get("Overall", 0.0),
            "eval_metrics/Mean_Hellinger_H36M": float(h36m_dist_df["Hellinger"].mean()),
            "eval_metrics/Mean_KS_H36M": float(h36m_dist_df["KS_Stat"].mean()),
            "eval_metrics/Mean_Hellinger_SMPL": float(smpl_dist_df["Hellinger"].mean()),
            "eval_metrics/Mean_KS_SMPL": float(smpl_dist_df["KS_Stat"].mean()),
        }
        
        for gif_path in gif_paths:
            wandb_logs[f"eval_videos/{gif_path.stem}"] = wandb.Video(str(gif_path), format="gif")

        visual_artifacts = {
            "eval_visuals/H36M_Summary_Card": "gen_00_dataset_summary.png",
            "eval_visuals/H36M_Features_Violin": "gen_02_pd_features_summary.png",
            "eval_visuals/H36M_Ankle_Clearance": "02b_max_ankle_clearance_comparison_plot.png",
            "eval_visuals/H36M_eMoS": "02b_mean_emos_comparison_plot.png",
            "eval_visuals/H36M_Step_Length": "02b_mean_step_length_comparison_plot.png",
            "eval_visuals/H36M_Walking_Speed": "02b_mean_walking_speed_comparison_plot.png",
            "eval_visuals/SMPL_MPJAE_Categories": "03a_smpl_mpjae_categories.png",
            "eval_visuals/SMPL_MPJAE_All_Joints": "03b_smpl_mpjae_all_24_joints.png",
            "eval_visuals/SMPL_Arm_Swing_SI": "03c_smpl_arm_swing_distributions.png",
            "eval_visuals/SMPL_SPARC_Categories": "03d_sparc_categories.png",
            "eval_visuals/SMPL_SPARC_All_Joints": "03e_sparc_all_24_joints.png",
            "eval_visuals/SMPL_SPARC_Knee_Discriminators": "03f_sparc_knee_class_discriminators.png",
        }

        for log_name, filename in visual_artifacts.items():
            img_path = self.vis_dir / filename
            if img_path.exists():
                wandb_logs[log_name] = wandb.Image(str(img_path))

        trainer.logger.experiment.log(wandb_logs, step=trainer.global_step)
        
        for f in self.vis_dir.glob("*.png"): f.unlink()
        for f in self.vis_dir.glob("*.gif"): f.unlink()
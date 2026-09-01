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
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.evaluate_distributions import DistributionComparator
from thesis.src.generate_prior import generate_prior_from_prefix
from thesis.utils.pipeline_utils import forward_6d_to_h36m, batched_6d_to_h36m
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
        
        self.cache_dir = Path(cfg['paths']['output_dir']) / "wandb_eval_cache"
        self.vis_dir = self.cache_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.gt_plots_logged = False

        self.val_epochs = []
        self.floating_gen_hist = []
        self.floating_gt_hist = []
        self.foot_disp_gen_hist = []
        self.foot_disp_gt_hist = []
        
        self.anchors = {}
        self.gt_6d_dict = None
        self.gt_h36m_dict = None
        self.gt_key_to_severity = None
        self.gt_h36m_data = None

        min_z_travel = cfg['windowing'].get('min_z_travel', 0.5)
        self.h36m_evaluator = H36MEvaluator(fps=30, min_z_travel=min_z_travel)
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

        val_start_time = time.time()

        print(f"\n--- [W&B Callback] Running Validation (Epoch {epoch}) ---")
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        # Generate Synthetic Suffixes
        gen_start = time.time()
        data_dict = generate_trajectories(
            model=pl_module, 
            dataloader=val_loader, 
            num_steps=self.cfg['sampling']['num_steps'], 
            device=pl_module.device,
            max_batches=self.cfg['training'].get('eval_batches', -1),
            desc=f"W&B Eval Ep {epoch}", 
            is_joint_model=self.is_joint_model,
        )
        print(f"  [Time] Trajectory Generation: {time.time() - gen_start:.2f}s")

        # Convert 6D to H36M and SMPL
        self.smpl_model = self.smpl_model.to(pl_module.device)
        self.h36m_regressor = self.h36m_regressor.to(pl_module.device)

        gen_6d_dict, gen_h36m_dict, gen_key_to_severity = {}, {}, {}
        gen_sevs_list = data_dict["gen_severities"] if self.is_joint_model else data_dict["severities"]

        # Only compute GT metrics once and cache in memory for future validation epochs
        conv_start = time.time()
        if self.gt_6d_dict is None:
            self.gt_6d_dict, self.gt_h36m_dict, self.gt_key_to_severity = {}, {}, {}

            gt_h36m_all = batched_6d_to_h36m(
                data_dict["gt"]["pose"], 
                data_dict["gt"]["trans"],
                self.smpl_model,
                self.h36m_regressor, 
                pl_module.device
            )

            for i, gt_sev in enumerate(data_dict["severities"]):
                seq_key = f"seq_{i:03d}"
                self.gt_key_to_severity[seq_key] = gt_sev
                self.gt_6d_dict[seq_key] = data_dict["gt"]["pose"][i].cpu().numpy()
                self.gt_6d_dict[f"{seq_key}_trans"] = data_dict["gt"]["trans"][i].cpu().numpy()
                self.gt_h36m_dict[seq_key] = gt_h36m_all[i]

            # Extract GT metrics once and hold in memory forever
            self.gt_h36m_data, _ = self.h36m_evaluator.evaluate_from_memory(self.gt_h36m_dict, self.gt_key_to_severity)

        gen_h36m_all = batched_6d_to_h36m(
            data_dict["gen"]["pose"], 
            data_dict["gen"]["trans"],
            self.smpl_model,
            self.h36m_regressor,  
            pl_module.device
        )

        for i, gen_sev in enumerate(gen_sevs_list):
            seq_key = f"seq_{i:03d}"
            gen_key_to_severity[seq_key] = gen_sev
            gen_6d_dict[seq_key] = data_dict["gen"]["pose"][i].cpu().numpy()
            gen_6d_dict[f"{seq_key}_trans"] = data_dict["gen"]["trans"][i].cpu().numpy()
            gen_h36m_dict[seq_key] = gen_h36m_all[i]

        print(f"  [Time] SMPL to H36M Batch Conversion: {time.time() - conv_start:.2f}s")

        # Extract Generated Metrics 
        metric_start = time.time()       
        smpl_summary, smpl_cache_data = self.smpl_evaluator.evaluate_from_memory(
            self.gt_6d_dict, gen_6d_dict, self.gt_key_to_severity
        )

        is_overfit = self.cfg['training'].get('overfit_severity_class', -1) >= 0

        # H36M Distribution Distances & Plots
        if not is_overfit:
            gen_h36m_data, _ = self.h36m_evaluator.evaluate_from_memory(gen_h36m_dict, gen_key_to_severity)

            h36m_results = self.comparator.compare(self.gt_h36m_data, gen_h36m_data)
            h36m_dist_df = self.comparator._format_results_to_dataframe(h36m_results)

            # only plot ground-truth data plots a single time
            if not self.gt_plots_logged:                
                gt_df = prepare_dataframe(self.gt_h36m_data)
                plot_dataset_summary_stats(gt_df, self.vis_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
                plot_pd_feature_violins(gt_df, self.vis_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
                self.gt_plots_logged = True
            
            gen_df = prepare_dataframe(gen_h36m_data)            
            plot_dataset_summary_stats(gen_df, self.vis_dir, prefix="gen_", dataset_label=f"Generated (Epoch {epoch})")
            plot_pd_feature_violins(gen_df, self.vis_dir, prefix="gen_", dataset_label=f"Generated (Epoch {epoch})")

            combined_df = prepare_combined_dataframe(self.gt_h36m_data, gen_h36m_data)
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
        
        print(f"  [Time] Metric Extraction & Plots: {time.time() - metric_start:.2f}s")

        # Render Side-by-Side Anchor GIFs (Prior vs Generated Suffix)
        gif_start = time.time()
        gif_paths = []
        for sev_val, anchor_data in self.anchors.items():
            if self.is_joint_model:
                gen_suffix, gen_severity = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], severity_score=None, 
                    num_steps=self.cfg['sampling']['num_steps']
                )
                gen_sev_val = gen_severity[0].item()
            else:
                gen_suffix = pl_module.generate_suffix(
                    anchor_data["prefix"], anchor_data["x_0"], 
                    severity_score=torch.tensor([sev_val]).to(pl_module.device), 
                    num_steps=self.cfg['sampling']['num_steps']
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
            render_three_way_gif(seq_gt, seq_prior, seq_gen, sev_val, gif_path, elev=55, azim=55, roll=135, gen_severity=gen_sev_val)
            gif_paths.append(gif_path)

        print(f"  [Time] Anchor GIF Rendering: {time.time() - gif_start:.2f}s")

        # Standard wandb logs
        wandb_logs = {
            "eval_metrics/Overall_MPJAE_rad": smpl_summary.get("Overall", {}).get("Overall", 0.0)
        }

        if self.is_joint_model:
            wandb_logs["eval_metrics/Label_Confusion_Matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=data_dict["severities"],
                preds=data_dict["gen_severities"],
                class_names=["Class 0", "Class 1", "Class 2", "Class 3"],
                title=f"Label Adherence Matrix (Epoch {epoch})"
            )
            
            wandb_logs["eval_metrics/Prior_State_Correlation_Matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=data_dict["prior_severities"],
                preds=data_dict["gen_severities"],
                class_names=["Class 0", "Class 1", "Class 2", "Class 3"],
                title=f"Prior Jump Correlation Matrix (Epoch {epoch})"
            )
        
        for gif_path in gif_paths:
            wandb_logs[f"eval_videos/{gif_path.stem}"] = wandb.Video(str(gif_path), format="gif")

        # Distribution metrics logs (only if not overfitting on single sequence)
        if not is_overfit:
            gt_floating = float(np.nanmean(self.gt_h36m_data["overall"]["floating"]))
            gen_floating = float(np.nanmean(gen_h36m_data["overall"]["floating"]))
            gt_foot_disp = float(np.nanmean(self.gt_h36m_data["overall"]["mean_stance_displacement"]))
            gen_foot_disp = float(np.nanmean(gen_h36m_data["overall"]["mean_stance_displacement"]))

            self.val_epochs.append(epoch)
            self.floating_gt_hist.append(gt_floating)
            self.floating_gen_hist.append(gen_floating)
            self.foot_disp_gt_hist.append(gt_foot_disp)
            self.foot_disp_gen_hist.append(gen_foot_disp)

            wandb_logs.update({
                "physical_realism/floating_tracking": wandb.plot.line_series(
                    xs=self.val_epochs,
                    ys=[self.floating_gen_hist, self.floating_gt_hist],
                    keys=["Generated", "Ground Truth Baseline"],
                    title="Floating / Skating over Epochs",
                    xname="Epoch"
                ),
                "physical_realism/foot_displacement_tracking": wandb.plot.line_series(
                    xs=self.val_epochs,
                    ys=[self.foot_disp_gen_hist, self.foot_disp_gt_hist],
                    keys=["Generated", "Ground Truth Baseline"],
                    title="Foot Displacement over Epochs",
                    xname="Epoch"
                ),
                "physical_realism/floating_error_abs": abs(gen_floating - gt_floating),
                "physical_realism/foot_displacement_error_abs": abs(gen_foot_disp - gt_foot_disp),
                
                "eval_metrics/Mean_Hellinger_H36M": float(h36m_dist_df["Hellinger"].mean()),
                "eval_metrics/Mean_KS_H36M": float(h36m_dist_df["KS_Stat"].mean()),
                "eval_metrics/Mean_Hellinger_SMPL": float(smpl_dist_df["Hellinger"].mean()),
                "eval_metrics/Mean_KS_SMPL": float(smpl_dist_df["KS_Stat"].mean()),
            })

            visuals = {
                "eval_visuals/H36M_Summary_Card_GEN": "gen_00_dataset_summary.png",
                "eval_visuals/H36M_Summary_Card_GT": "gt_00_dataset_summary.png",
                "eval_visuals/H36M_Features_Violin_GEN": "gen_02_pd_features_summary.png",
                "eval_visuals/H36M_Features_Violin_GT": "gt_02_pd_features_summary.png",
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

            for log_name, filename in visuals.items():
                img_path = self.vis_dir / filename
                if img_path.exists():
                    wandb_logs[log_name] = wandb.Image(str(img_path))

        trainer.logger.experiment.log(wandb_logs, step=trainer.global_step)
        
        for f in self.vis_dir.glob("*.png"): f.unlink()
        for f in self.vis_dir.glob("*.gif"): f.unlink()

        print(f"  [Time] TOTAL Validation Routine: {time.time() - val_start_time:.2f}s\n")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.loss import KinematicWeightedPoseLoss, JOINT_MPJAE_WEIGHTS
from thesis.src.model_backbones import (
    ConditionalBaselineBackbone, 
    FlowHead,
    generate_x0,
    mask,
    add,
    sub, 
    mul,
    add_noise
)


class ConditionalBaselineModel(pl.LightningModule):
    """Baseline conditional generator model for comparison against better backbones and joint models."""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.save_hyperparameters({
            'cfg': cfg,
            'kinematic_weights': JOINT_MPJAE_WEIGHTS
        })
        self.cfg = cfg
        self.gen_mode = cfg['model'].get('generation_mode', 'ar_rollout')
        self.lr = cfg['training'].get('learning_rate', 0.001)
        self.alpha_trans = cfg['training'].get('alpha_trans', 0.15) 
        self.alpha_label = cfg['training'].get('alpha_label', 0.50)
        self.num_steps = cfg['sampling'].get('num_steps', 100)

        self.AR_window_size = self.cfg['windowing'].get('total_window_size', 60)
        self.prefix_len = cfg['windowing'].get('prefix_length', 15)
        self.prior_noise_scale = cfg['training'].get('prior_noise_scale', 1.0)
        self.use_cumsum_prior = cfg['training'].get('use_cumsum_prior', False)
        self.path_smoothing_noise = cfg['training'].get('path_smoothing_noise', 0.005)
        self.prefix_noise_std = self.cfg['training'].get('prefix_noise_std', 0.05)

        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        
        self.backbone = ConditionalBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.seq_len, self.backbone.num_joints, self.backbone.pose_dim)
        self.evaluator = SMPLEvaluator()

        # TODO: fix this
        pose_loss_scale = self.cfg['training'].get('pose_loss_scale', 5.0)
        self.pose_loss_fn = KinematicWeightedPoseLoss(
            scale_factor=pose_loss_scale, 
            pose_dim=self.backbone.pose_dim
        )

    def forward(self, x_tau_dict, tau, severity_score):
        shared_latent = self.backbone(x_tau_dict, tau, severity_score)
        return self.flow_head(shared_latent)

    def _compute_ar_rollout_loss(self, batch):
        """Autoregressive Windowed Flow (AR-WG) Loss Computation."""
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        batch_size = severity_score.shape[0]

        M_cond = batch['cond_mask']
        M_targ = ~M_cond

        # Sample prior (x_0) and FM time (tau)
        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale, use_cumsum=self.use_cumsum_prior)
        tau = torch.rand(batch_size, 1, device=self.device)

        # Add noise to prefix frames to mitigate exposure bias
        x_1_noisy = add_noise(x_1, self.prefix_noise_std)

        # Define x_tau: (x_1 + eps) * M_cond + ((1 - tau) * x_0 + tau * x_1) * M_targ
        x_tau_cond = mask(x_1_noisy, M_cond)
        x_tau_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(x_tau_cond, x_tau_targ)

        # TODO: optimize this for smoothing velocity field
        x_tau = add_noise(x_tau, std=self.path_smoothing_noise)

        # Target Vector Field: (x_1 - x_0) * M_targ
        u_true = mask(sub(x_1, x_0), M_targ)

        # Predict and compute loss
        v_pred = self(x_tau, tau, severity_score)
        v_pred_masked = mask(v_pred, M_targ)

        # _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        # valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = self.pose_loss_fn(v_pred_masked['pose'], u_true['pose'], M_targ)
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements

        # Quick sanity check
        with torch.no_grad():
            # magnitude similarty to rule out 0 velocity prediction
            v_mag = torch.norm(v_pred_masked['pose'], p=2, dim=-1).mean()
            u_mag = torch.norm(u_true['pose'], p=2, dim=-1).mean()
            self.log("diag/v_pred_mag", v_mag, on_step=False, on_epoch=True)
            self.log("diag/u_true_mag", u_mag, on_step=False, on_epoch=True)

            # similarity of direction of velocities to rule out vectors pointing in arbitraty directions
            v_flat = v_pred_masked['pose'].reshape(batch_size, -1)
            u_flat = u_true['pose'].reshape(batch_size, -1)
            cos_sim = F.cosine_similarity(v_flat, u_flat, dim=-1).mean()
            self.log("diag/cos_sim", cos_sim, on_step=False, on_epoch=True)

            # Translation diagnostics
            v_trans_flat = v_pred_masked['trans'].reshape(batch_size, -1)
            u_trans_flat = u_true['trans'].reshape(batch_size, -1)
            
            trans_cos_sim = F.cosine_similarity(v_trans_flat, u_trans_flat, dim=-1).mean()
            self.log("diag/trans_cos_sim", trans_cos_sim, on_step=False, on_epoch=True)
            self.log("diag/v_trans_mag", torch.norm(v_trans_flat, dim=-1).mean(), on_step=False, on_epoch=True)
            self.log("diag/u_trans_mag", torch.norm(u_trans_flat, dim=-1).mean(), on_step=False, on_epoch=True)
        
        return loss_pose, loss_trans

    def _compute_one_shot_loss(self, batch):
        """One-Shot Padded Flow (OS-SG) Loss Computation."""
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        batch_size = severity_score.shape[0]
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static

        # Sample prior (x_0) and FM time (tau)
        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale, use_cumsum=self.use_cumsum_prior)
        tau = torch.rand(batch_size, 1, device=self.device)

        # Define x_tau: x_1 * M_static + ((1 - tau) * x_0 + tau * x_1) * M_targ
        x_tau_static = mask(x_1, M_static)
        x_tau_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(x_tau_static, x_tau_targ)

        # Target Vector Field: (x_1 - x_0) * M_targ
        u_true = mask(sub(x_1, x_0), M_targ)

        # Predict and compute loss
        v_pred = self(x_tau, tau, severity_score)
        v_pred_masked = mask(v_pred, M_targ)

        _, _, trans_dim = v_pred['trans'].shape
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = self.pose_loss_fn(v_pred_masked['pose'], u_true['pose'], M_targ)
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements

        # Quick sanity check
        with torch.no_grad():
            # magnitude similarty to rule out 0 velocity prediction
            v_mag = torch.norm(v_pred_masked['pose'], p=2, dim=-1).mean()
            u_mag = torch.norm(u_true['pose'], p=2, dim=-1).mean()
            self.log("diag/v_pred_mag", v_mag, on_step=False, on_epoch=True)
            self.log("diag/u_true_mag", u_mag, on_step=False, on_epoch=True)

            # similarity of direction of velocities to rule out vectors pointing in arbitraty directions
            v_flat = v_pred_masked['pose'].reshape(batch_size, -1)
            u_flat = u_true['pose'].reshape(batch_size, -1)
            cos_sim = F.cosine_similarity(v_flat, u_flat, dim=-1).mean()
            self.log("diag/cos_sim", cos_sim, on_step=False, on_epoch=True)

            # Translation diagnostics
            v_trans_flat = v_pred_masked['trans'].reshape(batch_size, -1)
            u_trans_flat = u_true['trans'].reshape(batch_size, -1)
            
            trans_cos_sim = F.cosine_similarity(v_trans_flat, u_trans_flat, dim=-1).mean()
            self.log("diag/trans_cos_sim", trans_cos_sim, on_step=False, on_epoch=True)
            self.log("diag/v_trans_mag", torch.norm(v_trans_flat, dim=-1).mean(), on_step=False, on_epoch=True)
            self.log("diag/u_trans_mag", torch.norm(u_trans_flat, dim=-1).mean(), on_step=False, on_epoch=True)
        
        return loss_pose, loss_trans

    def training_step(self, batch, batch_idx):        
        if self.gen_mode == 'one_shot':
            loss_pose, loss_trans = self._compute_one_shot_loss(batch)
        elif self.gen_mode == 'ar_rollout':
            loss_pose, loss_trans = self._compute_ar_rollout_loss(batch)

        loss_total = ((1.0 - self.alpha_trans) * loss_pose) + (self.alpha_trans * loss_trans)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=False, on_epoch=True)
        
        self.log("train/loss_pose", loss_pose, on_step=False, on_epoch=True)
        self.log("train/loss_trans", loss_trans, on_step=False, on_epoch=True)
        self.log("train/loss_total", loss_total, on_step=False, on_epoch=True)
        return loss_total

    def solve_ODE(self, x_tau, severity_score, M_targ, num_steps=100):
        """Euler ODE Solver strictly masking velocity updates to prevent prefix drift."""
        batch_size = x_tau['pose'].shape[0]
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            
            v_pred = self(x_tau, tau_tensor, severity_score)
            v_pred_masked = mask(v_pred, M_targ)
            
            x_tau = add(x_tau, mul(v_pred_masked, torch.tensor([dt], device=self.device)))
                
        return x_tau

    def _run_ar_inference(self, batch, num_steps=100, generator=None):
        """Generates the target sequence autoregressively using sliding windows."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        batch_size = severity_score.shape[0]
        
        max_seq_length = batch['seq_len'].max().item()
        
        gen_pose = true_dict['pose'][:, :self.prefix_len]
        gen_trans = true_dict['trans'][:, :self.prefix_len]
        num_joints, pose_dim = gen_pose.shape[2], gen_pose.shape[3]
        
        M_cond = torch.zeros((batch_size, self.AR_window_size), dtype=torch.bool, device=self.device)
        M_cond[:, :self.prefix_len] = True
        M_targ = ~M_cond

        # # Calculate NFEs per window to ensure fair comparison with one-shot generation
        # total_target_frames = max_seq_length - self.prefix_len
        # frames_per_window = self.AR_window_size - self.prefix_len
        # num_windows = max(1, math.ceil(total_target_frames / frames_per_window))
        # steps_per_window = max(1, num_steps // num_windows)
        
        while gen_pose.shape[1] < max_seq_length:
            # Create new window with prefix from last (generated) frames
            window_pose = torch.zeros((batch_size, self.AR_window_size, num_joints, pose_dim), device=self.device)
            window_trans = torch.zeros((batch_size, self.AR_window_size, 3), device=self.device)

            window_pose[:, :self.prefix_len] = gen_pose[:, -self.prefix_len:]
            window_trans[:, :self.prefix_len] = gen_trans[:, -self.prefix_len:]
            x1_window = {'pose': window_pose, 'trans': window_trans}
            
            x0_window = generate_x0(
                x1_window, self.prefix_len, self.prior_noise_scale, 
                generator=generator, use_cumsum=self.use_cumsum_prior
            )
            x_tau = add(mask(x1_window, M_cond), mask(x0_window, M_targ))
            
            x1 = self.solve_ODE(x_tau, severity_score, M_targ, num_steps)

            # Concat new generated frames to current sequence total until we pass max sequence length in batch
            gen_pose = torch.cat([gen_pose, x1['pose'][:, self.prefix_len:]], dim=1)
            gen_trans = torch.cat([gen_trans, x1['trans'][:, self.prefix_len:]], dim=1)
        
        return gen_pose[:, :max_seq_length], gen_trans[:, :max_seq_length]

    def _run_oneshot_inference(self, batch, num_steps=100, x_0=None, generator=None):
        """Generates the target sequence globally in a single pass."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static
        
        if x_0 is None:
            x_0 = generate_x0(
                true_dict, self.prefix_len, self.prior_noise_scale, 
                generator=generator, use_cumsum=self.use_cumsum_prior
            )

        x_tau = add(mask(true_dict, M_static), mask(x_0, M_targ))
        gen_dict = self.solve_ODE(x_tau, severity_score, M_targ, num_steps)
        
        return gen_dict['pose'], gen_dict['trans']

    def _compute_window_ar_metrics(self, batch, gen_pose):
        """Computes per window MPJAE and AR error accumulation for AR rollout models."""
        seq_lens = batch['seq_len']
        gen_step = self.AR_window_size - self.prefix_len
        max_windows = (seq_lens.max().item() - self.prefix_len) // gen_step
        if max_windows < 1:
            return {}

        metrics = {}
        win_errs = []
        for k in range(max_windows):
            w_start = self.prefix_len + k * gen_step
            w_end = w_start + gen_step
            valid_mask = seq_lens >= w_end
            if not valid_mask.any():
                continue

            gt_win = batch['pose'][valid_mask, w_start:w_end].cpu()
            gen_win = gen_pose[valid_mask, w_start:w_end].cpu()
            err = self.evaluator.compute_mpjae(gt_win, gen_win) * (180.0 / math.pi)
            metrics[f"val/mpjae_win{k}_deg"] = err
            win_errs.append(err)

        if len(win_errs) >= 2:
            metrics["val/ar_drift_ratio"] = win_errs[1] / (win_errs[0] + 1e-6)
        return metrics

    def validation_step(self, batch, batch_idx):
        """Automated validation step solving AR-WG iterative paths or OS-SG global paths."""
        seq_len = batch['seq_len']
        batch_size = batch['severity'].shape[0]
        
        if self.gen_mode == 'one_shot':
            gen_pose, gen_trans = self._run_oneshot_inference(batch, self.num_steps)
        elif self.gen_mode == 'ar_rollout':
            gen_pose, gen_trans = self._run_ar_inference(batch, self.num_steps)

        # Extract only the valid frames per sequence
        gt_pose_list, gen_pose_list, gt_trans_list, gen_trans_list = [], [], [], []
        for i in range(batch_size):
            l = seq_len[i].item()
            gt_pose_list.append(batch['pose'][i:i+1, self.prefix_len:l])
            gen_pose_list.append(gen_pose[i:i+1, self.prefix_len:l])
            gt_trans_list.append(batch['trans'][i:i+1, self.prefix_len:l])
            gen_trans_list.append(gen_trans[i:i+1, self.prefix_len:l])

        gt_pose_flat = torch.cat(gt_pose_list, dim=1).cpu()
        gen_pose_flat = torch.cat(gen_pose_list, dim=1).cpu()
        gt_trans_flat = torch.cat(gt_trans_list, dim=1)
        gen_trans_flat = torch.cat(gen_trans_list, dim=1)

        val_mpjae_deg = self.evaluator.compute_mpjae(gt_pose_flat, gen_pose_flat) * (180.0 / math.pi)
        val_trans_mse = F.mse_loss(gen_trans_flat, gt_trans_flat)
        
        self.log("val/mpjae_deg", val_mpjae_deg, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/trans_mse", val_trans_mse, on_step=False, on_epoch=True, sync_dist=True)

        if self.gen_mode == 'ar_rollout':
            for k, v in self._compute_window_ar_metrics(batch, gen_pose).items():
                self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)
            
        return val_mpjae_deg

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

      max_epochs = self.cfg["training"].get("epochs", 1000)
      warmup_epochs = self.cfg["training"].get("warmup_epochs", 30)
      eta_min = self.cfg["training"].get("eta_min", 6e-5)

      warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
          optimizer, start_factor=0.1, total_iters=warmup_epochs
      )

      cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=max_epochs - warmup_epochs, eta_min=eta_min
      )

      scheduler = torch.optim.lr_scheduler.SequentialLR(
          optimizer,
          schedulers=[warmup_scheduler, cosine_scheduler],
          milestones=[warmup_epochs],
      )

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "epoch",
              "frequency": 1,
          },
      }
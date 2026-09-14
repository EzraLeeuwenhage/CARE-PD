import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.model_backbones import (
    ConditionalBaselineBackbone, 
    JointBaselineBackbone, 
    FlowHead,
    JumpHead,
    generate_x0,
    mask,
    add,
    sub, 
    mul,
    add_noise
)

# ====================
# MODEL CLASSES
# ====================
def ctmc_jump_step(y_current, Q_pred, dt, num_classes):
    """Simulates a CTMC jump step over time interval dt."""
    Q_pred = Q_pred.float()
    
    # Ensure jump rates to other classes are non-negative
    # and zero out self-transition rate (diagonal of rate matrix Q)
    rates = F.relu(Q_pred)    
    mask = F.one_hot(y_current, num_classes=num_classes).bool()
    rates[mask] = 0.0
    
    # Compute transition probabilities for dt: P(i -> j) = dt * Q_ij (for j != i)
    probs = dt * rates
    
    # Compute self-transition probability: P(i -> i) = 1 - sum(P(i -> j) (for j != i)
    self_prob = (1.0 - probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    probs[mask] = self_prob.squeeze(-1)
    
    # Normalize probabilities to avoid float precision errors and sample next state
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ConditionalBaselineModel(pl.LightningModule):
    """Baseline conditional generator model for comparison against better backbones and joint models."""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.gen_mode = cfg['model'].get('generation_mode', 'ar_rollout')
        self.lr = cfg['training'].get('learning_rate', 0.001)
        self.lambda_pose = cfg['training'].get('lambda_pose', 1.0)
        self.lambda_trans = cfg['training'].get('lambda_trans', 1.0)
        self.num_steps = cfg['sampling'].get('num_steps', 100)

        self.AR_window_size = self.cfg['windowing'].get('total_window_size', 60)
        self.prefix_len = cfg['windowing'].get('prefix_length', 15)
        self.prior_noise_scale = cfg['training'].get('prior_noise_scale', 1.0)

        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        
        self.backbone = ConditionalBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.seq_len, self.backbone.num_joints, self.backbone.pose_dim)
        self.evaluator = SMPLEvaluator()

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
        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
        tau = torch.rand(batch_size, 1, device=self.device)

        # Add noise to prefix frames to mitigate exposure bias
        noise_std = self.cfg['training'].get('prefix_noise_std', 0.05)
        x_1_noisy = add_noise(x_1, noise_std)

        # Define x_tau: (x_1 + eps) * M_cond + ((1 - tau) * x_0 + tau * x_1) * M_targ
        x_tau_cond = mask(x_1_noisy, M_cond)
        x_tau_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(x_tau_cond, x_tau_targ)

        # Target Vector Field: (x_1 - x_0) * M_targ
        u_true = mask(sub(x_1, x_0), M_targ)

        # Predict and compute loss
        v_pred = self(x_tau, tau, severity_score)
        v_pred_masked = mask(v_pred, M_targ)

        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
        
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
        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
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

        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
        
        return loss_pose, loss_trans

    def training_step(self, batch, batch_idx):        
        if self.gen_mode == 'one_shot':
            loss_pose, loss_trans = self._compute_one_shot_loss(batch)
        elif self.gen_mode == 'ar_rollout':
            loss_pose, loss_trans = self._compute_ar_rollout_loss(batch)

        loss_total = (self.lambda_pose * loss_pose) + (self.lambda_trans * loss_trans)
        
        self.log("train/loss_pose", loss_pose, on_step=False, on_epoch=True)
        self.log("train/loss_trans", loss_trans, on_step=False, on_epoch=True)
        self.log("train/loss_total", loss_total, on_step=False, on_epoch=True)
        return loss_total

    def generate_suffix(self, x_tau, severity_score, M_targ, num_steps=100, y_0=None):
        """Euler ODE Solver strictly masking velocity updates to prevent prefix drift."""
        batch_size = x_tau['pose'].shape[0]
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            
            v_pred = self(x_tau, tau_tensor, severity_score)
            v_pred_masked = mask(v_pred, M_targ)
            
            x_tau = add(x_tau, mul(v_pred_masked, torch.tensor([dt], device=self.device)))
                
        return x_tau, None

    def _run_oneshot_inference(self, batch):
        """Generates the target sequence globally in a single pass."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static
        
        x_0 = generate_x0(true_dict, self.prefix_len, self.prior_noise_scale)
        x_tau0 = add(mask(true_dict, M_static), mask(x_0, M_targ))
        
        gen_dict, _ = self.generate_suffix(x_tau0, severity_score, M_targ, num_steps=self.num_steps)
        return gen_dict['pose'], gen_dict['trans']

    def _run_ar_inference(self, batch):
        """Generates the target sequence autoregressively using sliding windows."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        severity_score = batch['severity']
        batch_size = severity_score.shape[0]
        
        # Stop at length of longest sequence in the current batch
        max_seq_length = batch['seq_len'].max().item()
        
        gen_pose = true_dict['pose'][:, :self.prefix_len]
        gen_trans = true_dict['trans'][:, :self.prefix_len]
        num_joints, pose_dim = gen_pose.shape[2], gen_pose.shape[3]
        
        M_cond = torch.zeros((batch_size, self.AR_window_size), dtype=torch.bool, device=self.device)
        M_cond[:, :self.prefix_len] = True
        M_targ = ~M_cond
        
        while gen_pose.shape[1] < max_seq_length:
            # Create new window with prefix from last (generated) frames
            window_pose = torch.zeros((batch_size, self.AR_window_size, num_joints, pose_dim), device=self.device)
            window_trans = torch.zeros((batch_size, self.AR_window_size, 3), device=self.device)

            window_pose[:, :self.prefix_len] = gen_pose[:, -self.prefix_len:]
            window_trans[:, :self.prefix_len] = gen_trans[:, -self.prefix_len:]
            x1_window = {'pose': window_pose, 'trans': window_trans}
            
            x0_window = generate_x0(x1_window, self.prefix_len, self.prior_noise_scale)
            x_tau = add(mask(x1_window, M_cond), mask(x0_window, M_targ))
            
            x1, _ = self.generate_suffix(x_tau, severity_score, M_targ, num_steps=self.num_steps)

            # Concat new generated frames to current sequence total until we pass max sequence length in batch
            gen_pose = torch.cat([gen_pose, x1['pose'][:, self.prefix_len:]], dim=1)
            gen_trans = torch.cat([gen_trans, x1['trans'][:, self.prefix_len:]], dim=1)
        
        return gen_pose[:, :max_seq_length], gen_trans[:, :max_seq_length]

    def validation_step(self, batch, batch_idx):
        """Automated validation step solving AR-WG iterative paths or OS-SG global paths."""
        seq_len = batch['seq_len']
        batch_size = batch['severity'].shape[0]
        
        if self.gen_mode == 'one_shot':
            gen_pose, gen_trans = self._run_oneshot_inference(batch)
        elif self.gen_mode == 'ar_rollout':
            gen_pose, gen_trans = self._run_ar_inference(batch)

        # Extract only the valid frames per sequence to avoid evaluating zero-padding
        gt_pose_list = []
        gen_pose_list = []
        gt_trans_list = []
        gen_trans_list = []
        
        for i in range(batch_size):
            l = seq_len[i].item()
            # Align pose
            gt_pose_list.append(batch['pose'][i:i+1, :l])
            gen_pose_list.append(gen_pose[i:i+1, :l])
            # Align trans
            gt_trans_list.append(batch['trans'][i:i+1, :l])
            gen_trans_list.append(gen_trans[i:i+1, :l])

        # Move pose to CPU, keep trans on GPU 
        gt_pose_flat = torch.cat(gt_pose_list, dim=1).cpu()
        gen_pose_flat = torch.cat(gen_pose_list, dim=1).cpu()
        gt_trans_flat = torch.cat(gt_trans_list, dim=1)
        gen_trans_flat = torch.cat(gen_trans_list, dim=1)

        # Compute evaluation metrics
        val_mpjae_rad = self.evaluator.compute_mpjae(gt_pose_flat, gen_pose_flat)
        val_mpjae_deg = val_mpjae_rad * (180.0 / math.pi)
        val_trans_mse = F.mse_loss(gen_trans_flat, gt_trans_flat)
        
        self.log("val/mpjae_deg", val_mpjae_deg, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/trans_mse", val_trans_mse, on_step=False, on_epoch=True, sync_dist=True)
        return val_mpjae_deg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class JointBaselineModel(ConditionalBaselineModel):
    """Multimodal joint generator model."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lambda_motion = cfg['training'].get('lambda_motion', 1.0)
        self.lambda_label = cfg['training'].get('lambda_label', 1.0)
        self.num_classes = cfg['model'].get('num_classes', 4)
        
        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        
        self.backbone = JointBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.seq_len, self.backbone.num_joints, self.backbone.pose_dim)
        self.jump_head = JumpHead(hidden_dim, num_classes=self.num_classes)

    def forward(self, x_tau_dict, t, y_tau):
        shared_latent = self.backbone(x_tau_dict, t, y_tau)
        u_theta = self.flow_head(shared_latent) 
        Q_theta = self.jump_head(shared_latent) 
        return u_theta, Q_theta

    def _compute_ar_rollout_loss(self, batch):
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        y_target = batch['severity']
        batch_size = y_target.shape[0]

        M_cond = batch['cond_mask'] 
        M_targ = ~M_cond

        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
        tau = torch.rand(batch_size, 1, device=self.device)
        noise_std = self.cfg['training'].get('prefix_noise_std', 0.05)
        x_1_noisy = add_noise(x_1, noise_std)

        canvas_cond = mask(x_1_noisy, M_cond)
        canvas_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(canvas_cond, canvas_targ)

        u_true = mask(sub(x_1, x_0), M_targ)

        # Discrete CTMC Jump Mixture Path
        mask = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask, y_target, y_random)
        
        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / (M_targ.sum() * v_pred['pose'].shape[2] * v_pred['pose'].shape[3] + 1e-8)
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / (M_targ.sum() * v_pred['trans'].shape[2] + 1e-8)
        loss_label = F.cross_entropy(Q_pred, y_target)
        
        return loss_pose, loss_trans, loss_label

    def _compute_one_shot_loss(self, batch):
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        y_target = batch['severity']
        batch_size = y_target.shape[0]
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static

        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
        tau = torch.rand(batch_size, 1, device=self.device)

        canvas_static = mask(x_1, M_static)
        canvas_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(canvas_static, canvas_targ)

        u_true = mask(sub(x_1, x_0), M_targ)

        # Discrete CTMC Jump Mixture Path
        mask = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask, y_target, y_random)

        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / (M_targ.sum() * v_pred['pose'].shape[2] * v_pred['pose'].shape[3] + 1e-8)
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / (M_targ.sum() * v_pred['trans'].shape[2] + 1e-8)
        loss_label = F.cross_entropy(Q_pred, y_target)

        return loss_pose, loss_trans, loss_label

    def training_step(self, batch, batch_idx):
        gen_mode = self.cfg['model'].get('generation_mode', 'ar_rollout')
        
        if gen_mode == 'one_shot':
            loss_pose, loss_trans, loss_label = self._compute_one_shot_loss(batch)
        else:
            loss_pose, loss_trans, loss_label = self._compute_ar_rollout_loss(batch)

        loss_motion = (self.lambda_pose * loss_pose) + (self.lambda_trans * loss_trans)
        loss_total = (self.lambda_motion * loss_motion) + (self.lambda_label * loss_label)
        
        self.log("train/loss_motion", loss_motion, on_step=False, on_epoch=True)
        self.log("train/loss_label", loss_label, on_step=False, on_epoch=True)
        self.log("train/loss_total", loss_total, on_step=False, on_epoch=True)
        return loss_total

    def generate_suffix(self, x_tau, severity_score, M_targ, num_steps=100, y_0=None):
        batch_size = x_tau['pose'].shape[0]
        num_classes = self.cfg['model'].get('num_classes', 4)
        dt = 1.0 / num_steps

        if severity_score is not None and not self.training:
            y_tau = severity_score.clone()
            sample_labels = False
        else:
            y_tau = y_0.clone() if y_0 is not None else torch.randint(0, num_classes, (batch_size,), device=self.device)
            sample_labels = True

        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            
            u_theta, Q_theta = self(x_tau, tau_tensor, y_tau)
            v_pred_masked = mask(u_theta, M_targ)
            
            x_tau = add(x_tau, mul(v_pred_masked, torch.tensor([dt], device=self.device)))
            
            if sample_labels:
                y_tau = ctmc_jump_step(y_tau, Q_theta, dt, num_classes)

        return x_tau, y_tau
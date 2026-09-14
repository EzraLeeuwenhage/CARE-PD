import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.model_backbones import (
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
        mask_jump = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask_jump, y_target, y_random)
        
        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
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
        mask_jump = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask_jump, y_target, y_random)

        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8
        
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
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
            # Don't sample severity score, MGM-Cond 
            y_tau = severity_score.clone()
            sample_labels = False
        else:
            if y_0 is not None:
                # use provided initial severity score
                y_tau = y_0.clone() 
            else:
                # sample initial severity score randomly
                y_tau = torch.randint(0, num_classes, (batch_size,), device=self.device)
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
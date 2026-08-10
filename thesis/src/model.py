import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from thesis.src.generate_prior import generate_prior_from_prefix
from thesis.src.evaluate_smpl import SMPLEvaluator


# ====================
# COMPONENTS
# ====================
class SinusoidalEmbedding(nn.Module):
    """Standard Sinusoidal Positional/Time Embedding.
    
    Used for flow matching time, conditional severity scores, and current 
    discrete label states in jump processes.
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        
        # Pre-compute frequencies once during init
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size] or [batch_size, 1] containing scalars.
        returns: Tensor of shape [batch_size, dim]
        """
        x = x.view(-1).float()
        
        # Use the pre-computed frequencies
        args = x[:, None] * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class FlowHead(nn.Module):
    """Solves the conditional KFE for the continuous motion state space S_1."""
    def __init__(self, hidden_dim, target_frames, num_joints):
        super().__init__()
        self.target_frames = target_frames
        self.num_joints = num_joints
        
        # Dimensions: (frames * joints * 6 (pose)) + (frames * 3 (translations))
        self.target_dim = (self.target_frames * self.num_joints * 6) + (self.target_frames * 3)
        self.net = nn.Linear(hidden_dim, self.target_dim)

    def forward(self, shared_latent):
        u_pred_flat = self.net(shared_latent)
        batch_size = u_pred_flat.shape[0]
        
        # Unflatten output into pose and translation dict
        pose_size = self.target_frames * self.num_joints * 6
        u_pred_pose = u_pred_flat[:, :pose_size].reshape(batch_size, self.target_frames, self.num_joints, 6)
        u_pred_trans = u_pred_flat[:, pose_size:].reshape(batch_size, self.target_frames, 3)
        return {'pose': u_pred_pose, 'trans': u_pred_trans}


class JumpHead(nn.Module):
    """Solves the conditional KFE for the discrete categorical state space S_2.
    
    Returns: rate matrix Q_theta for the CTMC jump process.
    """
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        # Outputs jump rates to other categorical classes
        self.net = nn.Linear(hidden_dim, num_classes)

    def forward(self, shared_latent):
        return self.net(shared_latent)


# ====================
# BACKBONES
# ====================
def flatten_motion_inputs(x_tau_dict, prefix_dict):
    """Helper function to flatten pose and translation dicts."""
    batch_size = x_tau_dict['pose'].shape[0]
    
    x_t_pose_flat = x_tau_dict['pose'].reshape(batch_size, -1)
    x_t_trans_flat = x_tau_dict['trans'].reshape(batch_size, -1)
    x_t_flat = torch.cat([x_t_pose_flat, x_t_trans_flat], dim=1)

    prefix_pose_flat = prefix_dict['pose'].reshape(batch_size, -1)
    prefix_trans_flat = prefix_dict['trans'].reshape(batch_size, -1)
    prefix_flat = torch.cat([prefix_pose_flat, prefix_trans_flat], dim=1)
    
    return x_t_flat, prefix_flat


class ConditionalBaselineBackbone(nn.Module):
    """Model Backbone for FM conditioned on static severity score.
    
    Uses MLP for static conditional label + continuous state into shared latent.
    """
    def __init__(self, cfg, hidden_dim=1024, class_embed_dim=64, time_embed_dim=64):
        super().__init__()
        self.cfg = cfg
        self.target_frames = self.cfg['windowing']['total_window_size'] - self.cfg['windowing']['prefix_length']
        self.prefix_frames = self.cfg['windowing']['prefix_length']
        self.num_joints = self.cfg['data']['num_joints']

        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.class_embed = SinusoidalEmbedding(class_embed_dim)
        
        target_dim = (self.target_frames * self.num_joints * 6) + (self.target_frames * 3)
        prefix_dim = (self.prefix_frames * self.num_joints * 6) + (self.prefix_frames * 3)
        input_dim = target_dim + prefix_dim + class_embed_dim + time_embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x_tau_dict, prefix_dict, tau, severity_score):
        x_tau_flat, prefix_flat = flatten_motion_inputs(x_tau_dict, prefix_dict)
        t_emb = self.time_embed(tau)
        c_emb = self.class_embed(severity_score)
        
        nn_input = torch.cat([x_tau_flat, prefix_flat, c_emb, t_emb], dim=1)
        return self.net(nn_input)


class JointBaselineBackbone(ConditionalBaselineBackbone):
    """Adapts ConditionalBaselineBackbone to handle discrete categorical labels in addition to continuous motion.
    
    Processes noisy continuous motion (x_tau) and noisy discrete label (y_tau) into shared latent.
    """
    def forward(self, x_tau_dict, prefix_dict, tau, y_tau):
        # y_tau replaces severity_score, acting as the current state in the jump process.
        x_tau_flat, prefix_flat = flatten_motion_inputs(x_tau_dict, prefix_dict)
        t_emb = self.time_embed(tau)
        y_emb = self.class_embed(y_tau) 
        
        joint_input = torch.cat([x_tau_flat, prefix_flat, y_emb, t_emb], dim=1)
        return self.net(joint_input)


# ====================
# MODEL CLASSES
# ====================
class ConditionalBaselineModel(pl.LightningModule):
    """Baseline conditional generator model for comparison against better backbones and joint models.

    Wrapped in PyTorch Lightning for automated training loops and W&B logging.
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = cfg['training']['learning_rate']
        self.loss_weight = cfg['training'].get('loss_weight', 0.5)
        self.num_steps = cfg['sampling'].get('num_steps', 100)
        
        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        
        self.backbone = ConditionalBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.target_frames, self.backbone.num_joints)
        self.evaluator = SMPLEvaluator()

    def forward(self, x_tau_dict, prefix_dict, tau, severity_score):
        shared_latent = self.backbone(x_tau_dict, prefix_dict, tau, severity_score)
        return self.flow_head(shared_latent)

    def generate_suffix(self, prefix_dict, x_0_dict, severity_score, num_steps=100):
        """Euler ODE Solver for generating the target suffix (x_1) from the generated prior (x_0). 
        
        Starts from the generated prior (x_0) and iteratively applies the model's 
        predicted velocity field to generate the target suffix (x_1).
        """
        batch_size = prefix_dict['pose'].shape[0]
        x_tau = {'pose': x_0_dict['pose'].clone(), 'trans': x_0_dict['trans'].clone()}
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            velocity = self(x_tau, prefix_dict, tau_tensor, severity_score)
            
            x_tau['pose'] = x_tau['pose'] + (velocity['pose'] * dt)
            x_tau['trans'] = x_tau['trans'] + (velocity['trans'] * dt)
                
        return x_tau

    def training_step(self, batch, batch_idx):
        prefix_dict, target_dict, severity_score = batch
        batch_size = severity_score.shape[0]

        # Sample prior (x_0) and FM time (tau)
        x_0_dict = generate_prior_from_prefix(prefix_dict, target_dict)
        tau = torch.rand(batch_size, 1, device=self.device)

        # Linear interpolation (x_t)
        tau_pose = tau.view(batch_size, 1, 1, 1)
        tau_trans = tau.view(batch_size, 1, 1)

        x_tau_dict = {
            'pose': (1 - tau_pose) * x_0_dict['pose'] + tau_pose * target_dict['pose'],
            'trans': (1 - tau_trans) * x_0_dict['trans'] + tau_trans * target_dict['trans']
        }

        u_true_dict = {
            'pose': target_dict['pose'] - x_0_dict['pose'],
            'trans': target_dict['trans'] - x_0_dict['trans']
        }

        # Predict velocity field and compute loss
        u_pred_dict = self(x_tau_dict, prefix_dict, tau, severity_score)
        loss_pose = F.mse_loss(u_pred_dict['pose'], u_true_dict['pose'])
        loss_trans = F.mse_loss(u_pred_dict['trans'], u_true_dict['trans'])
        loss_total = ((1.0 - self.loss_weight) * loss_pose) + (self.loss_weight * loss_trans)
        
        self.log("train/loss_pose", loss_pose)
        self.log("train/loss_trans", loss_trans)
        self.log("train/loss_total", loss_total, prog_bar=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        """Automated validation step."""
        prefix_dict, target_dict, severity_score = batch

        x_0_dict = generate_prior_from_prefix(prefix_dict, target_dict)
        gen_suffix = self.generate_suffix(prefix_dict, x_0_dict, severity_score, num_steps=self.num_steps)

        gt_pose = torch.cat([prefix_dict['pose'], target_dict['pose']], dim=1).cpu()
        gen_pose = torch.cat([prefix_dict['pose'], gen_suffix['pose']], dim=1).cpu()

        val_mpjae = self.evaluator.compute_mpjae(gt_pose, gen_pose)
        
        self.log("val/mpjae_rad", val_mpjae, prog_bar=True, sync_dist=True)
        return val_mpjae

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class JointBaselineModel(ConditionalBaselineModel):
    """Multimodal joint generator model. 
    
    Approximates the marginal probability path for joint distribution through
    separate conditional probability paths for continuous motion and discrete label approximators.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lambda_motion = cfg['training'].get('lambda_motion', 1.0)
        self.lambda_label = cfg['training'].get('lambda_label', 1.0)
        
        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        num_classes = cfg['model'].get('num_classes', 4)
        
        self.backbone = JointBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.target_frames, self.backbone.num_joints)
        self.jump_head = JumpHead(hidden_dim, num_classes=num_classes)

    def forward(self, x_tau_dict, prefix_dict, t, y_tau):
        shared_latent = self.backbone(x_tau_dict, prefix_dict, t, y_tau)
        
        # Factorized conditional generators
        u_theta = self.flow_head(shared_latent)   # continuous vector field
        Q_theta = self.jump_head(shared_latent)   # dsiscrete rate matrix
        
        return u_theta, Q_theta

    def generate_suffix(self, prefix_dict, x_0_dict, severity_score, num_steps=100):
        """Overrides generate_suffix to unpack (u_theta, Q_theta) during joint inference."""
        batch_size = prefix_dict['pose'].shape[0]
        x_tau = {'pose': x_0_dict['pose'].clone(), 'trans': x_0_dict['trans'].clone()}
        y_tau = severity_score.clone()
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            u_theta, Q_theta = self(x_tau, prefix_dict, tau_tensor, y_tau)
            
            x_tau['pose'] = x_tau['pose'] + (u_theta['pose'] * dt)
            x_tau['trans'] = x_tau['trans'] + (u_theta['trans'] * dt)
                
        return x_tau

    def training_step(self, batch, batch_idx):
        prefix_dict, x_tau_dict, y_tau, u_target_dict, y_target, t = batch
        
        # Predict velocity field and rate matrix
        u_pred_dict, Q_pred = self(x_tau_dict, prefix_dict, t, y_tau)
        
        # Calculate Conditional FM Loss (Motion)
        loss_pose = F.mse_loss(u_pred_dict['pose'], u_target_dict['pose'])
        loss_trans = F.mse_loss(u_pred_dict['trans'], u_target_dict['trans'])
        loss_motion = loss_pose + loss_trans
        
        # Calculate Conditional GM Jump Loss
        # uses CrossEntropy as a Bregman divergence replacement for categorical target prediction (Holderrieth et al.)
        loss_label = F.cross_entropy(Q_pred, y_target)
        
        # Total joint training objective is the linear sum of the individual CGM losses
        loss_total = (self.lambda_motion * loss_motion) + (self.lambda_label * loss_label)
        
        self.log("train/loss_motion", loss_motion)
        self.log("train/loss_label", loss_label)
        self.log("train/loss_total", loss_total, prog_bar=True)
        return loss_total
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from thesis.src.generate_prior import generate_motion_prior_from_prefix


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
    def __init__(self, hidden_dim, seq_len, num_joints, pose_dim):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.pose_dim = pose_dim
        self.pose_size = seq_len * num_joints * pose_dim
        
        self.target_dim = self.pose_size + (seq_len * 3)
        self.net = nn.Linear(hidden_dim, self.target_dim)

    def forward(self, shared_latent):
        u_pred_flat = self.net(shared_latent)
        batch_size = u_pred_flat.shape[0]
        
        # Unflatten output into pose and translation dict matching the full sequence canvas
        u_pred_pose = u_pred_flat[:, :self.pose_size].reshape(batch_size, self.seq_len, self.num_joints, self.pose_dim)
        u_pred_trans = u_pred_flat[:, self.pose_size:].reshape(batch_size, self.seq_len, 3)
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
def flatten_motion_inputs(x_tau_dict):
    """Helper function to flatten pose and translation dicts."""
    batch_size = x_tau_dict['pose'].shape[0]
    x_t_pose_flat = x_tau_dict['pose'].reshape(batch_size, -1)
    x_t_trans_flat = x_tau_dict['trans'].reshape(batch_size, -1)
    
    return torch.cat([x_t_pose_flat, x_t_trans_flat], dim=1)


class ResMLPBlock(nn.Module):
    """Residual block with pre-LayerNorm and SiLU (Swish) activations."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)
    

class ConditionalBaselineBackbone(nn.Module):
    """Deep Residual MLP Backbone for Flow Matching with LayerNorm and SiLU activations."""
    def __init__(self, cfg, hidden_dim=1024, class_embed_dim=64, time_embed_dim=64):
        super().__init__()
        self.cfg = cfg
        gen_mode = self.cfg['model'].get('generation_mode', 'ar_rollout')
        
        if gen_mode == 'one_shot':
            self.seq_len = self.cfg['windowing'].get('max_sequence_len', 200)
        else:
            self.seq_len = self.cfg['windowing']['total_window_size']
            
        self.num_joints = self.cfg['data']['num_joints']
        representation = self.cfg['data'].get('representation', '6D')
        self.pose_dim = 3 if representation == '3D' else 6

        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.class_embed = SinusoidalEmbedding(class_embed_dim)

        pose_size = self.seq_len * self.num_joints * self.pose_dim
        motion_dim = pose_size + (self.seq_len * 3)
        input_dim = motion_dim + class_embed_dim + time_embed_dim
        
        # Linear projection into hidden space
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        
        # 3 Residual blocks providing 6 total non-linear transformations with identity skips
        self.res_blocks = nn.Sequential(
            ResMLPBlock(hidden_dim),
            ResMLPBlock(hidden_dim),
            ResMLPBlock(hidden_dim)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _forward_latent(self, motion_flat, cond_emb, tau):
        tau_emb = self.time_embed(tau)
        h = torch.cat([motion_flat, cond_emb, tau_emb], dim=1)
        h = self.in_proj(h)
        h = self.res_blocks(h)
        return self.out_norm(h)

    def forward(self, x_tau_dict, tau, severity_score):
        x_tau_flat = flatten_motion_inputs(x_tau_dict)
        c_emb = self.class_embed(severity_score)
        return self._forward_latent(x_tau_flat, c_emb, tau)


class JointBaselineBackbone(ConditionalBaselineBackbone):
    """Adapts ConditionalBaselineBackbone to handle discrete categorical labels in addition to continuous motion.
    
    Processes noisy continuous motion (x_tau) and noisy discrete label (y_tau) into shared latent.
    """
    def forward(self, x_tau_dict, tau, y_tau):
        # y_tau replaces severity_score, acting as the current state in the jump process.
        x_tau_flat = flatten_motion_inputs(x_tau_dict)
        y_tau_emb = self.class_embed(y_tau)
        return self._forward_latent(x_tau_flat, y_tau_emb, tau)

# ====================
# HELPER FUNCTIONS
# ====================
def mask(dict, mask):
    """Applies 1D boolean sequence mask [B, T] across all dimensions of Pose and Trans."""
    return {
        'pose': dict['pose'] * mask.view(dict['pose'].shape[0], -1, 1, 1).float(),
        'trans': dict['trans'] * mask.view(dict['trans'].shape[0], -1, 1).float()
    }
def add(dict1, dict2): return {'pose': dict1['pose'] + dict2['pose'], 'trans': dict1['trans'] + dict2['trans']}
def sub(dict1, dict2): return {'pose': dict1['pose'] - dict2['pose'], 'trans': dict1['trans'] - dict2['trans']}
def mul(dict, scalar):
    scalar_pose = scalar.view(-1, 1, 1, 1)
    scalar_trans = scalar.view(-1, 1, 1)
    return {'pose': dict['pose'] * scalar_pose, 'trans': dict['trans'] * scalar_trans}
def add_noise(dict, std):
    return {
        'pose': dict['pose'] + torch.randn_like(dict['pose']) * std,
        'trans': dict['trans'] + torch.randn_like(dict['trans']) * std
    }

def generate_x0(x1_dict, prefix_len, prior_noise_scale, generator=None, use_cumsum=False):
    """Helper to generate prior noise efficiently bounded by the valid generative horizon."""
    prefix_pose = x1_dict['pose'][:, :prefix_len]
    prefix_trans = x1_dict['trans'][:, :prefix_len]
    target_frames = x1_dict['pose'].shape[1] - prefix_len

    # Generate prior from prefix
    priors = generate_motion_prior_from_prefix(
        prefix_pose, prefix_trans, target_frames, 
        prior_noise_scale=prior_noise_scale, generator=generator, use_cumsum=use_cumsum
    )
    
    x0_pose = torch.cat([prefix_pose, priors['pose']], dim=1)
    x0_trans = torch.cat([prefix_trans, priors['trans']], dim=1)

    return {'pose': x0_pose, 'trans': x0_trans}
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


class ConditionalBaselineBackbone(nn.Module):
    """Model Backbone for FM conditioned on static severity score.
    
    Uses MLP for static conditional label + continuous state into shared latent.
    """
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
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
    def forward(self, x_tau_dict, tau, severity_score):
        x_tau_flat = flatten_motion_inputs(x_tau_dict)
        t_emb = self.time_embed(tau)
        c_emb = self.class_embed(severity_score)
        
        nn_input = torch.cat([x_tau_flat, c_emb, t_emb], dim=1)
        return self.net(nn_input)


class JointBaselineBackbone(ConditionalBaselineBackbone):
    """Adapts ConditionalBaselineBackbone to handle discrete categorical labels in addition to continuous motion.
    
    Processes noisy continuous motion (x_tau) and noisy discrete label (y_tau) into shared latent.
    """
    def forward(self, x_tau_dict, tau, y_tau):
        # y_tau replaces severity_score, acting as the current state in the jump process.
        x_tau_flat = flatten_motion_inputs(x_tau_dict)
        t_emb = self.time_embed(tau)
        y_emb = self.class_embed(y_tau) 
        
        joint_input = torch.cat([x_tau_flat, y_emb, t_emb], dim=1)
        return self.net(joint_input)

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

def generate_x0(x_1_dict, prefix_len, s_scale, generator=None):
    """Helper to generate prior noise efficiently bounded by the valid generative horizon."""
    prefix_pose = x_1_dict['pose'][:, :prefix_len]
    prefix_trans = x_1_dict['trans'][:, :prefix_len]
    num_frames_x0 = x_1_dict['pose'].shape[1] - prefix_len

    # Generate Brownian prior from prefix
    x_0 = generate_motion_prior_from_prefix(prefix_pose, prefix_trans, num_frames_x0, s_scale, generator=generator)
    
    x_0_pose = torch.zeros_like(x_1_dict['pose'])
    x_0_trans = torch.zeros_like(x_1_dict['trans'])
    x_0_pose[:, prefix_len:] = x_0['pose']
    x_0_trans[:, prefix_len:] = x_0['trans']

    return {'pose': x_0_pose, 'trans': x_0_trans}
import math
import yaml
import torch
import torch.nn as nn

CONFIG_PATH = "thesis/configs/dataloader.yaml"

class SinusoidalEmbedding(nn.Module):
    """Standard Sinusoidal Positional/Time Embedding.
    
    Used for both flow matching time and severity score inputs.
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        
        # Pre-compute frequencies once during init
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
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
    

class FlowMatchingMLP(nn.Module):
    def __init__(self, config_path=CONFIG_PATH, hidden_dim=1024, class_embed_dim=64, time_embed_dim=64):
        super().__init__()
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.target_frames = self.cfg['windowing']['total_window_size'] - self.cfg['windowing']['prefix_length']
        self.prefix_frames = self.cfg['windowing']['prefix_length']
        self.num_joints = self.cfg['data']['num_joints']

        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.class_embed = SinusoidalEmbedding(class_embed_dim)
        
        # dimensions: (frames * joints * 6D (pose)) + (frames * 3 (translations))
        self.target_dim = (self.target_frames * self.num_joints * 6) + (self.target_frames * 3)
        self.prefix_dim = (self.prefix_frames * self.num_joints * 6) + (self.prefix_frames * 3)
        
        # x_t + prefix + class_emb + t_emb
        input_dim = self.target_dim + self.prefix_dim + class_embed_dim + time_embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.target_dim)
        )
    
    def forward(self, x_t_dict, prefix_dict, t, severity_score):
        batch_size = t.shape[0]
        
        # Flatten vectors
        x_t_pose_flat = x_t_dict['pose'].reshape(batch_size, -1)
        x_t_trans_flat = x_t_dict['trans'].reshape(batch_size, -1)
        x_t_flat = torch.cat([x_t_pose_flat, x_t_trans_flat], dim=1)

        prefix_pose_flat = prefix_dict['pose'].reshape(batch_size, -1)
        prefix_trans_flat = prefix_dict['trans'].reshape(batch_size, -1)
        prefix_flat = torch.cat([prefix_pose_flat, prefix_trans_flat], dim=1)

        # Generate sinusoidal embeddings
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(severity_score)
        
        # Concat all
        nn_input = torch.cat([x_t_flat, prefix_flat, c_emb, t_emb], dim=1)
        
        # Forward pass and unflatten output
        u_pred_flat = self.net(nn_input)
        pose_size = self.target_frames * self.num_joints * 6
        
        u_pred_pose = u_pred_flat[:, :pose_size].reshape(batch_size, self.target_frames, self.num_joints, 6)
        u_pred_trans = u_pred_flat[:, pose_size:].reshape(batch_size, self.target_frames, 3)
        
        return {
            'pose': u_pred_pose,
            'trans': u_pred_trans
        }
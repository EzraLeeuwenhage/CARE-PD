import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.src.model_backbones import SinusoidalEmbedding


class SpatialTemporalBlock(nn.Module):
    """Alternates Spatial MHSA (Joint Mixing) with Temporal Causal Dilated Conv1D."""
    def __init__(self, dim, num_heads=8, dilation=1):
        super().__init__()
        self.dilation = dilation
        
        # Spatial Attention across J=25 tokens
        self.norm_spatial = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        
        # Temporal Causal Conv across T=60 frames
        self.norm_temp = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)
        
        # Unified Feed-Forward Network
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim)
        )
        
        # Single modulation projection predicting scale/shift for the 3 stages
        self.mod_proj = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.mod_proj.weight)
        nn.init.zeros_(self.mod_proj.bias)

    def forward(self, x, w):
        # x: (B, T, J, D), w: (B, D)
        B, T, J, D = x.shape
        
        # Project conditioning vector once per block
        mods = self.mod_proj(w).view(B, 1, 1, 6, D)
        s_attn, b_attn, s_conv, b_conv, s_ffn, b_ffn = mods.unbind(dim=3)

        # Spatial MHSA (Joints)
        h_spatial = self.norm_spatial(x) * (1.0 + s_attn) + b_attn
        h_spatial = h_spatial.reshape(B * T, J, D)
        attn_out, _ = self.attn(h_spatial, h_spatial, h_spatial, need_weights=False)
        x = x + attn_out.reshape(B, T, J, D)

        # TCN (Time)
        h_temp = self.norm_temp(x) * (1.0 + s_conv) + b_conv
        # Reshape to (B * J, D, T) for 1D convolution
        conv_in = h_temp.permute(0, 2, 3, 1).reshape(B * J, D, T)
        conv_in = F.pad(conv_in, (2 * self.dilation, 0))  # Causal padding
        conv_out = F.silu(self.conv(conv_in))
        x = x + conv_out.reshape(B, J, D, T).permute(0, 3, 1, 2)

        # Feed-Forward
        h_ffn = self.norm_ffn(x) * (1.0 + s_ffn) + b_ffn
        x = x + self.ffn(h_ffn)
        
        return x


class SpatioTemporalBackbone(nn.Module):
    """Efficient SpatioTemporal Backbone for Motion Flow Matching."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg['model'].get('hidden_dim', 256)
        num_heads = cfg['model'].get('num_heads', 8)
        dilations = cfg['model'].get('dilations', [1, 2, 4, 8])

        self.num_joints = cfg['data'].get('num_joints', 24)
        representation = cfg['data'].get('representation', '3D')
        self.pose_dim = 3 if representation == '3D' else 6
        self.total_tokens = self.num_joints + 1  # 24 Pose + 1 Translation

        # Token Projections & Modality Embeddings
        self.pose_in = nn.Linear(self.pose_dim, self.dim)
        self.trans_in = nn.Linear(3, self.dim)

        self.pose_modality = nn.Parameter(torch.randn(1, 1, 1, self.dim) * 0.02)
        self.trans_modality = nn.Parameter(torch.randn(1, 1, 1, self.dim) * 0.02)
        self.joint_pos_emb = nn.Parameter(torch.randn(1, 1, self.total_tokens, self.dim) * 0.02)

        # Conditioning
        time_dim = cfg['model'].get('time_embed_dim', 64)
        class_dim = cfg['model'].get('class_embed_dim', 64)
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.class_embed = SinusoidalEmbedding(class_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + class_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )

        # Blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(self.dim, num_heads=num_heads, dilation=d)
            for d in dilations
        ])

        # Output Projections (zero-initialized for stable flow starts)
        self.out_pose = nn.Linear(self.dim, self.pose_dim)
        self.out_trans = nn.Linear(self.dim, 3)
        nn.init.zeros_(self.out_pose.weight)
        nn.init.zeros_(self.out_pose.bias)
        nn.init.zeros_(self.out_trans.weight)
        nn.init.zeros_(self.out_trans.bias)

    def forward(self, x_tau_dict, tau, severity_score):
        pose = x_tau_dict['pose']    # (B, T, 24, pose_dim)
        trans = x_tau_dict['trans']  # (B, T, 3)

        # Tokenize
        h_pose = self.pose_in(pose) + self.pose_modality
        h_trans = self.trans_in(trans).unsqueeze(2) + self.trans_modality
        h = torch.cat([h_pose, h_trans], dim=2) + self.joint_pos_emb

        # Condition vector w: (B, D)
        c_t = self.time_embed(tau)
        c_s = self.class_embed(severity_score)
        w = self.cond_mlp(torch.cat([c_t, c_s], dim=-1))

        # Pass through streamlined blocks
        for block in self.blocks:
            h = block(h, w)

        # Heads
        out_pose = self.out_pose(h[:, :, :self.num_joints, :])
        out_trans = self.out_trans(h[:, :, self.num_joints, :])

        return {'pose': out_pose, 'trans': out_trans}
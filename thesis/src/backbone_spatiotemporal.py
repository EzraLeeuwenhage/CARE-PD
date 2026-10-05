import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.src.model_backbones import SinusoidalEmbedding


class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on vector w."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc = nn.Linear(dim, 2 * dim)
        # Zero-init so training begins as standard identity LayerNorm
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, w):
        # x: (N, L, dim), w: (N, dim)
        scale, shift = self.fc(w).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class SpatialTemporalBlock(nn.Module):
    """Alternates Spatial MHSA (Joint Mixing) with Temporal Causal Dilated Conv1D."""
    def __init__(self, dim, num_heads=8, dilation=1):
        super().__init__()
        self.dilation = dilation

        # Spatial MHSA across J=25 tokens
        self.adaln1 = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.adaln2 = AdaLN(dim)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim)
        )

        # Temporal Causal TCN across T=60 frames
        self.adaln3 = AdaLN(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)
        self.act = nn.SiLU()
        self.adaln4 = AdaLN(dim)
        self.temp_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(self, x, w):
        B, T, J, D = x.shape

        # ---------------------------------------------------------------------
        # 1. Spatial Attention across Joints (B*T, J, D)
        # ---------------------------------------------------------------------
        w_spatial = w.unsqueeze(1).expand(-1, T, -1).reshape(B * T, D)
        x_spatial = x.reshape(B * T, J, D)

        norm_x1 = self.adaln1(x_spatial, w_spatial)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1, need_weights=False)
        x_spatial = x_spatial + attn_out

        norm_x2 = self.adaln2(x_spatial, w_spatial)
        x_spatial = x_spatial + self.spatial_mlp(norm_x2)
        x = x_spatial.reshape(B, T, J, D)

        # ---------------------------------------------------------------------
        # 2. Temporal Causal Conv across Frames (B*J, T, D)
        # ---------------------------------------------------------------------
        w_temp = w.unsqueeze(1).expand(-1, J, -1).reshape(B * J, D)
        x_temp = x.permute(0, 2, 1, 3).reshape(B * J, T, D)

        norm_x3 = self.adaln3(x_temp, w_temp)
        conv_in = norm_x3.transpose(1, 2)  # (B*J, D, T)
        # Causal padding on left only: (pad_left, pad_right)
        conv_in = F.pad(conv_in, (2 * self.dilation, 0))
        conv_out = self.conv(conv_in).transpose(1, 2)
        x_temp = x_temp + self.act(conv_out)

        norm_x4 = self.adaln4(x_temp, w_temp)
        x_temp = x_temp + self.temp_mlp(norm_x4)
        x = x_temp.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()

        return x


class SpatioTemporalBackbone(nn.Module):
    """Spatial-Transformer / Temporal-TCN Flow Matching Backbone."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg['model'].get('hidden_dim', 256)
        num_heads = cfg['model'].get('num_heads', 8)
        dilations = cfg['model'].get('dilations', [1, 2, 4, 8, 16])

        self.num_joints = cfg['data'].get('num_joints', 24)
        representation = cfg['data'].get('representation', '3D')
        self.pose_dim = 3 if representation == '3D' else 6
        self.total_tokens = self.num_joints + 1  # 24 Pose + 1 Root Translation

        # 1. Token Projections & Modality Embeddings
        self.pose_in = nn.Linear(self.pose_dim, self.dim)
        self.trans_in = nn.Linear(3, self.dim)

        self.pose_modality = nn.Parameter(torch.randn(1, 1, 1, self.dim) * 0.02)
        self.trans_modality = nn.Parameter(torch.randn(1, 1, 1, self.dim) * 0.02)
        self.joint_pos_emb = nn.Parameter(torch.randn(1, 1, self.total_tokens, self.dim) * 0.02)

        # 2. Conditioning (AdaLN Hub)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.class_embed = SinusoidalEmbedding(class_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_embed_dim + class_embed_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )

        # 3. Stacked Spatiotemporal Blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(self.dim, num_heads=num_heads, dilation=d)
            for d in dilations
        ])

        # 4. Output Projections (zero-initialized for stable Flow Matching starts)
        self.out_pose = nn.Linear(self.dim, self.pose_dim)
        self.out_trans = nn.Linear(self.dim, 3)
        nn.init.zeros_(self.out_pose.weight)
        nn.init.zeros_(self.out_pose.bias)
        nn.init.zeros_(self.out_trans.weight)
        nn.init.zeros_(self.out_trans.bias)

    def forward(self, x_tau_dict, tau, severity_score):
        pose = x_tau_dict['pose']    # (B, T, 24, pose_dim)
        trans = x_tau_dict['trans']  # (B, T, 3)

        # Tokenize and add modality embeddings
        h_pose = self.pose_in(pose) + self.pose_modality
        h_trans = self.trans_in(trans).unsqueeze(2) + self.trans_modality
        h = torch.cat([h_pose, h_trans], dim=2) + self.joint_pos_emb

        # Condition vector w
        c_t = self.time_embed(tau)
        c_s = self.class_embed(severity_score)
        w = self.cond_mlp(torch.cat([c_t, c_s], dim=-1))

        # Alternating blocks
        for block in self.blocks:
            h = block(h, w)

        # Predict vector fields
        out_pose = self.out_pose(h[:, :, :self.num_joints, :])
        out_trans = self.out_trans(h[:, :, self.num_joints, :])

        return {'pose': out_pose, 'trans': out_trans}
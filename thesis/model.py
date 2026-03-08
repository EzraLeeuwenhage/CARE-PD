import torch
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv

class SinusoidalTimeEmbedding(nn.Module):
    """
    Step 1: Converts a scalar time 't' into a high-dimensional feature vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time shape: (Batch,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        
        # Combine sin and cos for the final embedding
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings # Shape: (Batch, dim)

class STGCNBlock(nn.Module):
    """
    Step 2 & 3: One block of Spatial (GCN) + Temporal (CNN) processing.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # Spatial Graph Convolution
        self.gcn = GCNConv(in_channels, out_channels)
        
        # Temporal Convolution (1D Conv over the time axis)
        # Kernel size 3 looks at 1 frame back, current frame, 1 frame forward.
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        
        # Projects the time embedding to match the channel dimension
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, t_emb):
        # x shape: (Batch, Channels, Time, Vertices) -> (N, 3, 60, 17)
        N, C, T, V = x.shape
        
        # --- SPATIAL PASS (GCN) ---
        # PyTorch Geometric expects (Num_Nodes, Features). 
        # We reshape x to fold Batch and Time into the Node dimension.
        x_reshaped = x.permute(0, 2, 3, 1).reshape(N * T, V, C) 
        
        # Apply GCN (this happens independently for every frame in the batch)
        x_gcn = self.gcn(x_reshaped, edge_index) # Output: (N*T, V, out_channels)
        
        # Reshape back to (N, out_channels, T, V)
        x_gcn = x_gcn.reshape(N, T, V, -1).permute(0, 3, 1, 2)
        
        # --- TIME INJECTION ---
        # Project time embedding and add it to the feature maps
        time_features = self.time_mlp(t_emb) # (N, out_channels)
        time_features = time_features.unsqueeze(2).unsqueeze(3) # (N, out_channels, 1, 1)
        x_gcn = x_gcn + time_features 
        
        x_gcn = self.relu(x_gcn)

        # --- TEMPORAL PASS (TCN) ---
        # Conv2d over (Time, Vertices). Kernel (3,1) slides over Time, keeping Vertices separate.
        x_tcn = self.tcn(x_gcn)
        x_tcn = self.relu(x_tcn)
        
        return x_tcn

class FlowMatchingVectorField(nn.Module):
    """
    The main model: Takes noisy data and time, predicts velocity.
    """
    def __init__(self, in_channels=3, hidden_dim=64, time_emb_dim=128):
        super().__init__()
        self.time_embedder = SinusoidalTimeEmbedding(time_emb_dim)
        
        # Lift the 3D coordinates into a higher dimensional feature space
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Stack two ST-GCN blocks
        self.block1 = STGCNBlock(hidden_dim, hidden_dim, time_emb_dim)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim, time_emb_dim)
        
        # Project back down to 3D coordinates (X, Y, Z velocity)
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x_t, t, edge_index):
        # x_t shape: (N, 3, 60, 17)
        # t shape: (N,)
        
        # 1. Embed the time
        t_emb = self.time_embedder(t)
        
        # 2. Initial feature projection
        h = self.input_proj(x_t)
        
        # 3. Spatio-Temporal processing
        h = self.block1(h, edge_index, t_emb)
        h = self.block2(h, edge_index, t_emb)
        
        # 4. Output the velocity vector field
        v_pred = self.output_proj(h) # Shape: (N, 3, 60, 17)
        
        return v_pred

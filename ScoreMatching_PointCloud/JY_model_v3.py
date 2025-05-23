import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device='cpu')
        elif t.dim() == 0:
            t = t[None]

        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class ScoreNet(nn.Module):
    def __init__(self, time_emb_dim=256, point_feat_dim=256, hidden_dim=512):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # input = (px, py, pz)
        self.point_encoder = nn.Linear(2, hidden_dim)

        self.point_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        """
        x: (N, 3)  # point cloud
        t: scalar or (B,)
        """

        N = x.shape[0]
        device = x.device

        # Time embedding
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t[None]

        t_emb = self.time_mlp(t)  # shape (B, hidden_dim)
        t_emb = t_emb.repeat_interleave(N // t.shape[0], dim=0)  # match shape: (N, hidden_dim)

        # Point MLP + residual
        x_enc = self.point_encoder(x)
        h = self.point_mlp(x_enc) + x_enc  # residual
        h = h + t_emb  # add time embedding

        score = self.score_head(h)  # (N, 3)
        return score

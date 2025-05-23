import torch
import torch.nn as nn
import math

# get_timestep_embedding: embedding time t to high-dimensional space
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: shape (B,) or scalar
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device='cpu')
        elif t.dim() == 0:
            t = t[None]

        half_dim = self.dim // 2 # "//" return int value, "/" return float value
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb  # shape (B, dim)


class ScoreNet(nn.Module):
    def __init__(self, time_emb_dim=64, point_feat_dim=64, hidden_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
        )

        self.point_mlp = nn.Sequential(
            nn.Linear(3, point_feat_dim),
            nn.ReLU(),
            nn.Linear(point_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, t):
        # x: (N, 3), t: scalar or (1,) tensor or (B,) broadcastable
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=x.device)
        if t.dim() == 0:
            t = t[None]

        t_emb = self.time_mlp(t)  # (1, hidden_dim) or (B, hidden_dim)
        t_emb = t_emb.expand(x.shape[0], -1)  # (N, hidden_dim)

        x_feat = self.point_mlp(x)  # (N, hidden_dim)
        h = x_feat + t_emb  # (N, hidden_dim)

        score = self.score_head(h)  # (N, 3)
        return score

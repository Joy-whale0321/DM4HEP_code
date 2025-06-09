# score_model.py
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
        emb = t[:, None] * emb[None, :]  # shape: [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # shape: [B, dim]

class ScoreNet(nn.Module):
    def __init__(self, time_emb_dim=64, point_feat_dim=128, hidden_dim=256):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        self.point_mlp = nn.Sequential(
            nn.Linear(2, point_feat_dim),
            nn.ReLU(),
            nn.Linear(point_feat_dim, point_feat_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(point_feat_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出 score vector: [dx, dy]
        )

    def forward(self, x, t):
        # x: [N, 2], t: [1] or [N]
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        time_emb = self.time_mlp(t.squeeze()).expand(x.size(0), -1)
        x_feat = self.point_mlp(x)
        xt = torch.cat([x_feat, time_emb], dim=-1)
        return self.net(xt)  # shape: [N, 2]

import torch
import torch.nn as nn
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(np.log(10000.0) / half_dim))
        emb = t * emb
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ScoreNetParticle(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        self.x_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # 新增一层
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim + time_emb_dim)

    def forward(self, x, t):
        x_embed = self.x_mlp(x)
        t_embed = self.time_mlp(t)
        xt = torch.cat([x_embed, t_embed], dim=-1)
        xt = self.norm(xt)
        return self.net(xt)

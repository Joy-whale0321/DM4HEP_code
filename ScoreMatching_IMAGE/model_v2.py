# === model_v2.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(1, embed_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        t = t.view(-1, 1)
        emb = self.linear1(t)
        emb = self.act(emb)
        return self.linear2(emb)

class ScoreNetImage(nn.Module):
    def __init__(self, img_channels=1, out_channels=1, time_emb_dim=128):
        super(ScoreNetImage, self).__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Conv2d(64 + time_emb_dim, 64, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        x_feat = self.encoder(x)
        t_embed = self.time_mlp(t).view(x.size(0), -1, 1, 1)
        t_embed = t_embed.expand(-1, -1, x_feat.shape[2], x_feat.shape[3])
        x_cat = torch.cat([x_feat, t_embed], dim=1)
        x_cat = self.middle(x_cat)
        return self.decoder(x_cat)
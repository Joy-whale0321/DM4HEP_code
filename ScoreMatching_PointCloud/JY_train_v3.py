import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from JY_model_v4 import ScoreNet
from JY_data import ParticlesPointDataset, my_collate_fn
import numpy as np
import os

# ---- HYPERPARAMETERS ----
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0

# ---- define linear noise schedule ----
def get_sigma_schedule(num_steps):
    return torch.linspace(SIGMA_MAX, SIGMA_MIN, num_steps + 1)

# ---- reshape function ----
def reshape_with_axis_tags(x):
    N = x.shape[0]
    device = x.device

    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    values = torch.cat([px, py, pz], dim=0).unsqueeze(1)  # (3N, 1)

    tag_px = torch.tensor([1, 0, 0], device=device).repeat(N, 1)
    tag_py = torch.tensor([0, 1, 0], device=device).repeat(N, 1)
    tag_pz = torch.tensor([0, 0, 1], device=device).repeat(N, 1)
    tags = torch.cat([tag_px, tag_py, tag_pz], dim=0)

    return torch.cat([values, tags], dim=1)  # (3N, 4)

# ---- load data ----
train_dataset = ParticlesPointDataset("/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)

# ---- initialize model and optimizer ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ScoreNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

sigma_sched = get_sigma_schedule(100).to(device)

# ---- training loop ----
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_loss_px = 0.0
    total_loss_py = 0.0
    total_loss_pz = 0.0
    total_events = 0

    for batch in train_loader:
        loss_batch = 0.0
        optimizer.zero_grad()

        for x in batch:
            x = x.to(device)  # (N, 3)
            N = x.shape[0]
            num_steps = len(sigma_sched) - 1
            t = torch.rand(1, device=device)
            t_idx = int((1 - t.item()) * num_steps)
            sigma = sigma_sched[t_idx].view(1, 1)

            eps = torch.randn_like(x)
            x_t = x + sigma * eps

            x_tagged = reshape_with_axis_tags(x_t)  # (3N, 4)
            score_pred_flat = model(x_tagged, t)    # (3N,)
            score_pred = score_pred_flat.view(3, -1).T  # (N, 3)

            target = -eps / sigma
            diff = (score_pred - target) * sigma

            loss_px = (diff[:, 0] ** 2).mean()
            loss_py = (diff[:, 1] ** 2).mean()
            loss_pz = (diff[:, 2] ** 2).mean()
            loss = loss_px + loss_py + loss_pz

            loss.backward()
            loss_batch += loss.item()
            total_loss_px += loss_px.item()
            total_loss_py += loss_py.item()
            total_loss_pz += loss_pz.item()
            total_events += 1

        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_loss += loss_batch

    avg_loss = total_loss / total_events
    print(f"[Epoch {epoch}] Loss: {total_loss:.4f}, px: {total_loss_px:.4f}, py: {total_loss_py:.4f}, pz: {total_loss_pz:.4f}")

    if epoch % 10 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/scorenet_epoch{epoch}.pt")

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from data_v1 import ParticleDataset
from model_v1 import ScoreNetParticle

# ---- HYPERPARAMETERS ----
EPOCHS = 100
BATCH_SIZE = 256
LR = 5e-4
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0

def get_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load Data ----
dataset = ParticleDataset("/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root", 
                          tree_name="T_Final_Hadron_Dis", 
                          max_events=10)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model & Optimizer ----
model = ScoreNetParticle(input_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Training ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch.to(device)
        x = x.squeeze(1)
        t = torch.rand(x.shape[0], 1, device=device)
        sigma = get_sigma(t)
        noise = torch.randn_like(x)
        x_t = x + sigma * noise
        score_pred = model(x_t, t)
        target = -noise / sigma
        loss = ((score_pred - target) ** 2 * sigma ** 2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    if epoch % 10 == 0:
        os.makedirs("checkpoints_v2", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints_v2/scorenet_epoch{epoch}.pt")

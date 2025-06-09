# train_dsm.py
import torch
from torch.utils.data import DataLoader
from particles_dataset import ParticlesPointDataset
from score_model import ScoreNet
import numpy as np
import os

# ==== CONFIG ====
EPOCHS = 100
BATCH_SIZE = 512
SIGMA_MIN = 0.1
SIGMA_MAX = 0.6
LEARNING_RATE = 1e-4

ROOT_PATH = "/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_thr0p3_1k.root"
TREE_NAME = "T_Final_Hadron_Dis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==== NOISE SCHEDULE ====
# def get_sigma(t):
#     return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

def get_sigma(t):
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)

# ==== DATASET ====
dataset = ParticlesPointDataset(ROOT_PATH, treename=TREE_NAME)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== MODEL ====
model = ScoreNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Path to save
os.makedirs("checkpoints", exist_ok=True)

# ==== TRAIN ====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x in dataloader:  # x: [Batch_size, 2]
        x = x.to(DEVICE)
        B = x.size(0)
        # print("Batch size B:", B) # check batch size
 
        t = torch.rand(B, device=DEVICE).view(-1, 1)   # .view(-1,1): [B] -> [B, 1]
        sigma = get_sigma(t).to(DEVICE)

        eps = torch.randn_like(x)
        x_t = x + sigma * eps
        target = -eps / sigma

        pred = model(x_t, t)  # [B, 2]
        loss = (((pred - target) ** 2) * (sigma ** 2)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        # total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    # avg_loss = total_loss
    print("dataset len is:", len(dataset)) # check batch size
    print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/score_model_epoch{epoch}.pth")

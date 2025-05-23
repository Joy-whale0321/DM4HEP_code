import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from JY_model_v2 import ScoreNet
from JY_data import ParticlesPointDataset, my_collate_fn
import numpy as np
import os

# ---- HYPERPARAMETERS ----
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.0001 # step size
SIGMA_MIN = 0.2
SIGMA_MAX = 1.0

# ---- define noise schedule ----
def get_sigma(t):
    # exponential schedule from SIGMA_MIN to SIGMA_MAX, SIGMA_MIN at t=0, SIGMA_MAX at t=1. torch.tensor(SIGMA_MIN, device=t.device) used to let SIGMA_MIN to a tensor on the same device as t
    return torch.tensor(SIGMA_MIN, device=t.device) * ((SIGMA_MAX / SIGMA_MIN) ** t)

# ---- load data ----
train_dataset = ParticlesPointDataset("/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)

# ---- initialize model and optimizer ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device setting gpu(cuda)/cpu
model = ScoreNet().to(device) # initialize model, create the ScoreNet on device
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- training loop ----
# loop over epochs
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    
    # loop over batches
    for batch in train_loader:
        # batch: list of (N_i, 3) tensors
        loss_batch = 0.0
        optimizer.zero_grad()

        # loop over each event in the batch
        for i, x in enumerate(batch):
            x = x.to(device)  # (N, 3)

            # === 只检查第一个 event 的内容 ===
            if epoch == 1 and i == 0:
                print("\n=== DEBUG OUTPUT (Epoch 1, First Event) ===")
                print("x.shape:", x.shape)
                print("x[:5]:", x[:5].cpu().numpy())
                print("x min/max:", x.min().item(), "/", x.max().item())

            # 时间和 sigma
            t = torch.rand(1, device=device)
            sigma = get_sigma(t)

            eps = torch.randn_like(x)
            x_t = x + sigma.view(1, 1) * eps
            target = -eps / sigma.view(1, 1)

            if epoch == 1 and i == 0:
                print("sigma:", sigma.item())
                print("x_t[:5]:", x_t[:5].cpu().detach().numpy())
                print("target std:", target.std().item())
                print("target[:5]:", target[:5].cpu().detach().numpy())

            score_pred = model(x_t, t)
            loss = ((score_pred - target) ** 2).sum(dim=1).mean()

            # === 监控 loss 数值 ===
            if epoch == 1 and i == 0:
                print("loss:", loss.item())

            loss.backward()

            # === 检查是否真的有梯度流到模型 ===
            if epoch == 1 and i == 0:
                first_grad = model.score_head[0].weight.grad
                if first_grad is not None:
                    print("grad mean (score_head[0]):", first_grad.abs().mean().item())
                else:
                    print("grad is None!")

            loss_batch += loss.item()

        optimizer.step()
        total_loss += loss_batch

    print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

    # save checkpoint
    if epoch % 10 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/scorenet_epoch{epoch}.pt")
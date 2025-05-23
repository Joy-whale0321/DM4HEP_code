import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from JY_model_v4 import ScoreNet
from JY_data_v2 import ParticlesPointDataset, my_collate_fn
import numpy as np
import os

# ---- HYPERPARAMETERS ----
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # step size
SIGMA_MIN = 0.5
SIGMA_MAX = 1.0

# 在训练脚本开头
NUM_BINS = 10
# losses_per_bin[i] 会保存属于第 i 个 t-bin 的所有样本的 loss
losses_per_bin = [[] for _ in range(NUM_BINS)]

# ---- define noise schedule ----
def get_sigma(t):
    # exponential schedule from SIGMA_MIN to , SIGMA_MIN at t=0, SIGMA_MAX at t=1. 
    return torch.tensor(SIGMA_MIN, device=t.device) * ((SIGMA_MAX / SIGMA_MIN) ** t)
    # torch.tensor(SIGMA_MIN, device=t.device) used to let SIGMA_MIN to a tensor on the same device as t

# ---- define linear noise schedule ----
def get_sigma_schedule(num_steps):
    return torch.linspace(SIGMA_MAX, SIGMA_MIN, num_steps + 1)

# ---- load data ----
train_dataset = ParticlesPointDataset("/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)

# ---- initialize model and optimizer ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device setting gpu(cuda)/cpu
print("Using device:", device)

model = ScoreNet().to(device) # initialize model, create the ScoreNet on device
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

sigma_sched = get_sigma_schedule(100).to(device)  # 100 可以改大，但训练用随机即可

# ---- training loop ----
# loop over epochs
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_loss_px = 0.0
    total_loss_py = 0.0
    total_loss_pz = 0.0
    total_events = 0

    # loop over batches
    for batch in train_loader:
        # batch: list of (N_i, 3) tensors
        loss_batch = 0.0
        optimizer.zero_grad()

        # loop over each event in the batch
        for x in batch:
            x = x.to(device)  # (N, 3)
            
            # noise create
            # t = torch.rand(1, device=device)  # scalar t ~ U(0, 1)
            # alpha = 2.0  # >1 时更偏向大 t
            # t = torch.rand(1, device=device)**alpha   # shape=(1,)
            # sigma = get_sigma(t).view(1, 1)   

            num_steps = len(sigma_sched) - 1  # == 100
            # t = torch.rand(1, device=device)
            alpha = 2.0 
            t = torch.rand(1, device=device)**alpha
            t_idx = int((1 - t.item()) * num_steps)
            sigma = sigma_sched[t_idx].view(1, 1)  # shape match

            eps = torch.randn_like(x)  # (N, 3) random normal like
            x_t = x + sigma.view(1, 1) * eps  # noised input

            score_pred = model(x_t, t)  # predict score
            target = -eps / sigma  # target score

            # with torch.no_grad():
                # print(f"score_pred mean: {score_pred.mean().item():.4f}, std: {score_pred.std().item():.4f}")
                # print(f"target     mean: {target.mean().item():.4f}, std: {target.std().item():.4f}")

            # loss = ((score_pred - target) ** 2).sum(dim=1).mean()  # loss function: DSM loss
            # loss = ((score_pred - target) ** 2 * sigma.view(1, 1) ** 2).sum(dim=1).mean()
            loss = ((score_pred - target) ** 2).sum(dim=1)
            loss = (loss * sigma.view(1, 1).detach() ** 1.5).mean()

            diff = (score_pred - target) * sigma.view(1, 1)

            loss_px = (diff[:, 0] ** 2).mean()
            loss_py = (diff[:, 1] ** 2).mean()
            # loss_pz = (diff[:, 2] ** 2).mean()

            # loss = loss_px + loss_py

            # 找到 t 落在哪个 bin
            bin_idx = min(int(t * NUM_BINS), NUM_BINS-1)
            losses_per_bin[bin_idx].append(loss)

            loss.backward()
            loss_batch += loss.item()
            total_loss_px += loss_px.item()
            total_loss_py += loss_py.item()
            # total_loss_pz += loss_pz.item()
            total_events += 1

            # print(f"[debug] loss: {loss.item():.4f}, sigma: {sigma.mean().item():.3f}, target.std: {target.std().item():.3f}, score_pred.std: {score_pred.std().item():.3f}")


        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # print(f"[Epoch {epoch}] Batch Loss (avg over events in batch): {loss_batch:.4f}")
        total_loss += loss_batch

    avg_loss = total_loss / total_events
    # print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")
    print(f"[Epoch {epoch}] Loss: {total_loss:.4f}, px: {total_loss_px:.4f}, py: {total_loss_py:.4f}")

    # avg_losses = [sum(bin_list)/len(bin_list) if bin_list else 0.0
    #               for bin_list in losses_per_bin]
    # for i, v in enumerate(avg_losses):
    #     print(f" Epoch {epoch:2d} — t ∈ [{i/NUM_BINS:.2f}, {(i+1)/NUM_BINS:.2f})  avg loss = {v:.4f}")
    # # 清空，为下个 epoch 重新统计
    # losses_per_bin = [[] for _ in range(NUM_BINS)]

    # save checkpoint
    if epoch % 10 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/scorenet_epoch{epoch}.pt")

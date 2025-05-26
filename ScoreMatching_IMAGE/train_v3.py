# === train_v3.py ===
import torch
from torch.utils.data import DataLoader
from data_v3 import PxPyVectorDataset
from model_v3 import ScoreNetImage
import torch.nn.functional as F
import torch.optim as optim
import os

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
SIGMA_MIN = 0.01
SIGMA_MAX = 20.0

os.makedirs("scoreNET", exist_ok=True)

def get_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PxPyVectorDataset(
    "/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root",
    img_size=(100, 100), px_range=(-4, 4), py_range=(-4, 4)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

model = ScoreNetImage().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for img_batch, _, _ in loader:
        img = torch.stack(img_batch).to(device)  # (B, 2, H, W)
        B, _, H, W = img.shape
        t = torch.rand(B, device=device)
        sigma = get_sigma(t).view(B, 1, 1, 1)

        noise = sigma * torch.randn((B, 2, H, W), device=device)
        noisy = img + noise
        score_target = -noise / sigma

        pred = model(noisy, t.unsqueeze(1))
        loss = F.mse_loss(pred, score_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"scoreNET/score_model_epoch{epoch+1}.pth")
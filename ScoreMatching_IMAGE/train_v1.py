# === train_v1.py ===
import torch
from torch.utils.data import DataLoader
from data_v1 import PxPyImageDataset
from model_v1 import ScoreNetImage
import torch.nn.functional as F
import torch.optim as optim

# ---- HYPERPARAMETERS ----
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0

def get_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PxPyImageDataset(
    "/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root",
    img_size=(500, 1000), px_range=(-5, 5), py_range=(-5, 5)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

model = ScoreNetImage(time_emb_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- TRAIN LOOP ----
from pathlib import Path
import numpy as np
out_path = Path("score_vector_output.pt")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for img_batch, _, _ in loader:
        img = torch.stack(img_batch).to(device)  # (B, 1, H, W)
        B, _, H, W = img.shape
        t = torch.rand(B, device=device)  # [0, 1)
        sigma = get_sigma(t).view(B, 1, 1, 1)

        noise = sigma * torch.randn((B, 2, H, W), device=device)  # 2 channel noise
        noisy = img.repeat(1, 2, 1, 1) + noise  # expand to 2 channels for matching
        score_target = -noise / sigma

        pred = model(noisy, t.unsqueeze(1))
        loss = F.mse_loss(pred, score_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    if epoch == EPOCHS - 1:
        with torch.no_grad():
            model.eval()
            for img_batch, _, _ in loader:
                img = torch.stack(img_batch).to(device)
                B, _, H, W = img.shape
                t = torch.zeros(B, device=device)
                img2 = img.repeat(1, 2, 1, 1)
                pred = model(img2, t.unsqueeze(1))
                torch.save(pred.cpu(), out_path)
                print(f"Saved final 2D score vector field to {out_path}")
                break

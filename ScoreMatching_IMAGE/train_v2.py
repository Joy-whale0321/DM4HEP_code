# === train_v2.py ===
import torch
from torch.utils.data import DataLoader
from data_v2 import PxPyImageDataset
from model_v2 import ScoreNetImage
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt  # ✅ 用于保存 score 图像
from pathlib import Path
import numpy as np

from Datacheck import save_example_events
from Datacheck import save_noisy_examples

# ---- HYPERPARAMETERS ----
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
SIGMA_MIN = 0.1
SIGMA_MAX = 4.0

# def get_sigma(t):
#     return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

def get_sigma(t):
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PxPyImageDataset(
    "/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root",
    img_size=(32, 32), px_range=(-3, 3), py_range=(-3, 3)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# save_example_events(dataset, out_dir="event_images", every_n=100)

model = ScoreNetImage(time_emb_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

out_path = Path("score_vector_output.pt")

# 创建保存目录
Path("score_vis").mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    # if epoch % 10 == 0:
    #     save_noisy_examples(dataset, get_sigma, device, out_file=f"event_images/noisy_epoch{epoch}.pdf")

    for img_batch, _, _ in loader:
        img = torch.stack(img_batch).to(device)  # (B, 1, H, W)
        B, _, H, W = img.shape
        # t = torch.rand(B, device=device)
        t = torch.ones(B, device=device) * 0.3
        sigma = get_sigma(t).view(B, 1, 1, 1)

        noise = sigma * torch.randn((B, 1, H, W), device=device)
        noisy = img + noise
        score_target = -noise / sigma # ??? -z/sigma ,  torch.randn((B, 1, H, W) / sigma --- noise / sigma^2

        pred = model(noisy, t.unsqueeze(1)) # ??? t:(B,)->(B,1)
        # loss = F.mse_loss(pred, score_target)
        # == 前景加权 ==
        with torch.no_grad():
            foreground = (img > 0).float()  # shape: [B,1,H,W]
            weights = 1.0 + 2.0 * foreground  # 前景加 2 倍权重

        loss = ((pred - score_target) ** 2 * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        ###### see see score ######
        # print("img max:", img.max().item(), "min:", img.min().item())
        # print("sigma:", sigma.view(-1)[0].item())
        # print("noise mean:", noise.mean().item(), "std:", noise.std().item())
        # print("score_target mean:", score_target.mean().item(), "std:", score_target.std().item())

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].imshow(score_target[0, 0].detach().cpu().numpy(), cmap="coolwarm", origin="lower")
        axs[0].set_title(f"Score Target t=0.3 (Epoch {epoch+1})")

        axs[1].imshow(pred[0, 0].detach().cpu().numpy(), cmap="coolwarm", origin="lower")
        axs[1].set_title(f"Score Pred t=0.3 (Epoch {epoch+1})")

        plt.tight_layout()
        plt.savefig(f"score_compare/score_epoch{epoch+1}.png")
        plt.close()
        # break

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # === 每100个 epoch 保存 score_pred 图像和模型 ===
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"scoreNET/score_model_epoch{epoch+1}.pth")

        model.eval()
        with torch.no_grad():
            for img_batch, _, _ in loader:
                img = torch.stack(img_batch).to(device)
                B, _, H, W = img.shape
                t_vis = torch.ones(B, device=device) * 0.3
                sigma = get_sigma(t_vis).view(B, 1, 1, 1)
                noise = sigma * torch.randn_like(img)
                noisy = img + noise
                score_pred = model(noisy, t_vis.unsqueeze(1))
                score_vis = score_pred[0, 0].cpu().numpy()

                plt.figure(figsize=(4, 4))
                plt.imshow(score_vis, cmap='coolwarm', origin='lower')
                plt.colorbar()
                plt.title(f"ScorePred Epoch {epoch+1}")
                plt.tight_layout()
                plt.savefig(f"score_vis/score_epoch{epoch+1}.png")
                plt.close()
                break

# === 最后一轮保存 score vector ===
if EPOCHS > 0:
    with torch.no_grad():
        model.eval()
        for img_batch, _, _ in loader:
            img = torch.stack(img_batch).to(device)
            B, _, H, W = img.shape
            t = torch.zeros(B, device=device)
            pred = model(img, t.unsqueeze(1))
            torch.save(pred.cpu(), out_path)
            print(f"Saved final 1D score field to {out_path}")
            break

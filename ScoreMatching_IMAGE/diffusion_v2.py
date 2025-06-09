# === sample_v1.py ===
import torch
import matplotlib.pyplot as plt
from model_v2 import ScoreNetImage
import os
from pathlib import Path

# ---------------- CONFIG ----------------
BATCH_SIZE = 1
IMG_SIZE = (32, 32)
STEPS = 20                # 扩散步数，从 t=1.0 到 t≈0
DT = 1e-3                 # Langevin 步长
SIGMA_MIN = 0.1
SIGMA_MAX = 4.0
SAVE_DIR = "sample_output"
MODEL_PATH = "scoreNET/score_model_epoch1000.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------- Load Model ----------------
model = ScoreNetImage(time_emb_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- Noise schedule (exponential) ----------
# def get_sigma(t):
#     return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

def get_sigma(t):
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)

# ---------- Initial image (pure noise) ----------
x = torch.randn(BATCH_SIZE, 1, *IMG_SIZE, device=device)

# ---------- Sampling steps ----------
t_vals = torch.linspace(1.0, 1e-3, STEPS)
snapshots = []

with torch.no_grad():
    for step, t in enumerate(t_vals):
        t_tensor = t.expand(BATCH_SIZE).to(device)
        sigma_t = get_sigma(t).to(device)
        
        for _ in range(50):  # Inner Langevin iterations per t step
            score = model(x, t_tensor.unsqueeze(1))  # predict score

            print("mean(abs(score)):", score.abs().mean().item())

            noise = torch.randn_like(x)
            x = x + (sigma_t ** 2) * score * DT + sigma_t * torch.sqrt(torch.tensor(DT, device=device)) * noise

        snapshots.append(x[0, 0].cpu().clone())

# ---------- Post-process last image ----------
x_log = x[0, 0].cpu()
# x_recon = torch.expm1(x_log).clamp(min=0)
# x_int = torch.poisson(x_recon)  # Optional: convert to integer count

# ---------- Save visualization ----------
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, len(snapshots), figsize=(3 * len(snapshots), 3))

for i, img in enumerate(snapshots):
    axes[i].imshow(img.numpy(), cmap="inferno", origin="lower", aspect="equal", vmin=0, vmax=3)
    axes[i].set_title(f"Step {i}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/diffusion_process.pdf")
plt.close()

# Save final result (optional)
plt.figure(figsize=(5, 5))
plt.imshow(x_log.numpy(), cmap="inferno", origin="lower", aspect="equal")
plt.title("Final Sampled Image (Poisson)")
plt.colorbar(label="count")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_image_poisson.pdf")
plt.close()

print(f"Saved sampled images to {SAVE_DIR}/")

# === diffusion_v2.py ===
import torch
from model_v2 import ScoreNetImage
import matplotlib.pyplot as plt
import os

# ---- CONFIG ----
out_dir = "samples"
os.makedirs(out_dir, exist_ok=True)

BATCH_SIZE = 16
IMG_SIZE = (60, 60)
T_STEPS = 1000
SIGMA_MIN = 0.5
SIGMA_MAX = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load Model ----
model = ScoreNetImage(time_emb_dim=128).to(device)
model.load_state_dict(torch.load("scoreNET/score_model_epoch100.pth", map_location=device))
model.eval()

# ---- Noise schedule ----
def get_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

# ---- Sampling ----
with torch.no_grad():
    # initial noise
    x = torch.randn(BATCH_SIZE, 1, *IMG_SIZE, device=device) * SIGMA_MAX
    t_steps = torch.linspace(1.0, 1e-5, T_STEPS, device=device)
    dt = 1.0 / T_STEPS

    for i, t in enumerate(t_steps):
        t_batch = t.expand(BATCH_SIZE)
        sigma = get_sigma(t).view(1, 1, 1, 1)
        score = model(x, t_batch.unsqueeze(1))

        noise = torch.randn_like(x)
        x = x + sigma**2 * score * dt + (2 * sigma**2 * dt).sqrt() * noise

        if (i + 1) % 200 == 0 or i == T_STEPS - 1:
            for j in range(BATCH_SIZE):
                img = x[j, 0].cpu().numpy()
                plt.imshow(img, cmap='hot', extent=[-3, 3, -3, 3])
                plt.colorbar()
                plt.title(f"Sample {j} at step {i+1}")
                plt.savefig(f"{out_dir}/sample_{j}_step_{i+1}.png")
                plt.close()
    print(f"Sampling finished. Images saved in '{out_dir}/'")

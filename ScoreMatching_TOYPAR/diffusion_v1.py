import torch
import numpy as np
import uproot
import awkward as ak
from model_v1 import ScoreNetParticle
import os

# ---- Config ----
NUM_SAMPLES = 100000
NUM_STEPS = 100
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0
OUT_ROOT = "sampled_output.root"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Noise Schedule ----
def get_sigma_schedule(num_steps):
    return torch.linspace(SIGMA_MAX, SIGMA_MIN, num_steps + 1)  # 逐步降低噪声

# ---- Load Trained Model ----
model = ScoreNetParticle(input_dim=2).to(device)
model.load_state_dict(torch.load("checkpoints/scorenet_epoch40.pt"))  # 确保之前训练后保存为 model.pth
model.eval()

# ---- Sampling ----
x = torch.randn(NUM_SAMPLES, 2, device=device) * SIGMA_MAX  # 初始为纯噪声
sigmas = get_sigma_schedule(NUM_STEPS).to(device)

for i in range(NUM_STEPS):
    t = torch.full((NUM_SAMPLES, 1), i / NUM_STEPS, device=device)
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]

    with torch.no_grad():
        score = model(x, t)  # 预测分数 ∇log p(x_t)
    
    # Euler-Maruyama 近似采样：反扩散
    dt = sigma_next - sigma
    x = x + score * dt + torch.randn_like(x) * torch.sqrt(torch.abs(dt))

# ---- Compute pt = sqrt(px^2 + py^2) ----
x_np = x.cpu().numpy()
px = x_np[:, 0]
py = x_np[:, 1]
pt = np.sqrt(px ** 2 + py ** 2)

# ---- Save to ROOT ----
with uproot.recreate(OUT_ROOT) as f:
    f["tree"] = {
        "px": px,
        "py": py,
        "pt": pt,
    }

print(f"Sampling complete! Results saved to {OUT_ROOT}")

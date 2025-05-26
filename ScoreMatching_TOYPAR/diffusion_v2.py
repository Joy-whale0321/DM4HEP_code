import torch
import numpy as np
import uproot
import os
from model_v1 import ScoreNetParticle

# ---- Config ----
NUM_SAMPLES = 10000
NUM_STEPS = 500
DT = 1e-4
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t

# ---- Load trained model ----
model = ScoreNetParticle(input_dim=2).to(device)
model.load_state_dict(torch.load("checkpoints_v2/scorenet_epoch90.pt"))
model.eval()

# ---- Initialization ----
t0 = torch.tensor(1.0, device=device)
sigma0 = get_sigma(t0)
x = torch.randn(NUM_SAMPLES, 2, device=device) * sigma0

# ---- Sampling steps ----
timesteps = torch.linspace(1.0, 1e-3, NUM_STEPS, device=device)

for i in range(NUM_STEPS):
    t = timesteps[i].expand(NUM_SAMPLES, 1)
    sigma = get_sigma(t)
    with torch.no_grad():
        score = model(x, t)
    noise = torch.randn_like(x)
    x = x + DT * score + torch.sqrt(torch.tensor(2 * DT)) * noise

# ---- Save to ROOT ----
x_np = x.cpu().numpy()
px = x_np[:, 0]
py = x_np[:, 1]
pt = np.sqrt(px**2 + py**2)

with uproot.recreate("langevin_sampled_output.root") as f:
    f["tree"] = {
        "px": px,
        "py": py,
        "pt": pt,
    }

print("Sampling complete! Saved to 'langevin_sampled_output.root'")

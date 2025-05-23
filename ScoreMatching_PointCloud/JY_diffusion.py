import torch
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
import os

from JY_model_v3 import ScoreNet

# === Settings ===
NUM_EVENTS = 10
NUM_PARTICLES = 1000
BATCH_SIZE = 1
NUM_STEPS = 100
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/scorenet_epoch20.pt"
MEAN_FILE = "pxpy_mean.npy"
STD_FILE = "pxpy_std.npy"
OUTPUT_ROOT = "GeneratedEvents.root"
TREE_NAME = "T_Generated"

# === Noise Schedule ===
def get_sigma(t):
    return torch.tensor(SIGMA_MIN, device=t.device) * ((SIGMA_MAX / SIGMA_MIN) ** t)

def get_sigma_schedule(num_steps):
    t_vals = torch.linspace(1, 0, num_steps + 1, device=DEVICE)  # t 从 1 降到 0
    return get_sigma(t_vals)

# === Load model ===
model = ScoreNet(time_emb_dim=256, point_feat_dim=256, hidden_dim=512).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# === Load normalization stats ===
mean = torch.tensor(np.load(MEAN_FILE), device=DEVICE).view(1, 1, 2)
std = torch.tensor(np.load(STD_FILE), device=DEVICE).view(1, 1, 2)

print("mean from file:", mean.cpu().numpy())
print("std from file:", std.cpu().numpy())

# === Sampling ===
def sample_batch_sde(total_events=NUM_EVENTS, n_particles=NUM_PARTICLES, steps=NUM_STEPS):
    sigma_sched = get_sigma_schedule(steps)
    all_samples = []

    dt = 1.0 / steps  # 每步时间长度

    with torch.no_grad():
        for start in tqdm(range(0, total_events, BATCH_SIZE), desc="Sampling SDE"):
            cur_batch_size = min(BATCH_SIZE, total_events - start)
            x = torch.randn(cur_batch_size, n_particles, 2, device=DEVICE) * sigma_sched[0]

            # 假设 sigma_sched 长度为 steps+1，从 σ_max 递减到 σ_min
            for i in range(steps):
                # 计算方差差
                var = sigma_sched[i]**2 - sigma_sched[i+1]**2  # >0

                # 构造时间刻度 t
                t = 1.0 - i/steps
                t_cur = torch.full((cur_batch_size,), t, device=DEVICE)

                # 预测 score
                score = model(
                  x.view(-1,2),
                  t_cur.repeat_interleave(n_particles)
                ).view(cur_batch_size, n_particles, 2)

                # 采样噪声
                noise = torch.randn_like(x)

                # 计算 drift 和 diffusion
                # drift 有的文献带 0.5，有的不带，要看你训练时 score 的定义
                drift     = -var * score          # 或者 -0.5 * var * score
                diffusion = torch.sqrt(var) * noise

                x = x + drift + diffusion

            x_real = x * std + mean
            # x_real = x
            all_samples.append(x_real.detach().cpu())

    return torch.cat(all_samples, dim=0).cpu().numpy()

# === Main ===
print("Sampling...")
sampled_data = sample_batch_sde()

# === Save to ROOT ===
print(f"Saving to ROOT file: {OUTPUT_ROOT}")
px_array = ak.Array([event[:, 0].tolist() for event in sampled_data])
py_array = ak.Array([event[:, 1].tolist() for event in sampled_data])
# pz_array = ak.Array([event[:, 2].tolist() for event in sampled_data])
pt_array = np.sqrt(px_array**2 + py_array**2)

with uproot.recreate(OUTPUT_ROOT) as root_file:
    root_file[TREE_NAME] = {
        "px": px_array,
        "py": py_array,
        # "pz": pz_array
        "pt": pt_array
    }

print("Done.")

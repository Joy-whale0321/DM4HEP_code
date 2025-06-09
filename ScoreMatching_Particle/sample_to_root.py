# sample_to_root.py
import torch
import numpy as np
import uproot
import awkward as ak
import os
from score_model import ScoreNet

# ==== CONFIG ====
MODEL_PATH = "checkpoints/score_model_epoch100.pth"
SAVE_ROOT = "sampled_particles.root"
NUM_EVENTS = 100
NUM_PARTICLES = 1000   # 每个 event 中的粒子数
STEPS = 2000             # Langevin 迭代步数
EPS = 1e-4            # 步长
SIGMA_MAX = 1.0       # 初始噪声幅度（如训练中一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 载入模型 ====
model = ScoreNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== Langevin 采样函数 ====
def langevin_sample(N):
    x = torch.randn(N, 2, device=DEVICE) * SIGMA_MAX  # 初始高斯噪声

    for step in range(STEPS):
        t_val = torch.ones(N, 1, device=DEVICE) * (1.0 - step / STEPS)  # 线性下降时间
        with torch.no_grad():
            score = model(x, t_val)
        noise = torch.randn_like(x)
        x = x + EPS * score + (2 * EPS) ** 0.5 * noise

    return x.detach().cpu().numpy()  # shape: [N, 2]

# ==== 开始采样 ====
all_px = []
all_py = []
all_pt = []

for i in range(NUM_EVENTS):
    samples = langevin_sample(NUM_PARTICLES)  # shape: [NUM_PARTICLES, 2]
    px, py = samples[:, 0], samples[:, 1]
    pt = np.sqrt(px ** 2 + py ** 2)
    all_px.append(px)
    all_py.append(py)
    all_pt.append(pt)

# ==== 保存为 ROOT 文件 ====
out_dict = {
    "px": ak.Array(all_px),
    "py": ak.Array(all_py),
    "pt": ak.Array(all_pt),
}

with uproot.recreate(SAVE_ROOT) as f:
    f["T"] = out_dict

print(f"Saved to {SAVE_ROOT}")

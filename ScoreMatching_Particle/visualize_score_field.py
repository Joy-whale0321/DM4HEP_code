# visualize_score_field.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from score_model import ScoreNet
import os

# === CONFIG ===
MODEL_PATH = "checkpoints/score_model_epoch10.pth"
GRID_SIZE = 40
PX_RANGE = [-3, 3]
PY_RANGE = [-3, 3]
T_VALUES = np.linspace(0.1, 1, 10)  # np.linspace(start, stop, num)
SAVE_DIR = "score_field_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 载入模型 ===
model = ScoreNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === 构造 2D 网格 ===
px = np.linspace(PX_RANGE[0], PX_RANGE[1], GRID_SIZE)
py = np.linspace(PY_RANGE[0], PY_RANGE[1], GRID_SIZE)
xx, yy = np.meshgrid(px, py)
positions = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape: [GRID_SIZE², 2]
x = torch.tensor(positions, dtype=torch.float32).to(DEVICE)  # [N, 2]

# === 遍历多个 t 值 ===
for t_val in T_VALUES:
    t = torch.ones(x.size(0), 1, device=DEVICE) * t_val

    with torch.no_grad():
        score = model(x, t).cpu().numpy()  # [N, 2]

    U = score[:, 0].reshape(GRID_SIZE, GRID_SIZE)
    V = score[:, 1].reshape(GRID_SIZE, GRID_SIZE)

    plt.figure(figsize=(6, 6))
    plt.quiver(xx, yy, U, V, scale=50)
    plt.title(f"Score Vector Field (t={t_val:.2f})")
    plt.xlabel("px")
    plt.ylabel("py")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/score_vector_t_{t_val:.2f}.png")
    plt.close()

print(f"Score vector fields saved to {SAVE_DIR}")

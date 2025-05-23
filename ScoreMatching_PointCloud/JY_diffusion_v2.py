import torch
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
import os

from JY_model_v4 import ScoreNet

# === Settings ===
NUM_EVENTS = 10
NUM_PARTICLES = 1600
BATCH_SIZE = 1
NUM_STEPS = 100
SIGMA_MIN = 0.1
SIGMA_MAX = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/scorenet_epoch30.pt"
MEAN_FILE = "pxpy_mean.npy"
STD_FILE = "pxpy_std.npy"
OUTPUT_ROOT = "GeneratedEvents.root"
TREE_NAME = "T_Generated"

# === Noise Schedule ===
def get_sigma_schedule(num_steps):
    return torch.linspace(SIGMA_MAX, SIGMA_MIN, num_steps + 1, device=DEVICE)

# === Axis tagging ===
def reshape_with_axis_tags(x):
    N = x.shape[0]
    device = x.device
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    values = torch.cat([px, py, pz], dim=0).unsqueeze(1)
    tag_px = torch.tensor([1, 0, 0], device=device).repeat(N, 1)
    tag_py = torch.tensor([0, 1, 0], device=device).repeat(N, 1)
    tag_pz = torch.tensor([0, 0, 1], device=device).repeat(N, 1)
    tags = torch.cat([tag_px, tag_py, tag_pz], dim=0)
    return torch.cat([values, tags], dim=1)

# === Load model ===
model = ScoreNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# === Load normalization stats ===
mean = torch.tensor(np.load(MEAN_FILE), device=DEVICE).view(1, 1, 3)
std = torch.tensor(np.load(STD_FILE), device=DEVICE).view(1, 1, 3)

# === Sampling ===
def sample_batch(total_events=NUM_EVENTS, n_particles=NUM_PARTICLES, steps=NUM_STEPS):
    sigma_sched = get_sigma_schedule(steps)
    all_samples = []

    with torch.no_grad():
        for start in tqdm(range(0, total_events, BATCH_SIZE), desc="Sampling"):
            cur_batch_size = min(BATCH_SIZE, total_events - start)
            x = torch.randn(cur_batch_size, n_particles, 3, device=DEVICE) * sigma_sched[0]

            for i in range(steps):
                t_cur = torch.full((cur_batch_size,), (steps - i) / steps, device=DEVICE)
                sigma = sigma_sched[i].view(1, 1, 1)

                x_flat = x.view(-1, 3)
                x_tagged = reshape_with_axis_tags(x_flat)  # (3N, 4)
                score_flat = model(x_tagged, t_cur.repeat_interleave(n_particles))  # (3N,)
                score = score_flat.view(3, -1).transpose(0, 1).view(cur_batch_size, n_particles, 3)

                x = x - sigma**2 * score

            x_real = x * std + mean
            # x_real = x
            all_samples.append(x_real.detach().cpu())

    return torch.cat(all_samples, dim=0).cpu().numpy()

# === Main ===
print("Sampling...")
sampled_data = sample_batch()

print(f"Saving to ROOT file: {OUTPUT_ROOT}")
px_array = ak.Array([event[:, 0].tolist() for event in sampled_data])
py_array = ak.Array([event[:, 1].tolist() for event in sampled_data])
pz_array = ak.Array([event[:, 2].tolist() for event in sampled_data])

with uproot.recreate(OUTPUT_ROOT) as root_file:
    root_file[TREE_NAME] = {
        "px": px_array,
        "py": py_array,
        "pz": pz_array
    }

print("Done.")

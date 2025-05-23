import torch
import uproot
import awkward as ak
import numpy as np
from torch.utils.data import DataLoader
from JY_data import ParticlesPointDataset, my_collate_fn

# 参数
INPUT_FILE = "/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root"
OUTPUT_FILE = "checkfile/AferNoise.root"
BATCH_SIZE = 1
NUM_STEPS = 100
SIGMA_MIN = 0.05
SIGMA_MAX = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sigma schedule 与训练时一致（线性 schedule）
def get_sigma_schedule(num_steps):
    return torch.linspace(SIGMA_MAX, SIGMA_MIN, num_steps + 1, device=DEVICE)

sigma_sched = get_sigma_schedule(NUM_STEPS)

# 数据加载
dataset = ParticlesPointDataset(INPUT_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

# 生成 t 的列表（避免 t=0 造成数值不稳定）
t_list = torch.linspace(0.0001, 0.9999, 10)

# 开始写入
with uproot.recreate(OUTPUT_FILE) as f:
    for i, t in enumerate(t_list):
        px_noised_list = []
        py_noised_list = []

        t_val = t.item()
        t_idx = int((1 - t_val) * NUM_STEPS)
        sigma = sigma_sched[t_idx].view(1, 1)

        for batch in loader:
            for event in batch:
                x = event.to(DEVICE)  # (N, 2)
                eps = torch.randn_like(x)
                x_t = x + sigma * eps

                px_noised_list.append(x_t[:, 0].cpu().numpy().tolist())
                py_noised_list.append(x_t[:, 1].cpu().numpy().tolist())

        # 转换并写入当前 t 的 Tree
        px_array = ak.Array(px_noised_list)
        py_array = ak.Array(py_noised_list)
        pt_array = np.sqrt(px_array ** 2 + py_array ** 2)

        tree_name = f"T_t{i}"
        f[tree_name] = {
            "px": px_array,
            "py": py_array,
            "pt": pt_array
        }

        print(f"✅ 写入完成: {tree_name} (t = {t_val:.4f}, sigma = {sigma.item():.4f})")

print(f"\n✅ 所有加噪 xt 已保存到：{OUTPUT_FILE}")

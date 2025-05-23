import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

# =============================
# ✨ 主轴对齐函数）
# =============================
def align_all_events(px_array, py_array):
    aligned_px_array = []
    aligned_py_array = []
    for px, py in zip(px_array, py_array):
        coords = np.stack([px, py], axis=1)
        mu = coords.mean(axis=0)
        coords_centered = coords - mu
        cov = np.cov(coords_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        R = eigvecs  # 主轴方向（正交旋转矩阵）
        aligned = coords_centered @ R  # 对齐主轴到 x 轴
        aligned_px_array.append(aligned[:, 0])
        aligned_py_array.append(aligned[:, 1])
    return aligned_px_array, aligned_py_array


# =============================
# 数据集定义
# =============================
class ParticlesPointDataset(Dataset):
    def __init__(self, rootfile_path):
        file = uproot.open(rootfile_path)
        tree = file["T_Final_Hadron_Dis"]

        # 提取 px, py
        px_array = tree["px"].arrays(library="ak")["px"]
        py_array = tree["py"].arrays(library="ak")["py"]

        px_array = tree["px"].arrays(library="ak")["px"]
        py_array = tree["py"].arrays(library="ak")["py"]

        # 把 awkward array 转为 list of np.array
        px_array_np = [np.array(evt, dtype=np.float32) for evt in px_array]
        py_array_np = [np.array(evt, dtype=np.float32) for evt in py_array]

        aligned_px_array, aligned_py_array = align_all_events(px_array_np, py_array_np)

        self.point_clouds = []
        for px_evt, py_evt in zip(aligned_px_array, aligned_py_array):
            px_evt = np.array(px_evt, dtype=np.float32)
            py_evt = np.array(py_evt, dtype=np.float32)
            points = np.stack([px_evt, py_evt], axis=-1)  # shape: (N, 2)
            self.point_clouds.append(points)

        # Step 1: 合并所有点，计算全局 mean 和 std
        all_points = np.concatenate(self.point_clouds, axis=0)  # shape: (total_particles, 2)
        self.mean = np.mean(all_points, axis=0)
        self.std = np.std(all_points, axis=0)

        # Step 2: 对所有事件做标准化
        self.point_clouds = [(pc - self.mean) / self.std for pc in self.point_clouds]

        # Step 3: 保存 mean 和 std（供生成样本反归一化使用）
        np.save("pxpy_mean.npy", self.mean)
        np.save("pxpy_std.npy", self.std)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx]  # (N, 2) np.array


# =============================
# 自定义 collate function
# =============================
def my_collate_fn(batch):
    return [torch.from_numpy(item) for item in batch]
